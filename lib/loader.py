import gc
from threading import Lock
from warnings import filterwarnings

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, AutoencoderTiny
from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0
from torch._dynamo import OptimizedModule

from .config import Config
from .upscaler import RealESRGAN

__import__("diffusers").logging.set_verbosity_error()
filterwarnings("ignore", category=FutureWarning, module="torch")
filterwarnings("ignore", category=FutureWarning, module="diffusers")


class Loader:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.pipe = None
                cls._instance.model = None
                cls._instance.ip_adapter = None
                cls._instance.upscaler_2x = None
                cls._instance.upscaler_4x = None
        return cls._instance

    def _should_unload_ip_adapter(self, ip_adapter=""):
        return self.ip_adapter is not None and not ip_adapter

    def _should_unload_pipeline(self, kind="", model=""):
        if self.pipe is None:
            return False
        if self.model.lower() != model.lower():
            return True
        if kind == "txt2img" and not isinstance(self.pipe, StableDiffusionPipeline):
            return True  # txt2img -> img2img
        if kind == "img2img" and not isinstance(self.pipe, StableDiffusionImg2ImgPipeline):
            return True  # img2img -> txt2img
        return False

    # https://github.com/huggingface/diffusers/blob/v0.28.0/src/diffusers/loaders/ip_adapter.py#L300
    def _unload_ip_adapter(self):
        print("Unloading IP Adapter...")
        if not isinstance(self.pipe, StableDiffusionImg2ImgPipeline):
            self.pipe.image_encoder = None
            self.pipe.register_to_config(image_encoder=[None, None])

        self.pipe.feature_extractor = None
        self.pipe.unet.encoder_hid_proj = None
        self.pipe.unet.config.encoder_hid_dim_type = None
        self.pipe.register_to_config(feature_extractor=[None, None])

        attn_procs = {}
        for name, value in self.pipe.unet.attn_processors.items():
            attn_processor_class = AttnProcessor2_0()  # raises if not torch 2
            attn_procs[name] = (
                attn_processor_class
                if isinstance(value, IPAdapterAttnProcessor2_0)
                else value.__class__()
            )
        self.pipe.unet.set_attn_processor(attn_procs)

    def _flush(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def _unload(self, kind="", model="", ip_adapter=""):
        to_unload = []
        if self._should_unload_ip_adapter(ip_adapter):
            self._unload_ip_adapter()
            to_unload.append("ip_adapter")
        if self._should_unload_pipeline(kind, model):
            to_unload.append("model")
            to_unload.append("pipe")
        for component in to_unload:
            delattr(self, component)
        self._flush()
        for component in to_unload:
            setattr(self, component, None)

    def _load_ip_adapter(self, ip_adapter=""):
        if self.ip_adapter is None and ip_adapter:
            print(f"Loading IP Adapter: {ip_adapter}...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name=f"ip-adapter-{ip_adapter}_sd15.safetensors",
            )
            # 50% works the best
            self.pipe.set_ip_adapter_scale(0.5)
            self.ip_adapter = ip_adapter

    def _load_upscaler(self, scale=1):
        if scale == 2 and self.upscaler_2x is None:
            try:
                print("Loading 2x upscaler...")
                self.upscaler_2x = RealESRGAN(2, "cuda")
                self.upscaler_2x.load_weights()
            except Exception as e:
                print(f"Error loading 2x upscaler: {e}")
                self.upscaler_2x = None
        if scale == 4 and self.upscaler_4x is None:
            try:
                print("Loading 4x upscaler...")
                self.upscaler_4x = RealESRGAN(4, "cuda")
                self.upscaler_4x.load_weights()
            except Exception as e:
                print(f"Error loading 4x upscaler: {e}")
                self.upscaler_4x = None

    def _load_pipeline(self, kind, model, tqdm, **kwargs):
        pipeline = Config.PIPELINES[kind]
        if self.pipe is None:
            try:
                print(f"Loading {model}...")
                self.model = model
                if model.lower() in Config.MODEL_CHECKPOINTS.keys():
                    self.pipe = pipeline.from_single_file(
                        f"https://huggingface.co/{model}/{Config.MODEL_CHECKPOINTS[model.lower()]}",
                        **kwargs,
                    ).to("cuda")
                else:
                    self.pipe = pipeline.from_pretrained(model, **kwargs).to("cuda")
            except Exception as e:
                print(f"Error loading {model}: {e}")
                self.model = None
                self.pipe = None
                return
        if not isinstance(self.pipe, pipeline):
            self.pipe = pipeline.from_pipe(self.pipe).to("cuda")
        if self.pipe is not None:
            self.pipe.set_progress_bar_config(disable=not tqdm)

    def _load_vae(self, taesd=False, model=""):
        vae_type = type(self.pipe.vae)
        is_kl = issubclass(vae_type, (AutoencoderKL, OptimizedModule))
        is_tiny = issubclass(vae_type, AutoencoderTiny)

        # by default all models use KL
        if is_kl and taesd:
            print("Switching to Tiny VAE...")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                # can't compile tiny VAE
                pretrained_model_name_or_path="madebyollin/taesd",
                torch_dtype=self.pipe.dtype,
            ).to(self.pipe.device)
            return

        if is_tiny and not taesd:
            print("Switching to KL VAE...")
            if model.lower() in Config.MODEL_CHECKPOINTS.keys():
                vae = AutoencoderKL.from_single_file(
                    f"https://huggingface.co/{model}/{Config.MODEL_CHECKPOINTS[model.lower()]}",
                    torch_dtype=self.pipe.dtype,
                ).to(self.pipe.device)
            else:
                vae = AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path=model,
                    torch_dtype=self.pipe.dtype,
                    subfolder="vae",
                    variant="fp16",
                ).to(self.pipe.device)
            self.pipe.vae = torch.compile(
                mode="reduce-overhead",
                fullgraph=True,
                model=vae,
            )

    def _load_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipe, "deepcache")
        if has_deepcache and self.pipe.deepcache.params["cache_interval"] == interval:
            return
        if has_deepcache:
            self.pipe.deepcache.disable()
        else:
            self.pipe.deepcache = DeepCacheSDHelper(pipe=self.pipe)
        self.pipe.deepcache.set_params(cache_interval=interval)
        self.pipe.deepcache.enable()

    # https://github.com/ChenyangSi/FreeU
    def _load_freeu(self, freeu=False):
        block = self.pipe.unet.up_blocks[0]
        attrs = ["b1", "b2", "s1", "s2"]
        has_freeu = all(getattr(block, attr, None) is not None for attr in attrs)
        if has_freeu and not freeu:
            print("Disabling FreeU...")
            self.pipe.disable_freeu()
        elif not has_freeu and freeu:
            print("Enabling FreeU...")
            self.pipe.enable_freeu(b1=1.5, b2=1.6, s1=0.9, s2=0.2)

    def load(
        self,
        kind,
        ip_adapter,
        model,
        scheduler,
        karras,
        taesd,
        freeu,
        deepcache,
        scale,
        tqdm,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scheduler_kwargs = {
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "steps_offset": 1,
        }

        if scheduler not in ["DDIM", "Euler a", "PNDM"]:
            scheduler_kwargs["use_karras_sigmas"] = karras

        # https://github.com/huggingface/diffusers/blob/8a3f0c1/scripts/convert_original_stable_diffusion_to_diffusers.py#L939
        if scheduler == "DDIM":
            scheduler_kwargs["clip_sample"] = False
            scheduler_kwargs["set_alpha_to_one"] = False

        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "scheduler": Config.SCHEDULERS[scheduler](**scheduler_kwargs),
        }

        # diffusers fp16 variant
        if model.lower() not in Config.MODEL_CHECKPOINTS.keys():
            pipe_kwargs["variant"] = "fp16"
        else:
            pipe_kwargs["variant"] = None

        # convert fp32 to bf16 if possible
        if model.lower() in ["linaqruf/anything-v3-1"]:
            pipe_kwargs["torch_dtype"] = (
                torch.bfloat16
                if torch.cuda.get_device_properties(device).major >= 8
                else torch.float16
            )
        else:
            # defaults to float32
            pipe_kwargs["torch_dtype"] = torch.float16

        self._unload(kind, model, ip_adapter)
        self._load_pipeline(kind, model, tqdm, **pipe_kwargs)

        # error loading model
        if self.pipe is None:
            return

        same_scheduler = isinstance(self.pipe.scheduler, Config.SCHEDULERS[scheduler])
        same_karras = (
            not hasattr(self.pipe.scheduler.config, "use_karras_sigmas")
            or self.pipe.scheduler.config.use_karras_sigmas == karras
        )

        # same model, different scheduler
        if self.model.lower() == model.lower():
            if not same_scheduler:
                print(f"Switching to {scheduler}...")
            if not same_karras:
                print(f"{'Enabling' if karras else 'Disabling'} Karras sigmas...")
            if not same_scheduler or not same_karras:
                self.pipe.scheduler = Config.SCHEDULERS[scheduler](**scheduler_kwargs)

        self._load_freeu(freeu)
        self._load_vae(taesd, model)
        self._load_deepcache(deepcache)
        self._load_ip_adapter(ip_adapter)
        self._load_upscaler(scale)
