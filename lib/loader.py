import gc
from threading import Lock

import torch
from DeepCache import DeepCacheSDHelper
from diffusers.models import AutoencoderKL, AutoencoderTiny
from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0

from .config import Config
from .logger import Logger
from .upscaler import RealESRGAN
from .utils import safe_progress, timer


class Loader:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.pipe = None
                cls._instance.model = None
                cls._instance.upscaler = None
                cls._instance.ip_adapter = None
                cls._instance.log = Logger("Loader")
        return cls._instance

    @property
    def _is_kl_vae(self):
        if self.pipe is not None:
            vae_type = type(self.pipe.vae)
            return issubclass(vae_type, AutoencoderKL)
        return False

    @property
    def _is_tiny_vae(self):
        if self.pipe is not None:
            vae_type = type(self.pipe.vae)
            return issubclass(vae_type, AutoencoderTiny)
        return False

    @property
    def _has_freeu(self):
        if self.pipe is not None:
            attrs = ["b1", "b2", "s1", "s2"]
            block = self.pipe.unet.up_blocks[0]
            return all(getattr(block, attr, None) is not None for attr in attrs)
        return False

    def _should_unload_upscaler(self, scale=1):
        if self.upscaler is not None and self.upscaler.scale != scale:
            return True
        return False

    def _should_unload_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipe, "deepcache")
        if has_deepcache and interval == 1:
            return True
        if has_deepcache and self.pipe.deepcache.params["cache_interval"] != interval:
            return True
        return False

    def _should_unload_freeu(self, freeu=False):
        if self._has_freeu and not freeu:
            return True
        return False

    def _should_unload_ip_adapter(self, model="", ip_adapter=""):
        # unload if model changed
        if self.model and self.model.lower() != model.lower():
            return True
        if self.ip_adapter and not ip_adapter:
            return True
        return False

    def _should_unload_pipeline(self, kind="", model=""):
        if self.pipe is None:
            return False
        if self.model.lower() != model.lower():
            return True
        if kind == "txt2img" and not isinstance(self.pipe, Config.PIPELINES["txt2img"]):
            return True  # txt2img -> img2img
        if kind == "img2img" and not isinstance(self.pipe, Config.PIPELINES["img2img"]):
            return True  # img2img -> txt2img
        return False

    def _unload_upscaler(self):
        if self.upscaler is not None:
            with timer(f"Unloading {self.upscaler.scale}x upscaler", logger=self.log.info):
                self.upscaler.to("cpu")

    def _unload_deepcache(self):
        if self.pipe.deepcache is not None:
            self.log.info("Disabling DeepCache")
            self.pipe.deepcache.disable()
            delattr(self.pipe, "deepcache")

    def _unload_freeu(self, freeu=False):
        if self._has_freeu and not freeu:
            self.log.info("Disabling FreeU")
            self.pipe.disable_freeu()

    # Copied from https://github.com/huggingface/diffusers/blob/v0.28.0/src/diffusers/loaders/ip_adapter.py#L300
    def _unload_ip_adapter(self):
        if self.ip_adapter is not None:
            with timer("Unloading IP-Adapter", logger=self.log.info):
                if not isinstance(self.pipe, Config.PIPELINES["img2img"]):
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

    def _unload_pipeline(self):
        if self.pipe is not None:
            with timer(f"Unloading {self.model}", logger=self.log.info):
                self.pipe.to("cpu")

    def _unload(self, kind="", model="", ip_adapter="", deepcache=1, scale=1, freeu=False):
        to_unload = []
        if self._should_unload_deepcache(deepcache):  # remove deepcache first
            self._unload_deepcache()

        if self._should_unload_freeu(freeu):
            self._unload_freeu()

        if self._should_unload_upscaler(scale):
            self._unload_upscaler()
            to_unload.append("upscaler")

        if self._should_unload_ip_adapter(model, ip_adapter):
            self._unload_ip_adapter()
            to_unload.append("ip_adapter")

        if self._should_unload_pipeline(kind, model):
            self._unload_pipeline()
            to_unload.append("model")
            to_unload.append("pipe")

        self.collect()
        for component in to_unload:
            setattr(self, component, None)
            gc.collect()

    def _should_load_upscaler(self, scale=1):
        if self.upscaler is None and scale > 1:
            return True
        return False

    def _should_load_freeu(self, freeu=False):
        if not self._has_freeu and freeu:
            return True
        return False

    def _should_load_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipe, "deepcache")
        if not has_deepcache and interval != 1:
            return True
        if has_deepcache and self.pipe.deepcache.params["cache_interval"] != interval:
            return True
        return False

    def _should_load_ip_adapter(self, ip_adapter=""):
        if not self.ip_adapter and ip_adapter:
            return True
        return False

    def _should_load_pipeline(self):
        if self.pipe is None:
            return True
        return False

    def _load_upscaler(self, scale=1):
        if self._should_load_upscaler(scale):
            try:
                msg = f"Loading {scale}x upscaler"
                with timer(msg, logger=self.log.info):
                    self.upscaler = RealESRGAN(scale, device=self.pipe.device)
                    self.upscaler.load_weights()
            except Exception as e:
                self.log.error(f"Error loading {scale}x upscaler: {e}")
                self.upscaler = None

    def _load_deepcache(self, interval=1):
        if self._should_load_deepcache(interval):
            self.log.info("Enabling DeepCache")
            self.pipe.deepcache = DeepCacheSDHelper(self.pipe)
            self.pipe.deepcache.set_params(cache_interval=interval)
            self.pipe.deepcache.enable()

    # https://github.com/ChenyangSi/FreeU
    def _load_freeu(self, freeu=False):
        if self._should_load_freeu(freeu):
            self.log.info("Enabling FreeU")
            self.pipe.enable_freeu(b1=1.5, b2=1.6, s1=0.9, s2=0.2)

    def _load_ip_adapter(self, ip_adapter=""):
        if self._should_load_ip_adapter(ip_adapter):
            msg = "Loading IP-Adapter"
            with timer(msg, logger=self.log.info):
                self.pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name=f"ip-adapter-{ip_adapter}_sd15.safetensors",
                )
                # 50% works the best
                self.pipe.set_ip_adapter_scale(0.5)
                self.ip_adapter = ip_adapter

    def _load_pipeline(
        self,
        kind,
        model,
        progress,
        **kwargs,
    ):
        pipeline = Config.PIPELINES[kind]
        if self._should_load_pipeline():
            try:
                with timer(f"Loading {model} ({kind})", logger=self.log.info):
                    self.model = model
                    if model.lower() in Config.MODEL_CHECKPOINTS.keys():
                        self.pipe = pipeline.from_single_file(
                            f"https://huggingface.co/{model}/{Config.MODEL_CHECKPOINTS[model.lower()]}",
                            progress,
                            **kwargs,
                        ).to("cuda")
                    else:
                        self.pipe = pipeline.from_pretrained(model, progress, **kwargs).to("cuda")
            except Exception as e:
                self.log.error(f"Error loading {model}: {e}")
                self.model = None
                self.pipe = None
                return
        if not isinstance(self.pipe, pipeline):
            self.pipe = pipeline.from_pipe(self.pipe).to("cuda")
        if self.pipe is not None:
            self.pipe.set_progress_bar_config(disable=progress is not None)

    def _load_vae(self, taesd=False, model=""):
        # by default all models use KL
        if self._is_kl_vae and taesd:
            msg = "Loading Tiny VAE"
            with timer(msg, logger=self.log.info):
                self.pipe.vae = AutoencoderTiny.from_pretrained(
                    pretrained_model_name_or_path="madebyollin/taesd",
                    torch_dtype=self.pipe.dtype,
                ).to(self.pipe.device)
            return

        if self._is_tiny_vae and not taesd:
            msg = "Loading KL VAE"
            with timer(msg, logger=self.log.info):
                if model.lower() in Config.MODEL_CHECKPOINTS.keys():
                    self.pipe.vae = AutoencoderKL.from_single_file(
                        f"https://huggingface.co/{model}/{Config.MODEL_CHECKPOINTS[model.lower()]}",
                        torch_dtype=self.pipe.dtype,
                    ).to(self.pipe.device)
                else:
                    self.pipe.vae = AutoencoderKL.from_pretrained(
                        pretrained_model_name_or_path=model,
                        torch_dtype=self.pipe.dtype,
                        subfolder="vae",
                        variant="fp16",
                    ).to(self.pipe.device)

    def collect(self):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def load(
        self,
        kind,
        ip_adapter,
        model,
        scheduler,
        deepcache,
        scale,
        karras,
        taesd,
        freeu,
        progress,
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

        self._unload(kind, model, ip_adapter, deepcache, scale, freeu)
        self._load_pipeline(kind, model, progress, **pipe_kwargs)

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
                self.log.info(f"Enabling {scheduler} scheduler")
            if not same_karras:
                self.log.info(f"{'Enabling' if karras else 'Disabling'} Karras sigmas")
            if not same_scheduler or not same_karras:
                self.pipe.scheduler = Config.SCHEDULERS[scheduler](**scheduler_kwargs)

        CURRENT_STEP = 1
        TOTAL_STEPS = sum(
            [
                self._is_kl_vae and taesd,
                self._is_tiny_vae and not taesd,
                self._should_load_freeu(freeu),
                self._should_load_deepcache(deepcache),
                self._should_load_ip_adapter(ip_adapter),
                self._should_load_upscaler(scale),
            ]
        )

        desc = "Configuring pipeline"
        if not self._has_freeu and freeu:
            self._load_freeu(freeu)
            safe_progress(progress, CURRENT_STEP, TOTAL_STEPS, desc)
            CURRENT_STEP += 1

        if self._should_load_deepcache(deepcache):
            self._load_deepcache(deepcache)
            safe_progress(progress, CURRENT_STEP, TOTAL_STEPS, desc)
            CURRENT_STEP += 1

        if self._should_load_ip_adapter(ip_adapter):
            self._load_ip_adapter(ip_adapter)
            safe_progress(progress, CURRENT_STEP, TOTAL_STEPS, desc)
            CURRENT_STEP += 1

        if self._should_load_upscaler(scale):
            self._load_upscaler(scale)
            safe_progress(progress, CURRENT_STEP, TOTAL_STEPS, desc)
            CURRENT_STEP += 1

        if self._is_kl_vae and taesd or self._is_tiny_vae and not taesd:
            self._load_vae(taesd, model)
            safe_progress(progress, CURRENT_STEP, TOTAL_STEPS, desc)
