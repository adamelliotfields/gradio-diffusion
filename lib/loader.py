import torch
from DeepCache import DeepCacheSDHelper
from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from diffusers.models import AutoencoderKL, AutoencoderTiny
from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0
from torch._dynamo import OptimizedModule

from .upscaler import RealESRGAN

__import__("warnings").filterwarnings("ignore", category=FutureWarning, module="diffusers")

PIPELINES = {
    "txt2img": StableDiffusionPipeline,
    "img2img": StableDiffusionImg2ImgPipeline,
}


class Loader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.pipe = None
            cls._instance.upscaler = None
            cls._instance.ip_adapter = None
        return cls._instance

    def _should_unload_upscaler(self, scale=1):
        return self.upscaler is not None and scale == 1

    def _should_unload_ip_adapter(self, ip_adapter=None):
        return self.ip_adapter is not None and ip_adapter is None

    def _should_unload_pipeline(self, kind="", model=""):
        if self.pipe is None:
            return False
        if self.pipe.config._name_or_path.lower() != model.lower():
            return True
        if kind == "txt2img" and not isinstance(self.pipe, StableDiffusionPipeline):
            return True  # txt2img -> img2img
        if kind == "img2img" and not isinstance(self.pipe, StableDiffusionImg2ImgPipeline):
            return True  # img2img -> txt2img
        return False

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

    def _unload(self, kind="", model="", ip_adapter=None, scale=1):
        to_unload = []

        if self._should_unload_upscaler(scale):
            to_unload.append("upscaler")

        if self._should_unload_ip_adapter(ip_adapter):
            self._unload_ip_adapter()
            to_unload.append("ip_adapter")

        if self._should_unload_pipeline(kind, model):
            to_unload.append("pipe")

        for component in to_unload:
            if hasattr(self, component):
                delattr(self, component)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        for component in to_unload:
            setattr(self, component, None)

    def _load_ip_adapter(self, ip_adapter=None):
        if self.ip_adapter is None and ip_adapter is not None:
            print(f"Loading IP Adapter: {ip_adapter}...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name=f"ip-adapter-{ip_adapter}_sd15.safetensors",
            )
            # TODO: slider for ip_scale
            self.pipe.set_ip_adapter_scale(0.5)
            self.ip_adapter = ip_adapter

    def _load_upscaler(self, device=None, scale=1):
        if scale > 1 and self.upscaler is None:
            print(f"Loading {scale}x upscaler...")
            self.upscaler = RealESRGAN(device=device, scale=scale)
            self.upscaler.load_weights()

    def _load_pipeline(self, kind, model, taesd, device, **kwargs):
        pipeline = PIPELINES[kind]
        if self.pipe is None:
            print(f"Loading {model.lower()} with {'Tiny' if taesd else 'KL'} VAE...")
            self.pipe = pipeline.from_pretrained(model, **kwargs).to(device)
        if not isinstance(self.pipe, pipeline):
            self.pipe = pipeline.from_pipe(self.pipe).to(device)

    def _load_vae(self, taesd=False, model_name=None, variant=None):
        vae_type = type(self.pipe.vae)
        is_kl = issubclass(vae_type, (AutoencoderKL, OptimizedModule))
        is_tiny = issubclass(vae_type, AutoencoderTiny)

        # by default all models use KL
        if is_kl and taesd:
            # can't compile tiny VAE
            print("Switching to Tiny VAE...")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                pretrained_model_name_or_path="madebyollin/taesd",
                torch_dtype=self.pipe.dtype,
            ).to(self.pipe.device)
            return

        if is_tiny and not taesd:
            print("Switching to KL VAE...")
            model = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=model_name,
                torch_dtype=self.pipe.dtype,
                subfolder="vae",
                variant=variant,
            ).to(self.pipe.device)
            self.pipe.vae = torch.compile(
                mode="reduce-overhead",
                fullgraph=True,
                model=model,
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

    def _load_freeu(self, freeu=False):
        # https://github.com/huggingface/diffusers/blob/v0.30.0/src/diffusers/models/unets/unet_2d_condition.py
        block = self.pipe.unet.up_blocks[0]
        attrs = ["b1", "b2", "s1", "s2"]
        has_freeu = all(getattr(block, attr, None) is not None for attr in attrs)
        if has_freeu and not freeu:
            print("Disabling FreeU...")
            self.pipe.disable_freeu()
        elif not has_freeu and freeu:
            # https://github.com/ChenyangSi/FreeU
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
        device,
        dtype,
    ):
        model_lower = model.lower()
        model_name = self.pipe.config._name_or_path.lower() if self.pipe is not None else ""

        schedulers = {
            "DDIM": DDIMScheduler,
            "DEIS 2M": DEISMultistepScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "PNDM": PNDMScheduler,
        }

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

        # no fp16 variant
        if model_lower not in [
            "sg161222/realistic_vision_v5.1_novae",
            "prompthero/openjourney-v4",
            "linaqruf/anything-v3-1",
        ]:
            variant = "fp16"
        else:
            variant = None

        pipe_kwargs = {
            "scheduler": schedulers[scheduler](**scheduler_kwargs),
            "requires_safety_checker": False,
            "safety_checker": None,
            "torch_dtype": dtype,
            "variant": variant,
        }

        self._unload(kind, model, ip_adapter, scale)
        self._load_pipeline(kind, model, taesd, device, **pipe_kwargs)

        same_scheduler = isinstance(self.pipe.scheduler, schedulers[scheduler])
        same_karras = (
            not hasattr(self.pipe.scheduler.config, "use_karras_sigmas")
            or self.pipe.scheduler.config.use_karras_sigmas == karras
        )

        # same model, different scheduler
        if model_name == model_lower:
            if not same_scheduler:
                print(f"Switching to {scheduler}...")
            if not same_karras:
                print(f"{'Enabling' if karras else 'Disabling'} Karras sigmas...")
            if not same_scheduler or not same_karras:
                self.pipe.scheduler = schedulers[scheduler](**scheduler_kwargs)

        self._load_upscaler(device, scale)
        self._load_ip_adapter(ip_adapter)
        self._load_vae(taesd, model_lower, variant)
        self._load_freeu(freeu)
        self._load_deepcache(deepcache)
        return self.pipe, self.upscaler
