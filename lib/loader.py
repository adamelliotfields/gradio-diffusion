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


# inspired by ComfyUI
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py
class Loader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.pipe = None
            cls._instance.upscaler = None
            cls._instance.ip_adapter = None
        return cls._instance

    def _load_upscaler(self, device=None, scale=4):
        same_scale = self.upscaler is not None and self.upscaler.scale == scale
        if scale == 1:
            self.upscaler = None
        if scale > 1 and not same_scale:
            self.upscaler = RealESRGAN(device=device, scale=scale)
            self.upscaler.load_weights()

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
            self.pipe.disable_freeu()
        elif not has_freeu and freeu:
            # https://github.com/ChenyangSi/FreeU
            self.pipe.enable_freeu(b1=1.5, b2=1.6, s1=0.9, s2=0.2)

    def _load_ip_adapter(self, ip_adapter=None):
        if self.ip_adapter is None and self.ip_adapter != ip_adapter:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name=f"ip-adapter-{ip_adapter}_sd15.safetensors",
            )
            self.pipe.set_ip_adapter_scale(0.6 if ip_adapter == "full-face" else 0.5)
            self.ip_adapter = ip_adapter

        if self.ip_adapter is not None and ip_adapter is None:
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
            self.pipe.ip_adapter = None

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

    def _load_pipeline(self, kind, model, device, **kwargs):
        pipelines = {
            "txt2img": StableDiffusionPipeline,
            "img2img": StableDiffusionImg2ImgPipeline,
        }
        if self.pipe is None:
            self.pipe = pipelines[kind].from_pretrained(model, **kwargs).to(device)
        if not isinstance(self.pipe, pipelines[kind]):
            self.pipe = pipelines[kind].from_pipe(self.pipe).to(device)
            self.ip_adapter = None

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

        if self.pipe is None:
            print(f"Loading {model_lower} with {'Tiny' if taesd else 'KL'} VAE...")

        self._load_pipeline(kind, model_lower, device, **pipe_kwargs)
        model_name = self.pipe.config._name_or_path
        same_model = model_name.lower() == model_lower
        same_scheduler = isinstance(self.pipe.scheduler, schedulers[scheduler])
        same_karras = (
            not hasattr(self.pipe.scheduler.config, "use_karras_sigmas")
            or self.pipe.scheduler.config.use_karras_sigmas == karras
        )

        if same_model:
            if not same_scheduler:
                print(f"Switching to {scheduler}...")
            if not same_karras:
                print(f"{'Enabling' if karras else 'Disabling'} Karras sigmas...")
            if not same_scheduler or not same_karras:
                self.pipe.scheduler = schedulers[scheduler](**scheduler_kwargs)
        else:
            self.pipe = None
            self._load_pipeline(kind, model_lower, device, **pipe_kwargs)

        self._load_ip_adapter(ip_adapter)
        self._load_vae(taesd, model_lower, variant)
        self._load_freeu(freeu)
        self._load_deepcache(deepcache)
        self._load_upscaler(device, scale)
        torch.cuda.empty_cache()
        return self.pipe, self.upscaler
