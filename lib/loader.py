import os

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from diffusers.models import AutoencoderKL, AutoencoderTiny
from torch._dynamo import OptimizedModule

from .upscaler import RealESRGAN

ZERO_GPU = (
    os.environ.get("SPACES_ZERO_GPU", "").lower() == "true"
    or os.environ.get("SPACES_ZERO_GPU", "") == "1"
)


# inspired by ComfyUI
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py
class Loader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.pipe = None
            cls._instance.upscaler = None
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

    def _load_vae(self, model_name=None, taesd=False, variant=None):
        vae_type = type(self.pipe.vae)
        is_kl = issubclass(vae_type, (AutoencoderKL, OptimizedModule))
        is_tiny = issubclass(vae_type, AutoencoderTiny)

        # by default all models use KL
        if is_kl and taesd:
            # can't compile tiny VAE
            print("Switching to Tiny VAE...")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                pretrained_model_name_or_path="madebyollin/taesd",
                use_safetensors=True,
            ).to(device=self.pipe.device)
            return

        if is_tiny and not taesd:
            print("Switching to KL VAE...")
            model = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=model_name,
                use_safetensors=True,
                subfolder="vae",
                variant=variant,
            ).to(device=self.pipe.device)
            self.pipe.vae = torch.compile(
                mode="reduce-overhead",
                fullgraph=True,
                model=model,
            )

    def load(
        self,
        model,
        scheduler,
        karras,
        taesd,
        freeu,
        deepcache_interval,
        scale,
        dtype,
        device,
    ):
        model_lower = model.lower()

        schedulers = {
            "DEIS 2M": DEISMultistepScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "DPM2 a": KDPM2AncestralDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "Heun": HeunDiscreteScheduler,
            "LMS": LMSDiscreteScheduler,
            "PNDM": PNDMScheduler,
        }

        scheduler_kwargs = {
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "use_karras_sigmas": karras,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "steps_offset": 1,
        }

        if scheduler in ["Euler a", "PNDM"]:
            del scheduler_kwargs["use_karras_sigmas"]

        # no fp16 variant
        if not ZERO_GPU and model_lower not in [
            "sg161222/realistic_vision_v5.1_novae",
            "prompthero/openjourney-v4",
            "linaqruf/anything-v3-1",
        ]:
            variant = "fp16"
        else:
            variant = None

        pipe_kwargs = {
            "scheduler": schedulers[scheduler](**scheduler_kwargs),
            "pretrained_model_name_or_path": model_lower,
            "requires_safety_checker": False,
            "use_safetensors": True,
            "safety_checker": None,
            "variant": variant,
        }

        # already loaded
        if self.pipe is not None:
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
                self._load_vae(model_lower, taesd, variant)
                self._load_freeu(freeu)
                self._load_deepcache(deepcache_interval)
                self._load_upscaler(device, scale)
                torch.cuda.empty_cache()
                return self.pipe, self.upscaler
            else:
                print(f"Unloading {model_name.lower()}...")
                self.pipe = None

        print(f"Loading {model_lower} with {'Tiny' if taesd else 'KL'} VAE...")
        self.pipe = StableDiffusionPipeline.from_pretrained(**pipe_kwargs).to(
            device=device,
            dtype=dtype,
        )
        self._load_vae(model_lower, taesd, variant)
        self._load_freeu(freeu)
        self._load_deepcache(deepcache_interval)
        self._load_upscaler(device, scale)
        torch.cuda.empty_cache()
        return self.pipe, self.upscaler
