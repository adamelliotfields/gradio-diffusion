import os
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
from warnings import filterwarnings

from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import logging as diffusers_logging
from transformers import logging as transformers_logging

from .pipelines import (
    CustomStableDiffusionControlNetImg2ImgPipeline,
    CustomStableDiffusionControlNetPipeline,
    CustomStableDiffusionImg2ImgPipeline,
    CustomStableDiffusionPipeline,
)

# Improved GPU handling and progress bars; set before importing spaces
os.environ["ZEROGPU_V2"] = "1"

# Errors if enabled and not installed
if find_spec("hf_transfer"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

filterwarnings("ignore", category=FutureWarning, module="diffusers")
filterwarnings("ignore", category=FutureWarning, module="transformers")

diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# Standard Stable Diffusion 1.5 file structure
sd_files = [
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "model_index.json",
]

# Using namespace instead of dataclass for simplicity
Config = SimpleNamespace(
    HF_TOKEN=os.environ.get("HF_TOKEN", None),
    ZERO_GPU=import_module("spaces").config.Config.zero_gpu,
    # TODO: fix model config redundancy
    HF_MODELS={
        # downloaded on startup
        "ai-forever/Real-ESRGAN": ["RealESRGAN_x2.pth", "RealESRGAN_x4.pth"],
        "cyberdelia/CyberRealistic": ["CyberRealistic_V5_FP16.safetensors"],
        "dreamlike-art/dreamlike-photoreal-2.0": ["dreamlike-photoreal-2.0.safetensors"],
        "fluently/Fluently-v4": ["Fluently-v4.safetensors"],
        "lllyasviel/control_v11p_sd15_canny": ["diffusion_pytorch_model.fp16.safetensors"],
        "Lykon/dreamshaper-8": [*sd_files],
        "s6yx/ReV_Animated": ["rev_1.2.2/rev_1.2.2-fp16.safetensors"],
        "SG161222/Realistic_Vision_V5.1_noVAE": ["Realistic_Vision_V5.1_fp16-no-ema.safetensors"],
        "stable-diffusion-v1-5/stable-diffusion-v1-5": [*sd_files],
        "XpucT/Deliberate": ["Deliberate_v6.safetensors"],
    },
    MONO_FONTS=["monospace"],
    SANS_FONTS=[
        "sans-serif",
        "Apple Color Emoji",
        "Segoe UI Emoji",
        "Segoe UI Symbol",
        "Noto Color Emoji",
    ],
    PIPELINES={
        "txt2img": CustomStableDiffusionPipeline,
        "img2img": CustomStableDiffusionImg2ImgPipeline,
        "controlnet_txt2img": CustomStableDiffusionControlNetPipeline,
        "controlnet_img2img": CustomStableDiffusionControlNetImg2ImgPipeline,
    },
    MODEL="Lykon/dreamshaper-8",
    MODELS=[
        "cyberdelia/CyberRealistic",
        "dreamlike-art/dreamlike-photoreal-2.0",
        "fluently/Fluently-v4",
        "Lykon/dreamshaper-8",
        "s6yx/ReV_Animated",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "XpucT/Deliberate",
    ],
    # Single-file model weights
    MODEL_CHECKPOINTS={
        # keep keys lowercase for case-insensitive matching in the loader
        "cyberdelia/cyberrealistic": "CyberRealistic_V5_FP16.safetensors",
        "dreamlike-art/dreamlike-photoreal-2.0": "dreamlike-photoreal-2.0.safetensors",
        "fluently/fluently-v4": "Fluently-v4.safetensors",
        "s6yx/rev_animated": "rev_1.2.2/rev_1.2.2-fp16.safetensors",
        "sg161222/realistic_vision_v5.1_novae": "Realistic_Vision_V5.1_fp16-no-ema.safetensors",
        "xpuct/deliberate": "Deliberate_v6.safetensors",
    },
    SCHEDULER="UniPC 2M",
    SCHEDULERS={
        "DDIM": DDIMScheduler,
        "DEIS 2M": DEISMultistepScheduler,
        "DPM++ 2M": DPMSolverMultistepScheduler,
        "Euler": EulerDiscreteScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "PNDM": PNDMScheduler,
        "UniPC 2M": UniPCMultistepScheduler,
    },
    ANNOTATOR="canny",
    ANNOTATORS={
        "canny": "lllyasviel/control_v11p_sd15_canny",
    },
    NEGATIVE_EMBEDDING="fast_negative",
    STYLE="enhance",
    WIDTH=512,
    HEIGHT=512,
    NUM_IMAGES=1,
    SEED=-1,
    GUIDANCE_SCALE=7.5,
    INFERENCE_STEPS=40,
    DENOISING_STRENGTH=0.7,
    DEEPCACHE_INTERVAL=1,
    SCALE=1,
    SCALES=[1, 2, 4],
)
