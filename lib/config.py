from types import SimpleNamespace
from warnings import filterwarnings

from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import logging as diffusers_logging
from transformers import logging as transformers_logging

filterwarnings("ignore", category=FutureWarning, module="diffusers")
filterwarnings("ignore", category=FutureWarning, module="transformers")

diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# Standard Stable Diffusion 1.5 file structure
_sd_files = [
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
    PIPELINES={
        "txt2img": StableDiffusionPipeline,
        "img2img": StableDiffusionImg2ImgPipeline,
        "controlnet_txt2img": StableDiffusionControlNetPipeline,
        "controlnet_img2img": StableDiffusionControlNetImg2ImgPipeline,
    },
    HF_REPOS={
        "ai-forever/Real-ESRGAN": ["RealESRGAN_x2.pth", "RealESRGAN_x4.pth", "RealESRGAN_x8.pth"],
        "cyberdelia/CyberRealistic": ["CyberRealistic_V5_FP16.safetensors"],
        "fluently/Fluently-v4": ["Fluently-v4.safetensors"],
        "h94/IP-Adapter": [
            "models/ip-adapter-full-face_sd15.safetensors",
            "models/ip-adapter-plus_sd15.safetensors",
            "models/image_encoder/model.safetensors",
        ],
        "lllyasviel/control_v11p_sd15_canny": ["diffusion_pytorch_model.fp16.safetensors"],
        "Lykon/dreamshaper-8": _sd_files,
        "s6yx/ReV_Animated": ["rev_1.2.2/rev_1.2.2-fp16.safetensors"],
        "SG161222/Realistic_Vision_V5.1_noVAE": ["Realistic_Vision_V5.1_fp16-no-ema.safetensors"],
        "stable-diffusion-v1-5/stable-diffusion-v1-5": _sd_files,
        "XpucT/Deliberate": ["Deliberate_v6.safetensors"],
        "XpucT/Reliberate": ["Reliberate_v3.safetensors"],
    },
    MODEL="XpucT/Reliberate",
    MODELS=[
        "cyberdelia/CyberRealistic",
        "fluently/Fluently-v4",
        "Lykon/dreamshaper-8",
        "s6yx/ReV_Animated",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "XpucT/Deliberate",
        "XpucT/Reliberate",
    ],
    SINGLE_FILE_MODELS=[
        "cyberdelia/CyberRealistic",
        "fluently/Fluently-v4",
        "s6yx/ReV_Animated",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "XpucT/Deliberate",
        "XpucT/Reliberate",
    ],
    SCHEDULER="UniPC",
    SCHEDULERS={
        "DEIS": DEISMultistepScheduler,
        "DPM++ 2M": DPMSolverMultistepScheduler,
        "Euler": EulerDiscreteScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "UniPC": UniPCMultistepScheduler,
    },
    ANNOTATOR="canny",
    ANNOTATORS={
        "canny": "lllyasviel/control_v11p_sd15_canny",
    },
    WIDTH=512,
    HEIGHT=512,
    NUM_IMAGES=1,
    GUIDANCE_SCALE=6,
    INFERENCE_STEPS=40,
    DENOISING_STRENGTH=0.8,
    DEEPCACHE_INTERVAL=1,
    SCALE=1,
    SCALES=[1, 2, 4, 8],
)
