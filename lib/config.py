from types import SimpleNamespace

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

Config = SimpleNamespace(
    MONO_FONTS=["monospace"],
    SANS_FONTS=[
        "sans-serif",
        "Apple Color Emoji",
        "Segoe UI Emoji",
        "Segoe UI Symbol",
        "Noto Color Emoji",
    ],
    PIPELINES={
        "txt2img": StableDiffusionPipeline,
        "img2img": StableDiffusionImg2ImgPipeline,
    },
    MODEL="Lykon/dreamshaper-8",
    MODELS=[
        "cyberdelia/CyberRealistic",
        "fluently/Fluently-v4",
        "Linaqruf/anything-v3-1",
        "Lykon/dreamshaper-8",
        "prompthero/openjourney-v4",
        "runwayml/stable-diffusion-v1-5",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "XpucT/Deliberate",
    ],
    MODEL_CHECKPOINTS={
        # keep keys lowercase
        "cyberdelia/cyberrealistic": "CyberRealistic_V5_FP16.safetensors",
        "fluently/fluently-v4": "Fluently-v4.safetensors",
        "linaqruf/anything-v3-1": "anything-v3-2.safetensors",
        "prompthero/openjourney-v4": "openjourney-v4.ckpt",
        "sg161222/realistic_vision_v5.1_novae": "Realistic_Vision_V5.1_fp16-no-ema.safetensors",
        "xpuct/deliberate": "Deliberate_v6.safetensors",
    },
    SCHEDULER="DEIS 2M",
    SCHEDULERS={
        "DDIM": DDIMScheduler,
        "DEIS 2M": DEISMultistepScheduler,
        "DPM++ 2M": DPMSolverMultistepScheduler,
        "Euler": EulerDiscreteScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "PNDM": PNDMScheduler,
    },
    EMBEDDING="fast_negative",
    EMBEDDINGS=[
        "bad_dream",
        "fast_negative",
        "unrealistic_dream",
    ],
    STYLE="sai-enhance",
    WIDTH=448,
    HEIGHT=576,
    NUM_IMAGES=1,
    SEED=-1,
    GUIDANCE_SCALE=6,
    INFERENCE_STEPS=35,
    DENOISING_STRENGTH=0.7,
    DEEPCACHE_INTERVAL=1,
    SCALE=1,
    SCALES=[1, 2, 4],
)
