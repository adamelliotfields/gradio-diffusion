import os
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
    HF_TOKEN=os.environ.get("HF_TOKEN", None),
    CIVIT_TOKEN=os.environ.get("CIVIT_TOKEN", None),
    HF_MODELS={
        "Lykon/dreamshaper-8": [
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
        ],
    },
    CIVIT_LORAS={
        # https://civitai.com/models/411088?modelVersionId=486099
        "perfection_style": {
            "model_id": "411088",
            "model_version_id": "486099",
            "name": "Perfection Style",
            "trigger": "perfection style",
        },
        # https://civitai.com/models/421162?modelVersionId=486110
        "detailed_style": {
            "model_id": "421162",
            "model_version_id": "486110",
            "name": "Detailed Style",
            "trigger": "detailed style",
        },
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
        "txt2img": StableDiffusionPipeline,
        "img2img": StableDiffusionImg2ImgPipeline,
    },
    MODEL="Lykon/dreamshaper-8",
    MODELS=[
        "Comfy-Org/stable-diffusion-v1-5-archive",
        "cyberdelia/CyberRealistic",
        "fluently/Fluently-v4",
        "Linaqruf/anything-v3-1",
        "Lykon/dreamshaper-8",
        "prompthero/openjourney-v4",
        "SG161222/Realistic_Vision_V5.1_noVAE",
        "XpucT/Deliberate",
    ],
    MODEL_CHECKPOINTS={
        # keep keys lowercase
        "comfy-org/stable-diffusion-v1-5-archive": "v1-5-pruned-emaonly-fp16.safetensors",
        "cyberdelia/cyberrealistic": "CyberRealistic_V5_FP16.safetensors",
        "fluently/fluently-v4": "Fluently-v4.safetensors",
        "linaqruf/anything-v3-1": "anything-v3-2.safetensors",
        "prompthero/openjourney-v4": "openjourney-v4.ckpt",
        "sg161222/realistic_vision_v5.1_novae": "Realistic_Vision_V5.1_fp16-no-ema.safetensors",
        "xpuct/deliberate": "Deliberate_v6.safetensors",
    },
    SCHEDULER="Euler",
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
        "cyberrealistic_negative",
        "fast_negative",
        "unrealistic_dream",
    ],
    STYLE="enhance",
    WIDTH=512,
    HEIGHT=512,
    NUM_IMAGES=1,
    SEED=-1,
    GUIDANCE_SCALE=5,
    INFERENCE_STEPS=35,
    DENOISING_STRENGTH=0.7,
    DEEPCACHE_INTERVAL=1,
    SCALE=1,
    SCALES=[1, 2, 4],
)
