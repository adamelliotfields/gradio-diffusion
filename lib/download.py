import os

from huggingface_hub._snapshot_download import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN", None)

SPACES_ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() == "true"

REPO = "Lykon/dreamshaper-8"

FILES = [
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


def download_repo_files():
    global REPO, FILES
    return snapshot_download(
        repo_id=REPO,
        repo_type="model",
        revision="main",
        token=HF_TOKEN,
        allow_patterns=FILES,
        ignore_patterns=None,
    )
