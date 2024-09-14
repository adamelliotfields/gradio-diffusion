---
# https://huggingface.co/docs/hub/en/spaces-config-reference
title: Diffusion Zero
short_description: Image generation studio on ZeroGPU
emoji: 🧨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.41.0
python_version: 3.11.9
app_file: app.py
fullWidth: false
pinned: true
header: mini
license: apache-2.0
models:
- ai-forever/Real-ESRGAN
- Comfy-Org/stable-diffusion-v1-5-archive
- cyberdelia/CyberRealistic
- fluently/Fluently-v4
- h94/IP-Adapter
- Linaqruf/anything-v3-1
- Lykon/dreamshaper-8
- prompthero/openjourney-v4
- SG161222/Realistic_Vision_V5.1_noVAE
- XpucT/Deliberate
preload_from_hub:  # up to 10
- >-
  ai-forever/Real-ESRGAN
  RealESRGAN_x2.pth,RealESRGAN_x4.pth
- >-
  Comfy-Org/stable-diffusion-v1-5-archive
  v1-5-pruned-emaonly-fp16.safetensors
- >-
  cyberdelia/CyberRealistic
  CyberRealistic_V5_FP16.safetensors
- >-
  fluently/Fluently-v4
  Fluently-v4.safetensors
- >-
  h94/IP-Adapter
  models/ip-adapter-full-face_sd15.safetensors,models/ip-adapter-plus_sd15.safetensors,models/image_encoder/model.safetensors
- >-
  Linaqruf/anything-v3-1
  anything-v3-2.safetensors
- >-
  Lykon/dreamshaper-8
  feature_extractor/preprocessor_config.json,safety_checker/config.json,scheduler/scheduler_config.json,text_encoder/config.json,text_encoder/model.fp16.safetensors,tokenizer/merges.txt,tokenizer/special_tokens_map.json,tokenizer/tokenizer_config.json,tokenizer/vocab.json,unet/config.json,unet/diffusion_pytorch_model.fp16.safetensors,vae/config.json,vae/diffusion_pytorch_model.fp16.safetensors,model_index.json
- >-
  prompthero/openjourney-v4
  openjourney-v4.ckpt
- >-
  SG161222/Realistic_Vision_V5.1_noVAE
  Realistic_Vision_V5.1_fp16-no-ema.safetensors
- >-
  XpucT/Deliberate
  Deliberate_v6.safetensors
---

# diffusion

Gradio app for Stable Diffusion 1.5 featuring:
* txt2img and img2img pipelines with IP-Adapter
* Curated models, LoRAs, and TI embeddings
* Compel prompt weighting
* dozens of styles and starter prompts
* Multiple samplers with Karras scheduling
* DeepCache, FreeU, and Clip Skip available
* Real-ESRGAN upscaling
* Optional tiny autoencoder

There's also a [CLI](https://huggingface.co/spaces/adamelliotfields/diffusion/blob/main/cli.py).

## Motivation

I want to:

* host a free and easy-to-use Stable Diffusion UI on ZeroGPU
* provide the necessary tools for common workflows
* curate useful models, adapters, and embeddings
* prefer Diffusers over custom PyTorch
* be fast on 8GB with no offloading
* only support CUDA on Linux/WSL

## Usage

See [`DOCS.md`](https://huggingface.co/spaces/adamelliotfields/diffusion/blob/main/DOCS.md).

## Installation

```bash
# clone
git clone https://huggingface.co/spaces/adamelliotfields/diffusion.git
cd diffusion
git remote set-url origin https://adamelliotfields:$HF_TOKEN@huggingface.co/spaces/adamelliotfields/diffusion

# install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt torch==2.4.0 torchvision==0.19.0

# gradio
python app.py --port 7860

# cli
python cli.py 'an astronaut riding a horse on mars'
```

## Development

See [pull requests and discussions](https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions).

```sh
git fetch origin refs/pr/42:pr/42
git checkout pr/42
# ...
git add .
git commit -m "Commit message"
git push origin pr/42:refs/pr/42
```
