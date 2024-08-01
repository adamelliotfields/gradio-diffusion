---
title: Stable Diffusion Zero
short_description: SD 1.5 on ZeroGPU
emoji: 😻
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.39.0
python_version: 3.11.9
app_file: app.py
fullWidth: false
pinned: true
header: mini
license: apache-2.0
models:
- fluently/Fluently-v4
- Linaqruf/anything-v3-1
- Lykon/dreamshaper-8
- prompthero/openjourney-v4
- runwayml/stable-diffusion-v1-5
- SG161222/Realistic_Vision_V5.1_noVAE
preload_from_hub:
- >-
  fluently/Fluently-v4
  text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,vae/diffusion_pytorch_model.fp16.safetensors
- >-
  Linaqruf/anything-v3-1
  text_encoder/model.safetensors,unet/diffusion_pytorch_model.safetensors,vae/diffusion_pytorch_model.safetensors
- >-
  Lykon/dreamshaper-8
  text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,vae/diffusion_pytorch_model.fp16.safetensors
- >-
  prompthero/openjourney-v4
  text_encoder/model.safetensors,unet/diffusion_pytorch_model.safetensors,vae/diffusion_pytorch_model.safetensors
- >-
  runwayml/stable-diffusion-v1-5
  text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,vae/diffusion_pytorch_model.fp16.safetensors
- >-
  SG161222/Realistic_Vision_V5.1_noVAE
  text_encoder/model.safetensors,unet/diffusion_pytorch_model.safetensors,vae/diffusion_pytorch_model.safetensors
---

# diffusion

Gradio-based Stable Diffusion 1.5 app on ZeroGPU.

## Usage

See [`info.md`](https://huggingface.co/spaces/adamelliotfields/diffusion/blob/main/info.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt torch==2.4.0 torchvision==0.19.0

# http://localhost:7860
python app.py
```

## TODO

- [ ] Styles
- [ ] Hires fix
- [ ] Support LoRA
- [ ] Metadata embed and display
- [ ] Latent preview
