---
# https://huggingface.co/docs/hub/en/spaces-config-reference
title: Stable Diffusion Zero
short_description: SD 1.5 on ZeroGPU
emoji: 🎨
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.39.0
python_version: 3.11.9
app_file: demo.py
fullWidth: false
pinned: true
header: mini
license: apache-2.0
preload_from_hub:
  - fluently/Fluently-v4 text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors
  - Lykon/dreamshaper-8 text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors
  - prompthero/openjourney-v4 text_encoder/model.safetensors,unet/diffusion_pytorch_model.safetensors
  - runwayml/stable-diffusion-v1-5 text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors
  - SG161222/Realistic_Vision_V5.1_noVAE text_encoder/model.safetensors,unet/diffusion_pytorch_model.safetensors
---

See [`about.md`](https://huggingface.co/spaces/adamelliotfields/sd/blob/main/about.md).
