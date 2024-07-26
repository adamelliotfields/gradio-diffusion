---
title: Stable Diffusion Zero
short_description: SD 1.5 on ZeroGPU
emoji: 🎨
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.39.0
python_version: 3.11.9
app_file: demo.py
pinned: true
fullWidth: false
header: mini
license: apache-2.0
preload_from_hub:
  - fluently/Fluently-v4 text_encoder/model.safetensors,text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,unet/diffusion_pytorch_model.safetensors
  - Lykon/dreamshaper-8 text_encoder/model.safetensors,text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,unet/diffusion_pytorch_model.safetensors
  - prompthero/openjourney-v4 text_encoder/model.safetensors,unet/diffusion_pytorch_model.safetensors
  - runwayml/stable-diffusion-v1-5 text_encoder/model.safetensors,text_encoder/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,unet/diffusion_pytorch_model.safetensors
  - SG161222/Realistic_Vision_V5.1_noVAE text_encoder/model.safetensors,unet/diffusion_pytorch_model.safetensors
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
