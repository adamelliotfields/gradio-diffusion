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
- fluently/Fluently-v4
- Linaqruf/anything-v3-1
- Lykon/dreamshaper-8
- prompthero/openjourney-v4
- runwayml/stable-diffusion-v1-5
- SG161222/Realistic_Vision_V5.1_noVAE
preload_from_hub:
- >-
  ai-forever/Real-ESRGAN
  RealESRGAN_x2.pth,RealESRGAN_x4.pth
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

Gradio app for Stable Diffusion 1.5 including:
* curated models and TI embeddings
* multiple samplers with Karras schedule
* Compel prompting
* 100+ styles from sdxl_prompt_styler
* FreeU and Clip Skip for quality
* DeepCache and ToMe for speed
* Real-ESRGAN upscaling
* optional TAESD

## Usage

See [`usage.md`](https://huggingface.co/spaces/adamelliotfields/diffusion/blob/main/usage.md).

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

## TODO

- [ ] Metadata embed and display
- [ ] Image-to-image
