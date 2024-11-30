---
# https://huggingface.co/docs/hub/en/spaces-config-reference
title: Diffusion
short_description: Image generation studio for SD 1.5
emoji: 🧨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
python_version: 3.11.9
app_file: app.py
fullWidth: false
pinned: true
header: mini
license: apache-2.0
models:
- ai-forever/Real-ESRGAN
- cyberdelia/CyberRealistic
- fluently/Fluently-v4
- h94/IP-Adapter
- Lykon/dreamshaper-8
- s6yx/ReV_Animated
- SG161222/Realistic_Vision_V5.1_noVAE
- stable-diffusion-v1-5/stable-diffusion-v1-5
- XpucT/Deliberate
- XpucT/Reliberate
preload_from_hub:  # up to 10
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
  lllyasviel/control_v11p_sd15_canny
  diffusion_pytorch_model.fp16.safetensors
- >-
  Lykon/dreamshaper-8
  feature_extractor/preprocessor_config.json,safety_checker/config.json,scheduler/scheduler_config.json,text_encoder/config.json,text_encoder/model.fp16.safetensors,tokenizer/merges.txt,tokenizer/special_tokens_map.json,tokenizer/tokenizer_config.json,tokenizer/vocab.json,unet/config.json,unet/diffusion_pytorch_model.fp16.safetensors,vae/config.json,vae/diffusion_pytorch_model.fp16.safetensors,model_index.json
- >-
  s6yx/ReV_Animated
  rev_1.2.2/rev_1.2.2-fp16.safetensors
- >-
  SG161222/Realistic_Vision_V5.1_noVAE
  Realistic_Vision_V5.1_fp16-no-ema.safetensors
- >-
  stable-diffusion-v1-5/stable-diffusion-v1-5
  feature_extractor/preprocessor_config.json,safety_checker/config.json,scheduler/scheduler_config.json,text_encoder/config.json,text_encoder/model.fp16.safetensors,tokenizer/merges.txt,tokenizer/special_tokens_map.json,tokenizer/tokenizer_config.json,tokenizer/vocab.json,unet/config.json,unet/diffusion_pytorch_model.fp16.safetensors,vae/config.json,vae/diffusion_pytorch_model.fp16.safetensors,model_index.json
- >-
  XpucT/Deliberate
  Deliberate_v6.safetensors
- >-
  XpucT/Deliberate
  Reliberate_v3.safetensors
---

# diffusion

Gradio app for Stable Diffusion 1.5 featuring:
* txt2img and img2img pipelines with IP-Adapter
* ControlNet with Canny edge detection (more preprocessors coming soon)
* Compel prompt weighting and blending
* Multiple samplers with Karras scheduling
* DeepCache available
* Real-ESRGAN upscaling

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
pip install -r requirements.txt

# gradio
python app.py --port 7860
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
