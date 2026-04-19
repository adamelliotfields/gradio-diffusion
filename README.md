# gradio-diffusion

Gradio app for Stable Diffusion 1.5 featuring:
* txt2img and img2img pipelines with IP-Adapter
* ControlNet with Canny edge detection
* FastNegative textual inversion
* Real-ESRGAN resizing up to 8x
* Compel prompt weighting support
* Multiple samplers with Karras scheduling
* DeepCache available for faster inference

## Installation

```bash
uv venv
uv pip install -r requirements.txt
uv run app.py
```

## Usage

Enter a prompt or roll the `🎲` and press `Generate`.

### AMD

You must `export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel). See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more.

### Models

Some require specific parameters to get the best results, so check the model's link for more information:

* [cyberdelia/CyberRealistic_V5](https://huggingface.co/cyberdelia/CyberRealistic)
* [fluently/Fluently-v4](https://huggingface.co/fluently/Fluently-v4)
* [Lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8)
* [s6yx/ReV_Animated](https://huggingface.co/s6yx/ReV_Animated)
* [SG161222/Realistic_Vision_V5](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)
* [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
* [XpucT/Deliberate_v6](https://huggingface.co/XpucT/Deliberate)
* [XpucT/Reliberate_v3](https://huggingface.co/XpucT/Reliberate) (default)

### Scale

Rescale up to 8x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) with weights from [ai-forever](ai-forever/Real-ESRGAN).

### Image-to-Image

The `Image-to-Image` settings allows you to provide input images for the initial latent, ControlNet, and IP-Adapter.

#### Strength

Initial image strength (known as _denoising strength_) is essentially how much the generation will differ from the input image. A value of `0` will be identical to the original, while `1` will be a completely new image. You may want to also increase the number of inference steps.

Note that denoising strength only applies to the `Initial Image` input; it doesn't affect ControlNet or IP-Adapter.

#### ControlNet

In [ControlNet](https://github.com/lllyasviel/ControlNet), the input image is used to get a feature map from an _annotator_. These are computer vision models used for tasks like edge detection and pose estimation. ControlNet models are trained to understand these feature maps. Read the [docs](https://huggingface.co/docs/diffusers/using-diffusers/controlnet) to learn more.

Currently, the only annotator available is [Canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) (edge detection).

#### IP-Adapter

In an image-to-image pipeline, the input image is used as the initial latent representation. With [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), the image is processed by a separate image encoder and the encoded features are used as conditioning along with the text prompt.

For capturing faces, enable `IP-Adapter Face` to use the full-face model. You should use an input image that is mostly a face and it should be high quality.

### Advanced

#### Textual Inversion

Add `<fast_negative>` anywhere in your negative prompt to apply the [FastNegative v2](https://civitai.com/models/71961?modelVersionId=94057) textual inversion embedding. Read [An Image is Worth One Word](https://huggingface.co/papers/2208.01618) to learn more.

> 💡 Wrap in parens to weight the embedding like `(<fast_negative>)0.8`.

#### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) caches lower UNet layers and reuses them every _n_ steps. Trade quality for speed:
- *1*: no caching (default)
- *2*: more quality
- *3*: balanced
- *4*: more speed
