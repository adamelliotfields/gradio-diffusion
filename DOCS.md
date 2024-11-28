## Usage

TL;DR: Enter a prompt or roll the `🎲` and press `Generate`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel) for weighting. See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more.

Use `+` or `-` to increase the weight of a token. The weight grows exponentially when chained. For example, `blue+` means 1.1x more attention is given to `blue`, while `blue++` means 1.1^2 more, and so on. The same applies to `-`.

Groups of tokens can be weighted together by wrapping in parantheses and multiplying by a float between 0 and 2. For example, `(masterpiece, best quality)1.2` will increase the weight of both `masterpiece` and `best quality` by 1.2x.

This is the same syntax used in [InvokeAI](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/) and it differs from AUTOMATIC1111:

| Compel      | AUTOMATIC1111 |
| ----------- | ------------- |
| `blue++`    | `((blue))`    |
| `blue--`    | `[[blue]]`    |
| `(blue)1.2` | `(blue:1.2)`  |
| `(blue)0.8` | `(blue:0.8)`  |

### Models

Each model checkpoint has a different aesthetic:

* [Comfy-Org/stable-diffusion-v1-5](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive): base
* [cyberdelia/CyberRealistic_V5](https://huggingface.co/cyberdelia/CyberRealistic): realistic
* [Lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8): general purpose (default)
* [fluently/Fluently-v4](https://huggingface.co/fluently/Fluently-v4): general purpose stylized
* [Linaqruf/anything-v3-1](https://huggingface.co/Linaqruf/anything-v3-1): anime
* [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4): Midjourney art style
* [SG161222/Realistic_Vision_V5](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE): realistic
* [XpucT/Deliberate_v6](https://huggingface.co/XpucT/Deliberate): general purpose stylized

### Styles

[Styles](https://huggingface.co/spaces/adamelliotfields/diffusion/blob/main/data/styles.json) are prompt templates that wrap your positive and negative prompts. They were originally derived from the [twri/sdxl_prompt_styler](https://github.com/twri/sdxl_prompt_styler) Comfy node, but have since been entirely rewritten.

Start by framing a simple subject like `portrait of a cat` or `landscape of a mountain range` and experiment.

#### Anime

The `Anime: *` styles work the best with Dreamshaper. When using the anime-specific Anything model, you should use the `Anime: Anything` style with the following settings:

* Scheduler: `DEIS 2M` or `DPM++ 2M`
* Guidance: `10`
* Steps: `50`

You subject should be a few simple tokens like `girl, brunette, blue eyes, armor, nebula, celestial`. Experiment with `Clip Skip` and `Karras`. 

### Scale

Rescale up to 4x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) with weights from [ai-forever](ai-forever/Real-ESRGAN). Necessary for high-resolution images.

### Image-to-Image

The `Image-to-Image` settings allows you to provide input images for the initial latents, ControlNet, and IP-Adapter.

#### Strength

Initial image strength (known as _denoising strength_) is essentially how much the generation will differ from the input image. A value of `0` will be identical to the original, while `1` will be a completely new image. You may want to also increase the number of inference steps.

> 💡 Denoising strength only applies to the `Initial Image` input; it doesn't affect ControlNet or IP-Adapter.

#### ControlNet

In [ControlNet](https://github.com/lllyasviel/ControlNet), the input image is used to get a feature map from an _annotator_. These are computer vision models used for tasks like edge detection and pose estimation. ControlNet models are trained to understand these feature maps. Read the [docs](https://huggingface.co/docs/diffusers/using-diffusers/controlnet) to learn more.

Currently, the only annotator available is [Canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) (edge detection).

#### IP-Adapter

In an image-to-image pipeline, the input image is used as the initial latent. With [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), the input image is processed by a separate image encoder and the encoded features are used as conditioning along with the text prompt.

For capturing faces, enable `IP-Adapter Face` to use the full-face model. You should use an input image that is mostly a face and it should be high quality. You can generate fake portraits with Realistic Vision to experiment.

### Advanced

#### Textual Inversion

Enable `Use negative TI` to append [`fast_negative`](https://civitai.com/models/71961?modelVersionId=94057) to your negative prompt. Read [An Image is Worth One Word](https://huggingface.co/papers/2208.01618) to learn more.

#### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) caches lower UNet layers and reuses them every `Interval` steps. Trade quality for speed:
* `1`: no caching (default)
* `2`: more quality
* `3`: balanced
* `4`: more speed

#### FreeU

[FreeU](https://github.com/ChenyangSi/FreeU) re-weights the contributions sourced from the UNet’s skip connections and backbone feature maps. Can sometimes improve image quality.

#### Clip Skip

When enabled, the last CLIP layer is skipped. Can sometimes improve image quality.
