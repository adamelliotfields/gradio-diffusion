## Diffusion ZERO

TL;DR: Enter a prompt or roll the `🎲` and press `Generate`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel) for weighting. See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more.

Use `+` or `-` to increase the weight of a token. The weight grows exponentially when chained. For example, `blue+` means 1.1x more attention is given to `blue`, while `blue++` means 1.1^2 more, and so on. The same applies to `-`.

For groups of tokens, wrap them in parentheses and multiply by a float between 0 and 2. For example, `a (birthday cake)1.3 on a table` will increase the weight of both `birthday` and `cake` by 1.3x. This also means the entire scene will be more birthday-like, not just the cake. To counteract this, you can use `-` inside the parentheses on specific tokens, e.g., `a (birthday-- cake)1.3`, to reduce the birthday aspect.

Note that this is also the same syntax used in [InvokeAI](https://invoke-ai.github.io/InvokeAI/features/PROMPTS/) and it differs from AUTOMATIC1111:

| Compel      | AUTOMATIC1111 |
| ----------- | ------------- |
| `blue++`    | `((blue))`    |
| `blue--`    | `[[blue]]`    |
| `(blue)1.2` | `(blue:1.2)`  |
| `(blue)0.8` | `(blue:0.8)`  |

#### Arrays

Arrays allow you to generate multiple different images from a single prompt. For example, `a [[cute,adorable]] [[cat,corgi]]` will expand into `a cute cat` and `a cute corgi`.

Before generating, make sure `Images` is set to the number of images you want and keep in mind that there is a max of 4. Note that arrays in the negative prompt are ignored. This implementation was inspired by [Fooocus](https://github.com/lllyasviel/Fooocus/pull/1503).

### Embeddings

Select one or more [textual inversion](https://huggingface.co/docs/diffusers/en/using-diffusers/textual_inversion_inference) embeddings:

* [`fast_negative`](https://civitai.com/models/71961?modelVersionId=94057): all-purpose (default)
* [`unrealistic_dream`](https://civitai.com/models/72437?modelVersionId=77173): realistic add-on (for RealisticVision)
* [`cyberrealistic_negative`](https://civitai.com/models/77976?modelVersionId=82745): realistic add-on (for CyberRealistic)

### Styles

[Styles](https://huggingface.co/spaces/adamelliotfields/diffusion/blob/main/data/styles.json) are prompt templates that wrap your positive and negative prompts. They were originally derived from the [twri/sdxl_prompt_styler](https://github.com/twri/sdxl_prompt_styler) Comfy node, but have since been entirely rewritten.

Start by framing a simple subject like `portrait of a young adult woman` or `landscape of a mountain range`. Experiment with different styles and don't forget about the negative prompt.

> NB: Most styles work best with the Dreamshaper model; however, the "Enhance" style is meant to be universal. The "Photography" styles work especially well with the realistic models.

### Scale

Rescale up to 4x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) with weights from [ai-forever](ai-forever/Real-ESRGAN). Necessary for high-resolution images.

> NB: I find this Real-ESRGAN model to work well, so I do not use a _hi-res fix_.

### Models

Each model checkpoint has a different aesthetic:

* [Comfy-Org/stable-diffusion-v1-5](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive): base
* [cyberdelia/CyberRealistic_v5](https://huggingface.co/cyberdelia/CyberRealistic): photorealistic
* [Lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8): general purpose (default)
* [fluently/Fluently-v4](https://huggingface.co/fluently/Fluently-v4): general purpose
* [Linaqruf/anything-v3-1](https://huggingface.co/Linaqruf/anything-v3-1): anime
* [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4): Midjourney-like
* [SG161222/Realistic_Vision_v5.1](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE): photorealistic
* [XpucT/Deliberate_v6](https://huggingface.co/XpucT/Deliberate): general purpose

### Image-to-Image

The `🖼️ Image` tab enables the image-to-image and IP-Adapter pipelines. Either use the image input or select a generation from the gallery. To disable, simply clear the image input (the `x` overlay button).

Denoising strength is essentially how much the generation will differ from the input image. A value of `0` will be identical to the original, while `1` will be a completely new image. You may want to also increase the number of inference steps. Only applies to the image-to-image input.

### IP-Adapter

In an image-to-image pipeline, the input image is used as the initial latent. With [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), the input image is processed by a separate image encoder and the encoded features are used as conditioning along with the text prompt.

For capturing faces, enable `IP-Adapter Face` to use the full-face model. You should use an input image that is mostly a face and it should be high quality. You can generate fake portraits with Realistic Vision to experiment. Note that you'll never get true identity preservation without an advanced pipeline like [InstantID](https://github.com/instantX-research/InstantID), which combines many techniques.

### Advanced

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

#### Tiny VAE

Enable [madebyollin/taesd](https://github.com/madebyollin/taesd) for near-instant latent decoding with a minor loss in detail. Useful for development.
