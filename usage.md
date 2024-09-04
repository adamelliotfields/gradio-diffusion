## Usage

Enter a prompt and click `Generate`. Roll the `🎲` for a random prompt.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel) for weighting. See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more and read [Civitai](https://civitai.com)'s guide on [prompting](https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/) for best practices.

#### Arrays

Arrays allow you to generate different images from a single prompt. For example, `[[cat,corgi]]` will expand into 2 separate prompts. Make sure `Images` is set accordingly (e.g., 2). Only works for the positive prompt. Inspired by [Fooocus](https://github.com/lllyasviel/Fooocus/pull/1503).

### Embeddings

Select multiple negative [textual inversion](https://huggingface.co/docs/diffusers/en/using-diffusers/textual_inversion_inference) embeddings. Fast Negative and Bad Dream can be used standalone or together; Unrealistic Dream should be combined with one of the others:

* [`<fast_negative>`](https://civitai.com/models/71961/fast-negative-embedding-fastnegativev2): all-purpose (default)
* [`<bad_dream>`](https://civitai.com/models/72437?modelVersionId=77169): DreamShaper-style
* [`<unrealistic_dream>`](https://civitai.com/models/72437?modelVersionId=77173): realistic add-on

### Styles

Styles are prompt templates from twri's [sdxl_prompt_styler](https://github.com/twri/sdxl_prompt_styler) Comfy node. Start with a subject like "cat", pick a style, and iterate from there.

### Scale

Rescale up to 4x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) from [ai-forever](ai-forever/Real-ESRGAN).

### Models

Each model checkpoint has a different aesthetic:

* [cyberdelia/CyberRealistic_v5](https://huggingface.co/cyberdelia/CyberRealistic): photorealistic
* [Lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8): general purpose (default)
* [fluently/Fluently-v4](https://huggingface.co/fluently/Fluently-v4): general purpose
* [Linaqruf/anything-v3-1](https://huggingface.co/Linaqruf/anything-v3-1): anime
* [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4): Midjourney-like
* [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5): base
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
