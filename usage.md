## Usage

Enter a prompt and click `Generate`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel) for weighting. You can use a float or +/-. For example:
* `man, portrait, blue+ eyes, close-up`
* `man, portrait, (blue)1.1 eyes, close-up`
* `man, portrait, (blue eyes)-, close-up`
* `man, portrait, (blue eyes)0.9, close-up`

Note that `++` is `1.1^2` (and so on). See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more and read [Civitai](https://civitai.com)'s guide on [prompting](https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/) for best practices.

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

Rescale up to 4x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

### Models

Each model checkpoint has a different aesthetic:

* [lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8): general purpose (default)
* [fluently/fluently-v4](https://huggingface.co/fluently/Fluently-v4): general purpose merge
* [linaqruf/anything-v3-1](https://huggingface.co/linaqruf/anything-v3-1): anime
* [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4): Midjourney-like
* [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5): base
* [sg161222/realistic_vision_v5.1](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE): photorealistic

### Schedulers

Optionally, the [Karras](https://arxiv.org/abs/2206.00364) noise schedule can be used:

* [DEIS 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/deis) (default)
* [DPM++ 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver)
* [DPM2 a](https://huggingface.co/docs/diffusers/api/schedulers/dpm_discrete_ancestral)
* [Euler a](https://huggingface.co/docs/diffusers/en/api/schedulers/euler_ancestral)
* [Heun](https://huggingface.co/docs/diffusers/api/schedulers/heun)
* [LMS](https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete)
* [PNDM](https://huggingface.co/docs/diffusers/api/schedulers/pndm)

### Image-to-Image

The `🖼️ Image` tab enables the image-to-image pipeline. Either use the image input or select a generation from the gallery and then adjust the denoising strength. To disable, simply clear the image input (the `x` overlay button).

Denoising strength is essentially how much the generation will differ from the input image. A value of `0` will be identical to the original, while `1` will be a completely new image. You may want to also increase the number of inference steps.

### Advanced

#### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) (Ma et al. 2023) caches lower U-Net layers and reuses them every `Interval` steps:
* `1`: no caching
* `2`: more quality (default)
* `3`: balanced
* `4`: more speed

#### FreeU

[FreeU](https://github.com/ChenyangSi/FreeU) (Si et al. 2023) re-weights the contributions sourced from the U-Net’s skip connections and backbone feature maps to potentially improve image quality.

#### Clip Skip

When enabled, the last CLIP layer is skipped. This can sometimes improve image quality with anime models.

#### Tiny VAE

Enable [madebyollin/taesd](https://github.com/madebyollin/taesd) for almost instant latent decoding with a minor loss in detail. Useful for development.

#### Prompt Truncation

When enabled, prompts will be truncated to CLIP's limit of 77 tokens. By default this is _disabled_, so Compel will chunk prompts into segments rather than cutting them off.
