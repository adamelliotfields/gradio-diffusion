## Usage

Enter a prompt and click `Generate`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel) for weighting. You can use a float or +/-. For example:
* `man, portrait, blue+ eyes, close-up`
* `man, portrait, (blue)1.1 eyes, close-up`
* `man, portrait, (blue eyes)-, close-up`
* `man, portrait, (blue eyes)0.9, close-up`

Note that `++` is `1.1^2` (and so on). See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more and read [Civitai](https://civitai.com)'s guide on [prompting](https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/) for best practices.

#### Negative Prompt

Start with a [textual inversion](https://huggingface.co/docs/diffusers/en/using-diffusers/textual_inversion_inference) embedding:

* [`<bad_prompt>`](https://civitai.com/models/55700/badprompt-negative-embedding)
* [`<negative_hand>`](https://civitai.com/models/56519/negativehand-negative-embedding)
* [`<fast_negative>`](https://civitai.com/models/71961/fast-negative-embedding-fastnegativev2)
* [`<bad_dream>`](https://civitai.com/models/72437?modelVersionId=77169)
* [`<unrealistic_dream>`](https://civitai.com/models/72437?modelVersionId=77173)

And add to it. You can use weighting in the negative prompt as well.

#### Arrays

Arrays allow you to generate different images from a single prompt. For example, `[[cat,corgi]]` will expand into 2 separate prompts. Make sure `Images` is set accordingly (e.g., 2). Only works for the positive prompt. Inspired by [Fooocus](https://github.com/lllyasviel/Fooocus/pull/1503).

### Styles

Styles are prompt templates from twri's [sdxl_prompt_styler](https://github.com/twri/sdxl_prompt_styler) Comfy node. Start with a subject like "cat", pick a style, and iterate from there.

### Models

Each model checkpoint has a different aesthetic:

* [lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8): general purpose (default)
* [fluently/fluently-v4](https://huggingface.co/fluently/Fluently-v4): general purpose merge
* [linaqruf/anything-v3-1](https://huggingface.co/linaqruf/anything-v3-1): anime
* [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4): Midjourney-like
* [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5): base
* [sg161222/realistic_vision_v5.1](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE): photorealistic

#### Schedulers

Optionally, the [Karras](https://arxiv.org/abs/2206.00364) noise schedule can be used:

* [DEIS 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/deis) (default)
* [DPM++ 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver)
* [DPM2 a](https://huggingface.co/docs/diffusers/api/schedulers/dpm_discrete_ancestral)
* [Euler a](https://huggingface.co/docs/diffusers/en/api/schedulers/euler_ancestral)
* [Heun](https://huggingface.co/docs/diffusers/api/schedulers/heun)
* [LMS](https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete)
* [PNDM](https://huggingface.co/docs/diffusers/api/schedulers/pndm)

#### Upscaler

[AuraSR](https://huggingface.co/fal/AuraSR-v2) from [fal.ai](https://fal.ai) can be enabled to upscale your image 4x. It's disabled if `Images` is greater than **1**, so use it once you've finalized your parameters and found a seed.

### Advanced

#### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) (Ma et al. 2023) caches lower UNet layers and reuses them every `Interval` steps:
* `1`: no caching
* `2`: more quality (default)
* `3`: balanced
* `4`: more speed

#### ToMe

[Token merging](https://arxiv.org/abs/2303.17604) (Bolya & Hoffman 2023) reduces the number of tokens processed by the model. Set `Ratio` to the desired reduction factor. ToMe's impact is more noticeable on larger images.

#### Tiny VAE

Enable [madebyollin/taesd](https://github.com/madebyollin/taesd) for almost instant latent decoding with a minor loss in detail. Useful for development.

#### Clip Skip

When enabled, the last CLIP layer is skipped. This _can_ improve image quality with anime models.

#### Prompt Truncation

When enabled, prompts will be truncated to CLIP's limit of 77 tokens. By default this is _disabled_, so Compel will chunk prompts into segments rather than cutting them off.
