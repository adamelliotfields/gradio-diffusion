## Usage

Enter a prompt and click `Generate`. Read [Civitai](https://civitai.com)'s guide on [prompting](https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/) to learn more.

### Compel

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel), enabling weighting and blending. See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md).

### Embeddings

Textual inversion embeddings are installed for use in the `Negative` prompt.

* [Bad Prompt](https://civitai.com/models/55700/badprompt-negative-embedding): `<bad_prompt>`
* [Negative Hand](https://civitai.com/models/56519/negativehand-negative-embedding): `<negative_hand>`
* [Fast Negative](https://civitai.com/models/71961/fast-negative-embedding-fastnegativev2): `<fast_negative>`
  - includes Negative Hand
* [Bad Dream](https://civitai.com/models/72437?modelVersionId=77169): `<bad_dream>`
* [Unrealistic Dream](https://civitai.com/models/72437?modelVersionId=77173): `<unrealistic_dream>`
  - pair with Fast Negative and the Realistic Vision model

### Arrays

Arrays allow you to generate different images from a single prompt. For example, `a cute [[cat,corgi,koala]]` will expand into 3 prompts. For this to work, you first have to increase `Images`. Note that it only works for the positive prompt. Inspired by [Fooocus](https://github.com/lllyasviel/Fooocus/pull/1503).

### Autoincrement

If `Autoincrement` checked, the seed will be incremented for each image in range `Images`. When using arrays, you might want this disabled so the same seed is used.

## Models

All use `float16` (or `bfloat16` if supported).

* [fluently/fluently-v4](https://huggingface.co/fluently/Fluently-v4)
* [linaqruf/anything-v3-1](https://huggingface.co/linaqruf/anything-v3-1)
* [lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8)
* [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4)
* [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* [sg161222/realistic_vision_v5.1](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)

### Schedulers

All are based on [k_diffusion](https://github.com/crowsonkb/k-diffusion) except [DEIS](https://github.com/qsh-zh/deis) and [DPM++](https://github.com/LuChengTHU/dpm-solver). Optionally, the [Karras](https://arxiv.org/abs/2206.00364) noise schedule can be used.

* [DEIS 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/deis)
* [DPM++ 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver)
* [DPM2 a](https://huggingface.co/docs/diffusers/api/schedulers/dpm_discrete_ancestral)
* [Euler a](https://huggingface.co/docs/diffusers/en/api/schedulers/euler_ancestral)
* [Heun](https://huggingface.co/docs/diffusers/api/schedulers/heun)
* [LMS](https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete)
* [PNDM](https://huggingface.co/docs/diffusers/api/schedulers/pndm)

## Advanced

### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) (Ma et al. 2023) caches UNet layers determined by `Branch` and reuses them every `Interval` steps. Leaving `Branch` on **0** caches lower layers, which provides a greater speedup. An `Interval` of **3** is the best balance between speed and quality; **1** means no cache.

### T-GATE

[T-GATE](https://github.com/HaozheLiu-ST/T-GATE) (Zhang et al. 2024) caches self and cross attention computations up to `Step`. Afterwards, attention is no longer computed and the cache is used, resulting in a noticeable speedup. Works well with DeepCache.

### Tiny VAE

Enable [madebyollin/taesd](https://github.com/madebyollin/taesd) for almost instant latent decoding with a minor loss in detail. Useful for development and ideation.

### Clip Skip

When enabled, the last CLIP layer is skipped. This can improve image quality and is commonly used with anime models.

### Prompt Truncation

When enabled, prompts will be truncated to CLIP's limit of 77 tokens. By default this is disabled, so Compel will chunk prompts into segments rather than cutting them off.
