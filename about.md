## Usage

Enter a prompt and click `Generate`. [Civitai](https://civitai.com) has an excellent guide on [prompting](https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/).

### Compel

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel), enabling weighting and blending. See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md).

### Arrays

Arrays allow you to generate different images from a single prompt. For example, `a cute [[cat,corgi,koala]]` will expand into 3 prompts. Note that it only works for the positive prompt. You must also increase `Images` to generate more than 1 image at a time. Inspired by [Fooocus](https://github.com/lllyasviel/Fooocus/pull/1503).

### Autoincrement

If `Autoincrement` is checked, the seed will be incremented for each image. When using arrays, you might want to uncheck this so the same seed is used for each prompt variation.

## Models

Models are diffusion pipelines. All use `float16`. Recommended settings are shown below:

* [fluently/fluently-v4](https://huggingface.co/fluently/Fluently-v4)
  - sampler: DPM++ 2M, guidance: 5-7, steps: 20-30
* [lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8)
  - sampler: DEIS 2M
* [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4)
  - sampler: PNDM
* [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - sampler: PNDM
* [sg161222/realistic_vision_v5.1](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)
  - sampler: DPM++ 2M, guidance: 4-7

### Schedulers

All are based on [k_diffusion](https://github.com/crowsonkb/k-diffusion) except [DEIS](https://github.com/qsh-zh/deis) and [DPM++](https://github.com/LuChengTHU/dpm-solver). Optionally, the [Karras](https://arxiv.org/abs/2206.00364) noise schedule can be used.

* [DEIS 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/deis)
* [DPM++ 2M](https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver)
* [DPM2 a](https://huggingface.co/docs/diffusers/api/schedulers/dpm_discrete_ancestral)
* [Euler a](https://huggingface.co/docs/diffusers/en/api/schedulers/euler_ancestral)
* [Heun](https://huggingface.co/docs/diffusers/api/schedulers/heun)
* [LMS](https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete)
* [PNDM](https://huggingface.co/docs/diffusers/api/schedulers/pndm)

### VAE

All models use [madebyollin/taesd](https://huggingface.co/madebyollin/taesd) for speed.

## TODO

- [ ] Performance improvements
- [ ] Support `bfloat16`
- [ ] Support LoRA
- [ ] Add VAE radio
- [ ] Add styles
- [ ] CLIP skip
- [ ] [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/)
- [ ] Badges
