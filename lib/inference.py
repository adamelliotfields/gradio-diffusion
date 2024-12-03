import os
import time
from datetime import datetime

import torch
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from gradio import Error, Info, Progress
from spaces import GPU, config

from .loader import get_loader
from .logger import Logger
from .utils import annotate_image, cuda_collect, resize_image, timer


@GPU
def generate(
    positive_prompt="",
    negative_prompt="",
    image_input=None,
    controlnet_input=None,
    ip_adapter_input=None,
    seed=None,
    model="XpucT/Reliberate",
    scheduler="UniPC",
    controlnet_annotator="canny",
    width=512,
    height=512,
    guidance_scale=6.0,
    inference_steps=40,
    denoising_strength=0.8,
    deepcache_interval=1,
    scale=1,
    num_images=1,
    use_karras=False,
    use_ip_adapter_face=False,
    _=Progress(track_tqdm=True),
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    if positive_prompt.strip() == "":
        raise Error("You must enter a prompt")

    start = time.perf_counter()
    log = Logger("generate")
    log.info(f"Generating {num_images} image{'s' if num_images > 1 else ''}...")

    KIND = "img2img" if image_input is not None else "txt2img"
    KIND = f"controlnet_{KIND}" if controlnet_input is not None else KIND

    EMBEDDINGS_TYPE = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED

    FAST_NEGATIVE = "<fast_negative>" in negative_prompt

    if ip_adapter_input:
        IP_KIND = "full-face" if use_ip_adapter_face else "plus"
    else:
        IP_KIND = ""

    # ZeroGPU is serverless so you want ephemeral instances
    # You want a singleton on localhost so the pipeline stays in memory
    loader = get_loader(singleton=not config.Config.zero_gpu)
    loader.load(
        KIND,
        IP_KIND,
        model,
        scheduler,
        controlnet_annotator,
        deepcache_interval,
        scale,
        use_karras,
    )

    pipeline = loader.pipeline
    upscaler = loader.upscaler

    # Probably a typo in the config
    if pipeline is None:
        raise Error(f"Error loading {model}")

    # Load fast negative embedding
    if FAST_NEGATIVE:
        embeddings_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "embeddings")
        )
        pipeline.load_textual_inversion(
            pretrained_model_name_or_path=f"{embeddings_dir}/fast_negative.pt",
            token="<fast_negative>",
        )

    # Embed prompts with weights
    compel = Compel(
        device=pipeline.device,
        tokenizer=pipeline.tokenizer,
        truncate_long_prompts=False,
        text_encoder=pipeline.text_encoder,
        returned_embeddings_type=EMBEDDINGS_TYPE,
        dtype_for_device_getter=lambda _: pipeline.dtype,
        textual_inversion_manager=DiffusersTextualInversionManager(pipeline),
    )

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**64)

    # Increment the seed after each iteration
    images = []
    current_seed = seed

    for i in range(num_images):
        try:
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
            positive_embeds, negative_embeds = compel.pad_conditioning_tensors_to_same_length(
                [compel(positive_prompt), compel(negative_prompt)]
            )
        except PromptParser.ParsingException:
            raise Error("Invalid prompt")

        kwargs = {
            "width": width,
            "height": height,
            "generator": generator,
            "prompt_embeds": positive_embeds,
            "guidance_scale": guidance_scale,
            "num_inference_steps": inference_steps,
            "negative_prompt_embeds": negative_embeds,
            "output_type": "np" if scale > 1 else "pil",
        }

        if KIND == "img2img" or KIND == "controlnet_img2img":
            kwargs["strength"] = denoising_strength
            kwargs["image"] = resize_image(image_input, (width, height))

        if KIND == "controlnet_txt2img":
            kwargs["image"] = annotate_image(controlnet_input, controlnet_annotator)

        if KIND == "controlnet_img2img":
            kwargs["control_image"] = annotate_image(controlnet_input, controlnet_annotator)

        if IP_KIND:
            # No size means preserve aspect ratio
            kwargs["ip_adapter_image"] = resize_image(ip_adapter_input)

        try:
            image = pipeline(**kwargs).images[0]
            images.append((image, str(current_seed)))  # tuple with seed for gallery caption
            current_seed += 1
        finally:
            if FAST_NEGATIVE:
                pipeline.unload_textual_inversion()

    # Upscale
    if scale > 1:
        with timer(f"Upscaling {num_images} images {scale}x", logger=log.info):
            for i, image in enumerate(images):
                image = upscaler.predict(image[0])
                seed = images[i][1]
                images[i] = (image, seed)  # tuple again

    end = time.perf_counter()
    msg = f"Generating {len(images)} image{'s' if len(images) > 1 else ''} took {end - start:.2f}s"
    log.info(msg)

    if Info:
        Info(msg)

    # Flush cache before returning
    cuda_collect()

    return images
