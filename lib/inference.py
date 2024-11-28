import os
import time
from datetime import datetime

import torch
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from spaces import GPU

from .config import Config
from .loader import Loader
from .logger import Logger
from .utils import (
    annotate_image,
    clear_cuda_cache,
    load_json,
    resize_image,
    safe_progress,
    timer,
)


# Inject prompts into style templates
def apply_style(positive_prompt, negative_prompt, style_id="none"):
    if style_id.lower() == "none":
        return (positive_prompt, negative_prompt)

    styles = load_json("./data/styles.json")
    style = styles.get(style_id)
    if style is None:
        return (positive_prompt, negative_prompt)

    style_base = styles.get("_base", {})
    return (
        style.get("positive")
        .format(prompt=positive_prompt, _base=style_base.get("positive"))
        .strip(),
        style.get("negative")
        .format(prompt=negative_prompt, _base=style_base.get("negative"))
        .strip(),
    )


# Dynamic signature for the GPU duration function
def gpu_duration(**kwargs):
    loading = 20
    duration = 10
    width = kwargs.get("width", 512)
    height = kwargs.get("height", 512)
    scale = kwargs.get("scale", 1)
    num_images = kwargs.get("num_images", 1)
    size = width * height
    if size > 500_000:
        duration += 5
    if scale == 4:
        duration += 5
    return loading + (duration * num_images)


# Request GPU when deployed to Hugging Face
@GPU(duration=gpu_duration)
def generate(
    positive_prompt,
    negative_prompt="",
    image_prompt=None,
    control_image_prompt=None,
    ip_image_prompt=None,
    style=None,
    seed=None,
    model="Lykon/dreamshaper-8",
    scheduler="DDIM",
    annotator="canny",
    width=512,
    height=512,
    guidance_scale=7.5,
    inference_steps=40,
    denoising_strength=0.8,
    deepcache=1,
    scale=1,
    num_images=1,
    karras=False,
    ip_face=False,
    negative_embedding=False,
    Error=Exception,
    Info=None,
    progress=None,
):
    start = time.perf_counter()
    log = Logger("generate")
    log.info(f"Generating {num_images} image{'s' if num_images > 1 else ''}")

    if Config.ZERO_GPU:
        safe_progress(progress, 100, 100, "ZeroGPU init")

    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**64)

    CURRENT_STEP = 0
    CURRENT_IMAGE = 1

    KIND = "img2img" if image_prompt is not None else "txt2img"
    KIND = f"controlnet_{KIND}" if control_image_prompt is not None else KIND

    EMBEDDINGS_TYPE = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED

    if ip_image_prompt:
        IP_ADAPTER = "full-face" if ip_face else "plus"
    else:
        IP_ADAPTER = ""

    # Custom progress bar for multiple images
    def callback_on_step_end(pipeline, step, timestep, latents):
        nonlocal CURRENT_STEP, CURRENT_IMAGE
        if progress is not None:
            # calculate total steps for img2img based on denoising strength
            strength = denoising_strength if KIND == "img2img" else 1
            total_steps = min(int(inference_steps * strength), inference_steps)
            CURRENT_STEP = step + 1
            progress(
                (CURRENT_STEP, total_steps),
                desc=f"Generating image {CURRENT_IMAGE}/{num_images}",
            )
        return latents

    loader = Loader()
    loader.load(
        KIND,
        IP_ADAPTER,
        model,
        scheduler,
        annotator,
        deepcache,
        scale,
        karras,
        progress,
    )

    if loader.pipe is None:
        raise Error(f"Error loading {model}")

    pipe = loader.pipe
    upscaler = loader.upscaler

    # Load negative embedding if requested
    if negative_embedding:
        embeddings_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "embeddings")
        )
        embedding = Config.NEGATIVE_EMBEDDING
        try:
            pipe.load_textual_inversion(
                pretrained_model_name_or_path=f"{embeddings_dir}/{embedding}.pt",
                token=f"<{embedding}>",
            )
        except (EnvironmentError, HFValidationError, RepositoryNotFoundError):
            raise Error(f"Invalid embedding: {embedding}")

    # Embed prompts with weights
    compel = Compel(
        device=pipe.device,
        tokenizer=pipe.tokenizer,
        truncate_long_prompts=False,
        text_encoder=pipe.text_encoder,
        returned_embeddings_type=EMBEDDINGS_TYPE,
        dtype_for_device_getter=lambda _: pipe.dtype,
        textual_inversion_manager=DiffusersTextualInversionManager(pipe),
    )

    images = []
    current_seed = seed
    safe_progress(progress, 0, num_images, f"Generating image 0/{num_images}")

    for i in range(num_images):
        try:
            generator = torch.Generator(device=pipe.device).manual_seed(current_seed)
            positive_styled, negative_styled = apply_style(positive_prompt, negative_prompt, style)

            # User didn't provide a negative prompt
            if negative_styled.startswith("(), "):
                negative_styled = negative_styled[4:]

            if negative_embedding:
                negative_styled += f", <{Config.NEGATIVE_EMBEDDING}>"

            positive_embeds, negative_embeds = compel.pad_conditioning_tensors_to_same_length(
                [compel(positive_styled), compel(negative_styled)]
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

        if progress is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end

        # Resizing so the initial latents are the same size as the generated image
        if KIND == "img2img":
            kwargs["strength"] = denoising_strength
            kwargs["image"] = resize_image(image_prompt, (width, height))

        if KIND == "controlnet_txt2img":
            kwargs["image"] = annotate_image(control_image_prompt, annotator)

        if KIND == "controlnet_img2img":
            kwargs["control_image"] = annotate_image(control_image_prompt, annotator)

        if IP_ADAPTER:
            kwargs["ip_adapter_image"] = resize_image(ip_image_prompt)

        try:
            image = pipe(**kwargs).images[0]
            images.append((image, str(current_seed)))
            current_seed += 1
        finally:
            if negative_embedding:
                pipe.unload_textual_inversion()
            CURRENT_STEP = 0
            CURRENT_IMAGE += 1

    # Upscale
    if scale > 1:
        msg = f"Upscaling {scale}x"
        with timer(msg, logger=log.info):
            safe_progress(progress, 0, num_images, desc=msg)
            for i, image in enumerate(images):
                image = upscaler.predict(image[0])
                images[i] = image
                safe_progress(progress, i + 1, num_images, desc=msg)

    # Flush memory after generating
    clear_cuda_cache()

    end = time.perf_counter()
    msg = f"Generating {len(images)} image{'s' if len(images) > 1 else ''} took {end - start:.2f}s"
    log.info(msg)

    # Alert if notifier provided
    if Info:
        Info(msg)

    return images
