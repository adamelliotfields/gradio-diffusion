import gc
import os
import re
import time
from datetime import datetime
from itertools import product

import numpy as np
import torch
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from PIL import Image
from spaces import GPU

from .config import Config
from .loader import Loader
from .logger import Logger
from .utils import load_json


def parse_prompt_with_arrays(prompt: str) -> list[str]:
    arrays = re.findall(r"\[\[(.*?)\]\]", prompt)

    if not arrays:
        return [prompt]

    tokens = [item.split(",") for item in arrays]  # [("a", "b"), ("1", "2")]
    combinations = list(product(*tokens))  # [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2")]

    # find all the arrays in the prompt and replace them with tokens
    prompts = []
    for combo in combinations:
        current_prompt = prompt
        for i, token in enumerate(combo):
            current_prompt = current_prompt.replace(f"[[{arrays[i]}]]", token.strip(), 1)
        prompts.append(current_prompt)
    return prompts


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


def prepare_image(input, size=None):
    image = None
    if isinstance(input, Image.Image):
        image = input
    if isinstance(input, np.ndarray):
        image = Image.fromarray(input)
    if isinstance(input, str):
        if os.path.isfile(input):
            image = Image.open(input)
    if image is not None:
        image = image.convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    if image is not None:
        return image
    else:
        raise ValueError("Invalid image prompt")


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


@GPU(duration=gpu_duration)
def generate(
    positive_prompt,
    negative_prompt="",
    image_prompt=None,
    ip_image_prompt=None,
    ip_face=False,
    lora_1=None,
    lora_1_weight=0.0,
    lora_2=None,
    lora_2_weight=0.0,
    embeddings=[],
    style=None,
    seed=None,
    model="Lykon/dreamshaper-8",
    scheduler="DDIM",
    width=512,
    height=512,
    guidance_scale=7.5,
    inference_steps=40,
    denoising_strength=0.8,
    deepcache=1,
    scale=1,
    num_images=1,
    karras=False,
    taesd=False,
    freeu=False,
    clip_skip=False,
    Error=Exception,
    Info=None,
    progress=None,
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**64)

    CURRENT_STEP = 0
    CURRENT_IMAGE = 1

    KIND = "img2img" if image_prompt is not None else "txt2img"

    EMBEDDINGS_TYPE = (
        ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
        if clip_skip
        else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    )

    if ip_image_prompt:
        IP_ADAPTER = "full-face" if ip_face else "plus"
    else:
        IP_ADAPTER = ""

    # custom progress bar for multiple images
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

    start = time.perf_counter()
    log = Logger("generate")
    log.info(f"Generating {num_images} image{'s' if num_images > 1 else ''}")

    if Config.ZERO_GPU and progress is not None:
        progress((100, 100), desc="ZeroGPU init")

    loader = Loader()
    loader.load(
        KIND,
        IP_ADAPTER,
        model,
        scheduler,
        deepcache,
        scale,
        karras,
        taesd,
        freeu,
        progress,
    )

    if loader.pipe is None:
        raise Error(f"Error loading {model}")

    pipe = loader.pipe
    upscaler = loader.upscaler

    # load loras
    loras = []
    weights = []
    loras_and_weights = [(lora_1, lora_1_weight), (lora_2, lora_2_weight)]
    loras_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "loras"))
    for lora, weight in loras_and_weights:
        if lora and lora.lower() != "none" and lora not in loras:
            config = Config.CIVIT_LORAS.get(lora)
            if config:
                try:
                    pipe.load_lora_weights(
                        loras_dir,
                        adapter_name=lora,
                        weight_name=f"{lora}.{config['model_version_id']}.safetensors",
                    )
                    weights.append(weight)
                    loras.append(lora)
                except Exception:
                    raise Error(f"Error loading {config['name']} LoRA")

    # unload after generating or if there was an error
    try:
        if loras:
            pipe.set_adapters(loras, adapter_weights=weights)
    except Exception:
        pipe.unload_lora_weights()
        raise Error("Error setting LoRA weights")

    # load embeddings
    embeddings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "embeddings"))
    for embedding in embeddings:
        try:
            # wrap embeddings in angle brackets
            pipe.load_textual_inversion(
                pretrained_model_name_or_path=f"{embeddings_dir}/{embedding}.pt",
                token=f"<{embedding}>",
            )
        except (EnvironmentError, HFValidationError, RepositoryNotFoundError):
            raise Error(f"Invalid embedding: {embedding}")

    # prompt embeds
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
    for i in range(num_images):
        try:
            generator = torch.Generator(device=pipe.device).manual_seed(current_seed)

            positive_prompts = parse_prompt_with_arrays(positive_prompt)
            index = i % len(positive_prompts)
            positive_styled, negative_styled = apply_style(
                positive_prompts[index],
                negative_prompt,
                style,
            )

            if negative_styled.startswith("(), "):
                negative_styled = negative_styled[4:]

            for lora in loras:
                positive_styled += f", {Config.CIVIT_LORAS[lora]['trigger']}"

            for embedding in embeddings:
                negative_styled += f", <{embedding}>"

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

        if KIND == "img2img":
            kwargs["strength"] = denoising_strength
            kwargs["image"] = prepare_image(image_prompt, (width, height))

        if IP_ADAPTER:
            # don't resize full-face images since they are usually square crops
            size = None if ip_face else (width, height)
            kwargs["ip_adapter_image"] = prepare_image(ip_image_prompt, size)

        try:
            image = pipe(**kwargs).images[0]
            if scale > 1:
                image = upscaler.predict(image)
            images.append((image, str(current_seed)))
            current_seed += 1
        except Exception as e:
            raise Error(f"{e}")
        finally:
            if embeddings:
                pipe.unload_textual_inversion()
            if loras:
                pipe.unload_lora_weights()
            CURRENT_STEP = 0
            CURRENT_IMAGE += 1

    # cleanup
    loader.collect()
    gc.collect()

    diff = time.perf_counter() - start
    msg = f"Generating {len(images)} image{'s' if len(images) > 1 else ''} done in {diff:.2f}s"
    log.info(msg)
    if Info:
        Info(msg)
    return images
