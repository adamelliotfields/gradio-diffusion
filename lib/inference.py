import functools
import inspect
import json
import os
import re
import time
from datetime import datetime
from itertools import product
from typing import Callable, TypeVar

import anyio
import numpy as np
import spaces
import torch
from anyio import Semaphore
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from PIL import Image
from typing_extensions import ParamSpec

from .loader import Loader

__import__("warnings").filterwarnings("ignore", category=FutureWarning, module="transformers")
__import__("transformers").logging.set_verbosity_error()

T = TypeVar("T")
P = ParamSpec("P")

MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)

with open("./data/styles.json") as f:
    STYLES = json.load(f)


# like the original but supports args and kwargs instead of a dict
# https://github.com/huggingface/huggingface-inference-toolkit/blob/0.2.0/src/huggingface_inference_toolkit/async_utils.py
async def async_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    async with MAX_THREADS_GUARD:
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        partial_fn = functools.partial(fn, **bound_args.arguments)
        return await anyio.to_thread.run_sync(partial_fn)


# parse prompts with arrays
def parse_prompt(prompt: str) -> list[str]:
    arrays = re.findall(r"\[\[(.*?)\]\]", prompt)

    if not arrays:
        return [prompt]

    tokens = [item.split(",") for item in arrays]
    combinations = list(product(*tokens))
    prompts = []

    for combo in combinations:
        current_prompt = prompt
        for i, token in enumerate(combo):
            current_prompt = current_prompt.replace(f"[[{arrays[i]}]]", token.strip(), 1)
        prompts.append(current_prompt)
    return prompts


def apply_style(prompt, style_id, negative=False):
    global STYLES
    if not style_id or style_id == "None":
        return prompt
    for style in STYLES:
        if style["id"] == style_id:
            if negative:
                return prompt + " . " + style["negative_prompt"]
            else:
                return style["prompt"].format(prompt=prompt)
    return prompt


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
    duration = 15
    scale = kwargs.get("scale", 1)
    num_images = kwargs.get("num_images", 1)
    if scale == 4:
        duration += 5
    return duration * num_images


@spaces.GPU(duration=gpu_duration)
def generate(
    positive_prompt,
    negative_prompt="",
    image_prompt=None,
    ip_image=None,
    ip_face=False,
    embeddings=[],
    style=None,
    seed=None,
    model="Lykon/dreamshaper-8",
    scheduler="DEIS 2M",
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
    Info: Callable[[str], None] = None,
    Error=Exception,
    progress=None,
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**64)

    DEVICE = torch.device("cuda")

    EMBEDDINGS_TYPE = (
        ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
        if clip_skip
        else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    )

    KIND = "img2img" if image_prompt is not None else "txt2img"

    CURRENT_IMAGE = 1

    if ip_image:
        IP_ADAPTER = "full-face" if ip_face else "plus"
    else:
        IP_ADAPTER = ""

    if progress is not None:
        TQDM = False
        progress((0, inference_steps), desc=f"Generating image {CURRENT_IMAGE}/{num_images}")
    else:
        TQDM = True

    def callback_on_step_end(pipeline, step, timestep, latents):
        nonlocal CURRENT_IMAGE
        if progress is None:
            return latents
        strength = denoising_strength if KIND == "img2img" else 1
        total_steps = min(int(inference_steps * strength), inference_steps)
        current_step = step + 1
        progress(
            (current_step, total_steps),
            desc=f"Generating image {CURRENT_IMAGE}/{num_images}",
        )
        if current_step == total_steps:
            CURRENT_IMAGE += 1
        return latents

    start = time.perf_counter()
    loader = Loader()
    pipe, upscaler = loader.load(
        KIND,
        IP_ADAPTER,
        model,
        scheduler,
        karras,
        taesd,
        freeu,
        deepcache,
        scale,
        TQDM,
        DEVICE,
    )

    # load embeddings and append to negative prompt
    embeddings_dir = os.path.join(os.path.dirname(__file__), "..", "embeddings")
    embeddings_dir = os.path.abspath(embeddings_dir)
    for embedding in embeddings:
        try:
            # wrap embeddings in angle brackets
            pipe.load_textual_inversion(
                pretrained_model_name_or_path=f"{embeddings_dir}/{embedding}.pt",
                token=f"<{embedding}>",
            )
            # boost embeddings slightly
            negative_prompt = (
                f"{negative_prompt}, (<{embedding}>)1.1"
                if negative_prompt
                else f"(<{embedding}>)1.1"
            )
        except (EnvironmentError, HFValidationError, RepositoryNotFoundError):
            raise Error(f"Invalid embedding: <{embedding}>")

    # prompt embeds
    compel = Compel(
        device=pipe.device,
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        returned_embeddings_type=EMBEDDINGS_TYPE,
        dtype_for_device_getter=lambda _: pipe.dtype,
        textual_inversion_manager=DiffusersTextualInversionManager(pipe),
    )

    images = []
    current_seed = seed

    try:
        styled_negative_prompt = apply_style(negative_prompt, style, negative=True)
        neg_embeds = compel(styled_negative_prompt)
    except PromptParser.ParsingException:
        raise Error("ParsingException: Invalid negative prompt")

    for i in range(num_images):
        # seeded generator for each iteration
        generator = torch.Generator(device=pipe.device).manual_seed(current_seed)

        try:
            all_positive_prompts = parse_prompt(positive_prompt)
            prompt_index = i % len(all_positive_prompts)
            pos_prompt = all_positive_prompts[prompt_index]
            styled_pos_prompt = apply_style(pos_prompt, style)
            pos_embeds = compel(styled_pos_prompt)
            pos_embeds, neg_embeds = compel.pad_conditioning_tensors_to_same_length(
                [pos_embeds, neg_embeds]
            )
        except PromptParser.ParsingException:
            raise Error("ParsingException: Invalid prompt")

        kwargs = {
            "width": width,
            "height": height,
            "generator": generator,
            "prompt_embeds": pos_embeds,
            "guidance_scale": guidance_scale,
            "negative_prompt_embeds": neg_embeds,
            "num_inference_steps": inference_steps,
            "output_type": "np" if scale > 1 else "pil",
        }

        if progress is not None:
            kwargs["callback_on_step_end"] = callback_on_step_end

        if KIND == "img2img":
            kwargs["strength"] = denoising_strength
            kwargs["image"] = prepare_image(image_prompt, (width, height))

        if IP_ADAPTER:
            # don't resize full-face images
            size = None if ip_face else (width, height)
            kwargs["ip_adapter_image"] = prepare_image(ip_image, size)

        try:
            image = pipe(**kwargs).images[0]
            if scale > 1:
                image = upscaler.predict(image)
            images.append((image, str(current_seed)))
        finally:
            pipe.unload_textual_inversion()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # increment seed for next image
        current_seed += 1

    diff = time.perf_counter() - start
    if Info:
        Info(f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {diff:.2f}s")
    return images
