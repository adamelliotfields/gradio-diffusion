import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from itertools import product
from typing import Callable

import numpy as np
import spaces
import tomesd
import torch
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
from PIL import Image

from .loader import Loader

__import__("warnings").filterwarnings("ignore", category=FutureWarning, module="transformers")
__import__("transformers").logging.set_verbosity_error()

with open("./data/styles.json") as f:
    styles = json.load(f)


# applies tome to the pipeline
@contextmanager
def token_merging(pipe, tome_ratio=0):
    try:
        if tome_ratio > 0:
            tomesd.apply_patch(pipe, max_downsample=1, sx=2, sy=2, ratio=tome_ratio)
        yield
    finally:
        tomesd.remove_patch(pipe)  # idempotent


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
    global styles
    if not style_id or style_id == "None":
        return prompt
    for style in styles:
        if style["id"] == style_id:
            if negative:
                return prompt + " . " + style["negative_prompt"]
            else:
                return style["prompt"].format(prompt=prompt)
    return prompt


def prepare_image(input, size=(512, 512)):
    image = None
    if isinstance(input, Image.Image):
        image = input
    if isinstance(input, np.ndarray):
        image = Image.fromarray(input)
    if isinstance(input, str):
        if os.path.isfile(input):
            image = Image.open(input)
    if image is not None:
        return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    else:
        raise ValueError("Invalid image prompt")


@spaces.GPU(duration=40)
def generate(
    positive_prompt,
    negative_prompt="",
    image_prompt=None,
    embeddings=[],
    style=None,
    seed=None,
    model="runwayml/stable-diffusion-v1-5",
    scheduler="PNDM",
    width=512,
    height=512,
    guidance_scale=7.5,
    inference_steps=50,
    denoising_strength=0.8,
    num_images=1,
    karras=False,
    taesd=False,
    freeu=False,
    clip_skip=False,
    truncate_prompts=False,
    increment_seed=True,
    deepcache=1,
    tome_ratio=0,
    scale=1,
    Info: Callable[[str], None] = None,
    Error=Exception,
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**64)

    DEVICE = torch.device("cuda")

    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_properties(DEVICE).major >= 8
        else torch.float16
    )

    EMBEDDINGS_TYPE = (
        ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
        if clip_skip
        else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    )

    KIND = "img2img" if image_prompt is not None else "txt2img"

    with torch.inference_mode():
        start = time.perf_counter()
        loader = Loader()
        pipe, upscaler = loader.load(
            KIND,
            model,
            scheduler,
            karras,
            taesd,
            freeu,
            deepcache,
            scale,
            DEVICE,
            DTYPE,
        )

        # load embeddings and append to negative prompt
        embeddings_dir = os.path.join(os.path.dirname(__file__), "..", "embeddings")
        embeddings_dir = os.path.abspath(embeddings_dir)
        for embedding in embeddings:
            try:
                pipe.load_textual_inversion(
                    pretrained_model_name_or_path=f"{embeddings_dir}/{embedding}.pt",
                    token=f"<{embedding}>",
                )
                negative_prompt = (
                    f"{negative_prompt}, {embedding}" if negative_prompt else embedding
                )
            except (EnvironmentError, HFValidationError, RepositoryNotFoundError):
                raise Error(f"Invalid embedding: {embedding}")

        # prompt embeds
        compel = Compel(
            device=pipe.device,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            truncate_long_prompts=truncate_prompts,
            dtype_for_device_getter=lambda _: DTYPE,
            returned_embeddings_type=EMBEDDINGS_TYPE,
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

            if KIND == "img2img":
                kwargs["strength"] = denoising_strength
                kwargs["image"] = prepare_image(image_prompt, (width, height))

            with token_merging(pipe, tome_ratio=tome_ratio):
                try:
                    image = pipe(**kwargs).images[0]
                    if scale > 1:
                        image = upscaler.predict(image)
                    images.append((image, str(current_seed)))
                finally:
                    pipe.unload_textual_inversion()
                    torch.cuda.empty_cache()

            if increment_seed:
                current_seed += 1

        diff = time.perf_counter() - start
        if Info:
            Info(f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {diff:.2f}s")
        return images
