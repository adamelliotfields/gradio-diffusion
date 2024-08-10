import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from itertools import product
from typing import Callable

import spaces
import tomesd
import torch
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser

from .loader import Loader

__import__("warnings").filterwarnings("ignore", category=FutureWarning, module="diffusers")
__import__("warnings").filterwarnings("ignore", category=FutureWarning, module="transformers")
__import__("transformers").logging.set_verbosity_error()

ZERO_GPU = (
    os.environ.get("SPACES_ZERO_GPU", "").lower() == "true"
    or os.environ.get("SPACES_ZERO_GPU", "") == "1"
)

with open("./styles/twri.json") as f:
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


@spaces.GPU(duration=40)
def generate(
    positive_prompt,
    negative_prompt="",
    style=None,
    seed=None,
    model="runwayml/stable-diffusion-v1-5",
    scheduler="PNDM",
    width=512,
    height=512,
    guidance_scale=7.5,
    inference_steps=50,
    num_images=1,
    karras=False,
    taesd=False,
    clip_skip=False,
    truncate_prompts=False,
    increment_seed=True,
    deepcache_interval=1,
    tome_ratio=0,
    log: Callable[[str], None] = None,
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

    with torch.inference_mode():
        start = time.perf_counter()
        loader = Loader()
        pipe = loader.load(
            model,
            scheduler,
            karras,
            taesd,
            deepcache_interval,
            DTYPE,
            DEVICE,
        )

        # prompt embeds
        compel = Compel(
            textual_inversion_manager=DiffusersTextualInversionManager(pipe),
            dtype_for_device_getter=lambda _: DTYPE,
            returned_embeddings_type=EMBEDDINGS_TYPE,
            truncate_long_prompts=truncate_prompts,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            device=pipe.device,
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

            with token_merging(pipe, tome_ratio=tome_ratio):
                try:
                    image = pipe(
                        num_inference_steps=inference_steps,
                        negative_prompt_embeds=neg_embeds,
                        guidance_scale=guidance_scale,
                        prompt_embeds=pos_embeds,
                        generator=generator,
                        height=height,
                        width=width,
                    ).images[0]
                    images.append((image, str(current_seed)))
                finally:
                    if not ZERO_GPU:
                        torch.cuda.empty_cache()

            if increment_seed:
                current_seed += 1

        if ZERO_GPU:
            # spaces always start fresh
            loader.pipe = None

        diff = time.perf_counter() - start
        if log:
            log(f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {diff:.2f}s")
        return images
