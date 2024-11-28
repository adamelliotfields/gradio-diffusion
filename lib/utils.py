import functools
import inspect
import json
import os
import time
from contextlib import contextmanager
from typing import Callable, Tuple, TypeVar

import anyio
import numpy as np
import torch
from anyio import Semaphore
from diffusers.utils import logging as diffusers_logging
from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled
from PIL import Image
from transformers import logging as transformers_logging
from typing_extensions import ParamSpec

from .annotators import CannyAnnotator

T = TypeVar("T")
P = ParamSpec("P")

MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)


@contextmanager
def timer(message="Operation", logger=print):
    start = time.perf_counter()
    logger(message)
    try:
        yield
    finally:
        end = time.perf_counter()
        logger(f"{message} took {end - start:.2f}s")


@functools.lru_cache()
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@functools.lru_cache()
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def disable_progress_bars():
    transformers_logging.disable_progress_bar()
    diffusers_logging.disable_progress_bar()


def enable_progress_bars():
    # warns if `HF_HUB_DISABLE_PROGRESS_BARS` env var is not None
    transformers_logging.enable_progress_bar()
    diffusers_logging.enable_progress_bar()


def safe_progress(progress, current=0, total=0, desc=""):
    if progress is not None:
        progress((current, total), desc=desc)


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def download_repo_files(repo_id, allow_patterns, token=None):
    was_disabled = are_progress_bars_disabled()
    enable_progress_bars()
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=None,
    )
    if was_disabled:
        disable_progress_bars()
    return snapshot_path


def image_to_pil(image: Image.Image):
    """Converts various image inputs to RGB PIL Image."""
    if isinstance(image, str) and os.path.isfile(image):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError("Invalid image input")


def get_valid_image_size(
    width: int,
    height: int,
    step=64,
    min_size=512,
    max_size=4096,
):
    """Get new image dimensions while preserving aspect ratio."""

    def round_down(x):
        return int((x // step) * step)

    def clamp(x):
        return max(min_size, min(x, max_size))

    aspect_ratio = width / height

    # try width first
    if width > height:
        new_width = round_down(clamp(width))
        new_height = round_down(new_width / aspect_ratio)
    else:
        new_height = round_down(clamp(height))
        new_width = round_down(new_height * aspect_ratio)

    # if new dimensions are out of bounds, try height
    if not min_size <= new_width <= max_size:
        new_width = round_down(clamp(width))
        new_height = round_down(new_width / aspect_ratio)
    if not min_size <= new_height <= max_size:
        new_height = round_down(clamp(height))
        new_width = round_down(new_height * aspect_ratio)

    return (new_width, new_height)


def resize_image(
    image: Image.Image,
    size: Tuple[int, int] = None,
    resampling: Image.Resampling = None,
):
    """Resize image with proper interpolation and dimension constraints."""
    image = image_to_pil(image)
    if size is None:
        size = get_valid_image_size(*image.size)
    if resampling is None:
        resampling = Image.Resampling.LANCZOS
    return image.resize(size, resampling)


def annotate_image(image: Image.Image, annotator="canny"):
    """Get the feature map of an image using the specified annotator."""
    size = get_valid_image_size(*image.size)
    image = resize_image(image, size)
    if annotator.lower() == "canny":
        canny = CannyAnnotator()
        return canny(image, size)
    raise ValueError(f"Invalid annotator: {annotator}")


# Like the original but supports args and kwargs instead of a dict
# https://github.com/huggingface/huggingface-inference-toolkit/blob/0.2.0/src/huggingface_inference_toolkit/async_utils.py
async def async_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    async with MAX_THREADS_GUARD:
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        partial_fn = functools.partial(fn, **bound_args.arguments)
        return await anyio.to_thread.run_sync(partial_fn)
