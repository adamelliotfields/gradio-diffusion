import functools
import inspect
import json
import os
import time
from contextlib import contextmanager
from typing import Callable, TypeVar

import anyio
import cv2
import httpx
import numpy as np
from anyio import Semaphore
from diffusers.utils import logging as diffusers_logging
from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled
from PIL import Image
from transformers import logging as transformers_logging
from typing_extensions import ParamSpec

from .logger import Logger

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


def download_civit_file(lora_id, version_id, file_path=".", token=None):
    base_url = "https://civitai.com/api/download/models"
    file = f"{file_path}/{lora_id}.{version_id}.safetensors"
    log = Logger("download_civit_file")

    if os.path.exists(file):
        return

    try:
        params = {"token": token}
        response = httpx.get(
            f"{base_url}/{version_id}",
            timeout=None,
            params=params,
            follow_redirects=True,
        )

        response.raise_for_status()
        os.makedirs(file_path, exist_ok=True)

        with open(file, "wb") as f:
            f.write(response.content)
    except httpx.HTTPStatusError as e:
        log.error(f"{e.response.status_code} {e.response.text}")
    except httpx.RequestError as e:
        log.error(f"RequestError: {e}")


# resize an image while preserving the aspect ratio (size is width-first)
def resize_image(image, size):
    if isinstance(image, Image.Image):
        image = np.array(image)

    H, W, _ = image.shape
    W = float(W)
    H = float(H)
    target_W, target_H = size

    # Use the smaller scaling factor to maintain the aspect ratio.
    k_w = float(target_W) / W
    k_h = float(target_H) / H
    k = min(k_w, k_h)

    new_W = int(np.round(W * k / 64.0)) * 64
    new_H = int(np.round(H * k / 64.0)) * 64
    img = cv2.resize(
        image,
        (new_W, new_H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


# ensure image is within bounds
def get_valid_size(image, step=64, low=512, high=4096):
    def round_down(x, step=step):
        return int((x // step) * step)

    def clamp_range(x, low=low, high=high):
        return max(low, min(x, high))

    if isinstance(image, Image.Image):
        image = np.array(image)

    H, W = image.shape[:2]
    ar = W / H

    # try width first
    if W > H:
        new_W = round_down(clamp_range(W))
        new_H = round_down(new_W / ar)
    else:
        new_H = round_down(clamp_range(H))
        new_W = round_down(new_H * ar)

    # if the new size is out of bounds, try the other dimension
    if new_W < low or new_W > high:
        new_W = round_down(clamp_range(W))
        new_H = round_down(new_W / ar)
    if new_H < low or new_H > high:
        new_H = round_down(clamp_range(H))
        new_W = round_down(new_H * ar)
    return (new_W, new_H)


# like the original but supports args and kwargs instead of a dict
# https://github.com/huggingface/huggingface-inference-toolkit/blob/0.2.0/src/huggingface_inference_toolkit/async_utils.py
async def async_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    async with MAX_THREADS_GUARD:
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        partial_fn = functools.partial(fn, **bound_args.arguments)
        return await anyio.to_thread.run_sync(partial_fn)
