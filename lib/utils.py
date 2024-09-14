import functools
import inspect
import json
from typing import Callable, TypeVar

import anyio
from anyio import Semaphore
from huggingface_hub._snapshot_download import snapshot_download
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)


@functools.lru_cache()
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@functools.lru_cache()
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def download_repo_files(repo_id, allow_patterns, token=None):
    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=None,
    )


# like the original but supports args and kwargs instead of a dict
# https://github.com/huggingface/huggingface-inference-toolkit/blob/0.2.0/src/huggingface_inference_toolkit/async_utils.py
async def async_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    async with MAX_THREADS_GUARD:
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        partial_fn = functools.partial(fn, **bound_args.arguments)
        return await anyio.to_thread.run_sync(partial_fn)
