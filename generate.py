import json
import re
import time
from contextlib import contextmanager
from datetime import datetime
from itertools import product
from os import environ
from types import MethodType
from typing import Callable

import spaces
import tomesd
import torch
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from DeepCache import DeepCacheSDHelper
from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from diffusers.models import AutoencoderKL, AutoencoderTiny
from tgate.SD import tgate as tgate_sd
from tgate.SD_DeepCache import tgate as tgate_sd_deepcache
from torch._dynamo import OptimizedModule

# some models use the deprecated CLIPFeatureExtractor class (should use CLIPImageProcessor)
__import__("warnings").filterwarnings("ignore", category=FutureWarning, module="transformers")
__import__("transformers").logging.set_verbosity_error()

ZERO_GPU = (
    environ.get("SPACES_ZERO_GPU", "").lower() == "true"
    or environ.get("SPACES_ZERO_GPU", "") == "1"
)

EMBEDDINGS = {
    "./embeddings/bad_prompt_version2.pt": "<bad_prompt>",
    "./embeddings/BadDream.pt": "<bad_dream>",
    "./embeddings/FastNegativeV2.pt": "<fast_negative>",
    "./embeddings/negative_hand.pt": "<negative_hand>",
    "./embeddings/UnrealisticDream.pt": "<unrealistic_dream>",
}

with open("./styles/twri.json") as f:
    styles = json.load(f)


# inspired by ComfyUI
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_management.py
class Loader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.cpu = torch.device("cpu")
            cls._instance.gpu = torch.device("cuda")
            cls._instance.pipe = None
        return cls._instance

    def _load_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipe, "deepcache")

        if has_deepcache and self.pipe.deepcache.params["cache_interval"] == interval:
            return self.pipe.deepcache
        if has_deepcache:
            self.pipe.deepcache.disable()
        else:
            self.pipe.deepcache = DeepCacheSDHelper(pipe=self.pipe)

        self.pipe.deepcache.set_params(cache_interval=interval)
        self.pipe.deepcache.enable()
        return self.pipe.deepcache

    def _load_tgate(self):
        has_tgate = hasattr(self.pipe, "tgate")
        has_deepcache = hasattr(self.pipe, "deepcache")

        if not has_tgate:
            self.pipe.tgate = MethodType(
                tgate_sd_deepcache if has_deepcache else tgate_sd,
                self.pipe,
            )
        return self.pipe.tgate

    def _load_vae(self, model_name=None, taesd=False, dtype=None):
        vae_type = type(self.pipe.vae)
        is_kl = issubclass(vae_type, (AutoencoderKL, OptimizedModule))
        is_tiny = issubclass(vae_type, AutoencoderTiny)

        # by default all models use KL
        if is_kl and taesd:
            # can't compile tiny VAE
            print("Switching to Tiny VAE...")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                pretrained_model_name_or_path="madebyollin/taesd",
                use_safetensors=True,
                torch_dtype=dtype,
            ).to(self.gpu)
            return self.pipe.vae

        if is_tiny and not taesd:
            print("Switching to KL VAE...")
            self.pipe.vae = torch.compile(
                fullgraph=True,
                mode="reduce-overhead",
                model=AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    use_safetensors=True,
                    torch_dtype=dtype,
                    subfolder="vae",
                ).to(self.gpu),
            )
        return self.pipe.vae

    def load(self, model, scheduler, karras, taesd, deepcache_interval, dtype=None):
        model_lower = model.lower()

        schedulers = {
            "DEIS 2M": DEISMultistepScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "DPM2 a": KDPM2AncestralDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "Heun": HeunDiscreteScheduler,
            "LMS": LMSDiscreteScheduler,
            "PNDM": PNDMScheduler,
        }

        scheduler_kwargs = {
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "use_karras_sigmas": karras,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "steps_offset": 1,
        }

        if scheduler == "PNDM" or scheduler == "Euler a":
            del scheduler_kwargs["use_karras_sigmas"]

        pipe_kwargs = {
            "scheduler": schedulers[scheduler](**scheduler_kwargs),
            "pretrained_model_name_or_path": model_lower,
            "requires_safety_checker": False,
            "use_safetensors": True,
            "safety_checker": None,
            "torch_dtype": dtype,
        }

        # already loaded
        if self.pipe is not None:
            model_name = self.pipe.config._name_or_path
            same_model = model_name.lower() == model_lower
            same_scheduler = isinstance(self.pipe.scheduler, schedulers[scheduler])
            same_karras = (
                not hasattr(self.pipe.scheduler.config, "use_karras_sigmas")
                or self.pipe.scheduler.config.use_karras_sigmas == karras
            )

            if same_model:
                if not same_scheduler:
                    print(f"Switching to {scheduler}...")
                if not same_karras:
                    print(f"{'Enabling' if karras else 'Disabling'} Karras sigmas...")
                if not same_scheduler or not same_karras:
                    self.pipe.scheduler = schedulers[scheduler](**scheduler_kwargs)

                self._load_vae(model_lower, taesd, dtype)
                self._load_deepcache(interval=deepcache_interval)
                self._load_tgate()
                return self.pipe
            else:
                print(f"Unloading {model_name.lower()}...")
                self.pipe = None
                torch.cuda.empty_cache()

        # no fp16 available
        if not ZERO_GPU and model_lower not in [
            "sg161222/realistic_vision_v5.1_novae",
            "prompthero/openjourney-v4",
            "linaqruf/anything-v3-1",
        ]:
            pipe_kwargs["variant"] = "fp16"

        print(f"Loading {model_lower} with {'Tiny' if taesd else 'KL'} VAE...")
        self.pipe = StableDiffusionPipeline.from_pretrained(**pipe_kwargs).to(self.gpu)
        self._load_vae(model_lower, taesd, dtype)
        self._load_deepcache(interval=deepcache_interval)
        self._load_tgate()
        self.pipe.load_textual_inversion(
            pretrained_model_name_or_path=list(EMBEDDINGS.keys()),
            tokens=list(EMBEDDINGS.values()),
        )
        return self.pipe


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


def apply_style(prompt, style_name, negative=False):
    global styles
    if not style_name or style_name == "None":
        return prompt
    for style in styles:
        if style["name"] == style_name:
            if negative:
                return prompt + " . " + style["negative_prompt"]
            else:
                return style["prompt"].format(prompt=prompt)
    return prompt


# 1024x1024 for 50 steps can take ~10s each
@spaces.GPU(duration=44)
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
    tgate_step=0,
    tome_ratio=0,
    log: Callable[[str], None] = None,
    Error=Exception,
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**64)

    TORCH_DTYPE = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
        pipe = loader.load(model, scheduler, karras, taesd, deepcache_interval, TORCH_DTYPE)

        # prompt embeds
        compel = Compel(
            textual_inversion_manager=DiffusersTextualInversionManager(pipe),
            dtype_for_device_getter=lambda _: TORCH_DTYPE,
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
                # cap the tgate step
                gate_step = min(
                    tgate_step if tgate_step > 0 else inference_steps,
                    inference_steps,
                )
                result = pipe.tgate(
                    num_inference_steps=inference_steps,
                    negative_prompt_embeds=neg_embeds,
                    guidance_scale=guidance_scale,
                    prompt_embeds=pos_embeds,
                    gate_step=gate_step,
                    generator=generator,
                    height=height,
                    width=width,
                )
                images.append((result.images[0], str(current_seed)))

            if increment_seed:
                current_seed += 1

        if ZERO_GPU:
            # spaces always start fresh
            loader.pipe = None

        end = time.perf_counter()
        diff = end - start
        if log:
            log(f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {diff:.2f}s")
        return images
