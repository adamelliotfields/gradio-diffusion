import re
from contextlib import contextmanager
from datetime import datetime
from itertools import product
from os import environ
from types import MethodType
from warnings import filterwarnings

import spaces
import torch
from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType
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

# some models use the deprecated CLIPFeatureExtractor class
# should use CLIPImageProcessor instead
filterwarnings("ignore", category=FutureWarning, module="transformers")


class Loader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.cpu = torch.device("cpu")
            cls._instance.gpu = torch.device("cuda")
            cls._instance.pipe = None
        return cls._instance

    def _load_vae(self, model_name=None, taesd=False, dtype=None):
        if taesd:
            # can't compile tiny VAE
            return AutoencoderTiny.from_pretrained(
                pretrained_model_name_or_path="madebyollin/taesd",
                use_safetensors=True,
                torch_dtype=dtype,
            ).to(self.gpu)

        return torch.compile(
            fullgraph=True,
            mode="reduce-overhead",
            model=AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=model_name,
                use_safetensors=True,
                torch_dtype=dtype,
                subfolder="vae",
            ).to(self.gpu),
        )

    def load(self, model, scheduler, karras, taesd, dtype=None):
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

                # if compiled will be an OptimizedModule
                vae_type = type(self.pipe.vae)
                if (issubclass(vae_type, (AutoencoderKL, OptimizedModule)) and taesd) or (
                    issubclass(vae_type, AutoencoderTiny) and not taesd
                ):
                    print(f"Switching to {'Tiny' if taesd else 'KL'} VAE...")
                    self.pipe.vae = self._load_vae(model_lower, taesd, dtype)
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
        self.pipe.vae = self._load_vae(model_lower, taesd, dtype)
        self.pipe.load_textual_inversion(
            pretrained_model_name_or_path=list(EMBEDDINGS.keys()),
            tokens=list(EMBEDDINGS.values()),
        )
        return self.pipe


@contextmanager
def deep_cache(pipe, interval=1, branch=0, tgate_step=0):
    if interval > 1:
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(cache_interval=interval, cache_branch_id=branch)
        helper.enable()

        if tgate_step > 0:
            pipe.deepcache = helper
            pipe.tgate = MethodType(tgate_sd_deepcache, pipe)

        try:
            yield helper
        finally:
            helper.disable()
    elif interval < 2 and tgate_step > 0:
        pipe.tgate = MethodType(tgate_sd, pipe)
        yield None
    else:
        yield None


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


@spaces.GPU(duration=30)
def generate(
    positive_prompt,
    negative_prompt="",
    seed=None,
    model="Lykon/dreamshaper-8",
    scheduler="DEIS 2M",
    width=512,
    height=512,
    guidance_scale=7.5,
    inference_steps=30,
    num_images=1,
    karras=True,
    taesd=False,
    clip_skip=False,
    truncate_prompts=False,
    increment_seed=True,
    deep_cache_interval=1,
    deep_cache_branch=0,
    tgate_step=0,
    Error=Exception,
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    if seed is None:
        seed = int(datetime.now().timestamp())

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
        loader = Loader()
        pipe = loader.load(model, scheduler, karras, taesd, dtype=TORCH_DTYPE)

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
        neg_embeds = compel(negative_prompt)

        for i in range(num_images):
            # seeded generator for each iteration
            generator = torch.Generator(device=pipe.device).manual_seed(current_seed)

            # get the prompt for this iteration
            all_positive_prompts = parse_prompt(positive_prompt)
            prompt_index = i % len(all_positive_prompts)
            pos_prompt = all_positive_prompts[prompt_index]
            pos_embeds = compel(pos_prompt)
            pos_embeds, neg_embeds = compel.pad_conditioning_tensors_to_same_length(
                [pos_embeds, neg_embeds]
            )

            with deep_cache(
                pipe,
                interval=deep_cache_interval,
                branch=deep_cache_branch,
                tgate_step=tgate_step,
            ):
                pipe_kwargs = {
                    "num_inference_steps": inference_steps,
                    "negative_prompt_embeds": neg_embeds,
                    "guidance_scale": guidance_scale,
                    "prompt_embeds": pos_embeds,
                    "generator": generator,
                    "height": height,
                    "width": width,
                }
                result = (
                    pipe.tgate(**pipe_kwargs, gate_step=tgate_step)
                    if tgate_step > 0
                    else pipe(**pipe_kwargs)
                )
                images.append((result.images[0], str(current_seed)))

            if increment_seed:
                current_seed += 1

        if ZERO_GPU:
            # spaces always start fresh
            loader.pipe = None

        return images
