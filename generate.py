import re
from datetime import datetime
from itertools import product
from os import environ
from warnings import filterwarnings

import spaces
import torch
from compel import Compel
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
from diffusers.models import AutoencoderTiny

ZERO_GPU = (
    environ.get("SPACES_ZERO_GPU", "").lower() == "true"
    or environ.get("SPACES_ZERO_GPU", "") == "1"
)

TORCH_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

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

    def load(self, model, scheduler, karras):
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
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "steps_offset": 1,
            "use_karras_sigmas": karras,
        }

        if scheduler == "PNDM" or scheduler == "Euler a":
            del scheduler_kwargs["use_karras_sigmas"]

        pipe_kwargs = {
            "pretrained_model_name_or_path": model_lower,
            "requires_safety_checker": False,
            "safety_checker": None,
            "scheduler": schedulers[scheduler](**scheduler_kwargs),
            "torch_dtype": TORCH_DTYPE,
            "use_safetensors": True,
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
                    print(f"Swapping scheduler to {scheduler}...")
                elif not same_karras:
                    print(f"{'Enabling' if karras else 'Disabling'} Karras sigmas...")
                elif not (same_scheduler and same_karras):
                    self.pipe.scheduler = schedulers[scheduler](**scheduler_kwargs)
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

        # uses special VAE
        if model_lower not in ["linaqruf/anything-v3-1"]:
            pipe_kwargs["vae"] = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=TORCH_DTYPE,
                use_safetensors=True,
            )

        print(f"Loading {model_lower}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(**pipe_kwargs).to(self.gpu)
        return self.pipe


# prepare prompts for Compel
def join_prompt(prompt: str) -> str:
    lines = prompt.strip().splitlines()
    return '("' + '", "'.join(lines) + '").and()' if len(lines) > 1 else prompt


# parse prompts with arrays
def parse_prompt(prompt: str) -> list[str]:
    joined_prompt = join_prompt(prompt)
    arrays = re.findall(r"\[\[(.*?)\]\]", joined_prompt)

    if not arrays:
        return [joined_prompt]

    tokens = [item.split(",") for item in arrays]
    combinations = list(product(*tokens))
    prompts = []

    for combo in combinations:
        current_prompt = joined_prompt
        for i, token in enumerate(combo):
            current_prompt = current_prompt.replace(f"[[{arrays[i]}]]", token.strip(), 1)

        prompts.append(current_prompt)
    return prompts


@spaces.GPU(duration=30)
def generate(
    positive_prompt,
    negative_prompt="",
    seed=None,
    model="lykon/dreamshaper-8",
    scheduler="DEIS 2M",
    aspect_ratio="1:1",
    guidance_scale=7.5,
    inference_steps=30,
    karras=True,
    num_images=1,
    increment_seed=True,
    Error=Exception,
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # image dimensions
    aspect_ratios = {
        "16:9": (640, 360),
        "4:3": (576, 432),
        "1:1": (512, 512),
        "3:4": (432, 576),
        "9:16": (360, 640),
    }
    width, height = aspect_ratios[aspect_ratio]

    with torch.inference_mode():
        loader = Loader()
        pipe = loader.load(model, scheduler, karras)

        # prompt embeds
        compel = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            truncate_long_prompts=False,
            device=pipe.device,
            dtype_for_device_getter=lambda _: TORCH_DTYPE,
        )

        neg_prompt = join_prompt(negative_prompt)
        neg_embeds = compel(neg_prompt)

        if seed is None:
            seed = int(datetime.now().timestamp())

        current_seed = seed
        images = []

        for i in range(num_images):
            generator = torch.Generator(device=pipe.device).manual_seed(current_seed)

            # run the prompt for this iteration
            all_positive_prompts = parse_prompt(positive_prompt)
            prompt_index = i % len(all_positive_prompts)
            pos_prompt = all_positive_prompts[prompt_index]
            pos_embeds = compel(pos_prompt)
            pos_embeds, neg_embeds = compel.pad_conditioning_tensors_to_same_length(
                [pos_embeds, neg_embeds]
            )

            result = pipe(
                width=width,
                height=height,
                prompt_embeds=pos_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            images.append((result.images[0], str(current_seed)))

            if increment_seed:
                current_seed += 1

        if ZERO_GPU:
            # spaces always start fresh
            loader.pipe = None

        return images
