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
            cls._instance.model_cpu = None
            cls._instance.model_gpu = None
        return cls._instance

    def load(self, model, scheduler, karras):
        SPACES_ZERO_GPU = (
            environ.get("SPACES_ZERO_GPU", "").lower() == "true"
            or environ.get("SPACES_ZERO_GPU", "") == "1"
        )
        model_lower = model.lower()

        scheduler_map = {
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
        }

        if self.model_gpu is not None:
            same_model = self.model_gpu.config._name_or_path.lower() == model_lower
            same_scheduler = isinstance(self.model_gpu.scheduler, scheduler_map[scheduler])
            same_karras = (
                not hasattr(self.model_gpu.scheduler.config, "use_karras_sigmas")
                or self.model_gpu.scheduler.config.use_karras_sigmas == karras
            )
            if same_model and same_scheduler and same_karras:
                return self.model_gpu

        if karras:
            scheduler_kwargs["use_karras_sigmas"] = True

        if scheduler == "PNDM":
            del scheduler_kwargs["use_karras_sigmas"]

        variant = (
            None
            if model_lower in ["sg161222/realistic_vision_v5.1_novae", "prompthero/openjourney-v4"]
            else "fp16"
        )

        pipeline_kwargs = {
            "pretrained_model_name_or_path": model_lower,
            "requires_safety_checker": False,
            "safety_checker": None,
            "scheduler": scheduler_map[scheduler](**scheduler_kwargs),
            "torch_dtype": torch.float16,
            "variant": variant,
            "use_safetensors": True,
            "vae": AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=torch.float16,
                use_safetensors=True,
            ),
        }

        scheduler_cls = scheduler_map[scheduler]
        pipeline_kwargs["scheduler"] = scheduler_cls(**scheduler_kwargs)

        # in ZeroGPU we always start fresh
        if SPACES_ZERO_GPU:
            self.model_gpu = None
            self.model_cpu = None

        if self.model_gpu is not None:
            model_gpu_name = self.model_gpu.config._name_or_path
            self.model_cpu = self.model_gpu.to(self.cpu, silence_dtype_warnings=True)
            self.model_gpu = None
            torch.cuda.empty_cache()
            print(f"Moved {model_gpu_name} to CPU ✓")

        self.model_gpu = StableDiffusionPipeline.from_pretrained(**pipeline_kwargs).to(self.gpu)
        print(f"Moved {model_lower} to GPU ✓")
        return self.model_gpu


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
    guidance_scale=7,
    inference_steps=30,
    karras=True,
    num_images=1,
    increment_seed=True,
):
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
            device=pipe.device.type,
            dtype_for_device_getter=lambda _: torch.float16,
        )

        neg_prompt = join_prompt(negative_prompt)
        neg_embeds = compel(neg_prompt)

        if seed is None:
            seed = int(datetime.now().timestamp())

        current_seed = seed
        images = []

        for i in range(num_images):
            generator = torch.Generator(device=pipe.device.type).manual_seed(current_seed)
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

        return images
