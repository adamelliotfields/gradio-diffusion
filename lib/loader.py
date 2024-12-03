import torch
from DeepCache import DeepCacheSDHelper
from diffusers import ControlNetModel
from diffusers.models.attention_processor import AttnProcessor2_0, IPAdapterAttnProcessor2_0

from .config import Config
from .logger import Logger
from .upscaler import RealESRGAN
from .utils import timer


class Loader:
    """
    A lazy-loading resource manager for Stable Diffusion pipelines. Lifecycles are managed by
    comparing the current state with desired. Can be used as a singleton when created by the
    `get_loader()` helper.

    Usage:
        loader = get_loader(singleton=True)
        loader.load(
            pipeline_id="controlnet_txt2img",
            ip_adapter_model="full-face",
            model="XpucT/Reliberate",
            scheduler="UniPC",
            controlnet_annotator="canny",
            deepcache_interval=2,
            scale=2,
            use_karras=True
        )
    """

    def __init__(self):
        self.model = ""
        self.pipeline = None
        self.upscaler = None
        self.controlnet = None
        self.annotator = ""  # controlnet annotator (canny)
        self.ip_adapter = ""  # ip-adapter kind (full-face or plus)
        self.log = Logger("Loader")

    def should_unload_upscaler(self, scale=1):
        return self.upscaler is not None and self.upscaler.scale != scale

    def should_unload_deepcache(self, cache_interval=1):
        has_deepcache = hasattr(self.pipeline, "deepcache")
        if has_deepcache and cache_interval == 1:
            return True
        if has_deepcache and self.pipeline.deepcache.params["cache_interval"] != cache_interval:
            # Unload if interval is different so it can be reloaded
            return True
        return False

    def should_unload_ip_adapter(self, ip_adapter_model=""):
        if not self.ip_adapter:
            return False
        if not ip_adapter_model:
            return True
        if self.ip_adapter != ip_adapter_model:
            # Unload if model is different so it can be reloaded
            return True
        return False

    def should_unload_controlnet(self, pipeline_id="", annotator=""):
        if self.controlnet is None:
            return False
        if self.annotator != annotator:
            return True
        if not pipeline_id.startswith("controlnet_"):
            return True
        return False

    def should_unload_pipeline(self, model=""):
        if self.pipeline is None:
            return False
        if self.model != model:
            return True
        return False

    # Copied from https://github.com/huggingface/diffusers/blob/v0.28.0/src/diffusers/loaders/ip_adapter.py#L300
    def unload_ip_adapter(self):
        # Remove the image encoder if text-to-image
        if isinstance(self.pipeline, Config.PIPELINES["txt2img"]):
            self.pipeline.image_encoder = None
            self.pipeline.register_to_config(image_encoder=[None, None])

        # Remove hidden projection layer added by IP-Adapter
        self.pipeline.unet.encoder_hid_proj = None
        self.pipeline.unet.config.encoder_hid_dim_type = None

        # Remove the feature extractor
        self.pipeline.feature_extractor = None
        self.pipeline.register_to_config(feature_extractor=[None, None])

        # Replace the custom attention processors with defaults
        attn_procs = {}
        for name, value in self.pipeline.unet.attn_processors.items():
            attn_processor_class = AttnProcessor2_0()  # raises if not torch 2
            attn_procs[name] = (
                attn_processor_class
                if isinstance(value, IPAdapterAttnProcessor2_0)
                else value.__class__()
            )
        self.pipeline.unet.set_attn_processor(attn_procs)
        self.ip_adapter = ""

    def unload_all(
        self,
        pipeline_id="",
        ip_adapter_model="",
        model="",
        controlnet_annotator="",
        deepcache_interval=1,
        scale=1,
    ):
        if self.should_unload_deepcache(deepcache_interval):  # remove deepcache first
            self.log.info("Disabling DeepCache")
            self.pipeline.deepcache.disable()
            delattr(self.pipeline, "deepcache")

        if self.should_unload_ip_adapter(ip_adapter_model):
            self.log.info("Unloading IP-Adapter")
            self.unload_ip_adapter()

        if self.should_unload_controlnet(pipeline_id, controlnet_annotator):
            self.log.info("Unloading ControlNet")
            self.controlnet = None
            self.annotator = ""

        if self.should_unload_upscaler(scale):
            self.log.info("Unloading upscaler")
            self.upscaler = None

        if self.should_unload_pipeline(model):
            self.log.info("Unloading pipeline")
            self.pipeline = None
            self.model = ""

    def should_load_upscaler(self, scale=1):
        return self.upscaler is None and scale > 1

    def should_load_deepcache(self, cache_interval=1):
        has_deepcache = hasattr(self.pipeline, "deepcache")
        if not has_deepcache and cache_interval > 1:
            return True
        return False

    def should_load_controlnet(self, pipeline_id=""):
        return self.controlnet is None and pipeline_id.startswith("controlnet_")

    def should_load_ip_adapter(self, ip_adapter_model=""):
        has_ip_adapter = (
            hasattr(self.pipeline.unet, "encoder_hid_proj")
            and self.pipeline.unet.config.encoder_hid_dim_type == "ip_image_proj"
        )
        return not has_ip_adapter and ip_adapter_model != ""

    def should_load_scheduler(self, cls, use_karras=False):
        has_karras = hasattr(self.pipeline.scheduler.config, "use_karras_sigmas")
        if not isinstance(self.pipeline.scheduler, cls):
            return True
        if has_karras and self.pipeline.scheduler.config.use_karras_sigmas != use_karras:
            return True
        return False

    def should_load_pipeline(self, pipeline_id=""):
        if self.pipeline is None:
            return True
        if not isinstance(self.pipeline, Config.PIPELINES[pipeline_id]):
            return True
        return False

    def load_upscaler(self, scale=1):
        with timer(f"Loading {scale}x upscaler", logger=self.log.info):
            self.upscaler = RealESRGAN(scale, device=self.pipeline.device)
            self.upscaler.load_weights()

    def load_deepcache(self, cache_interval=1):
        self.log.info(f"Enabling DeepCache interval {cache_interval}")
        self.pipeline.deepcache = DeepCacheSDHelper(self.pipeline)
        self.pipeline.deepcache.set_params(cache_interval=cache_interval)
        self.pipeline.deepcache.enable()

    def load_controlnet(self, controlnet_annotator):
        with timer("Loading ControlNet", logger=self.log.info):
            self.controlnet = ControlNetModel.from_pretrained(
                Config.ANNOTATORS[controlnet_annotator],
                variant="fp16",
                torch_dtype=torch.float16,
            )
            self.annotator = controlnet_annotator

    def load_ip_adapter(self, ip_adapter_model=""):
        with timer("Loading IP-Adapter", logger=self.log.info):
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name=f"ip-adapter-{ip_adapter_model}_sd15.safetensors",
            )
            self.pipeline.set_ip_adapter_scale(0.5)  # 50% works the best
            self.ip_adapter = ip_adapter_model

    def load_scheduler(self, cls, use_karras=False, **kwargs):
        self.log.info(f"Loading {cls.__name__}{' with Karras' if use_karras else ''}")
        self.pipeline.scheduler = cls(**kwargs)

    def load_pipeline(
        self,
        pipeline_id,
        model,
        **kwargs,
    ):
        Pipeline = Config.PIPELINES[pipeline_id]

        # Load from scratch
        if self.pipeline is None:
            with timer(f"Loading {model} ({pipeline_id})", logger=self.log.info):
                if self.controlnet is not None:
                    kwargs["controlnet"] = self.controlnet
                if model in Config.SINGLE_FILE_MODELS:
                    checkpoint = Config.HF_REPOS[model][0]
                    self.pipeline = Pipeline.from_single_file(
                        f"https://huggingface.co/{model}/{checkpoint}",
                        **kwargs,
                    ).to("cuda")
                else:
                    self.pipeline = Pipeline.from_pretrained(model, **kwargs).to("cuda")

        # Change to a different one
        else:
            with timer(f"Changing pipeline to {pipeline_id}", logger=self.log.info):
                kwargs = {}
                if self.controlnet is not None:
                    kwargs["controlnet"] = self.controlnet
                self.pipeline = Pipeline.from_pipe(
                    self.pipeline,
                    **kwargs,
                ).to("cuda")

        # Update model and disable terminal progress bars
        self.model = model
        self.pipeline.set_progress_bar_config(disable=True)

    def load(
        self,
        pipeline_id,
        ip_adapter_model,
        model,
        scheduler,
        controlnet_annotator,
        deepcache_interval,
        scale,
        use_karras,
    ):
        Scheduler = Config.SCHEDULERS[scheduler]

        scheduler_kwargs = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "steps_offset": 1,
        }

        if scheduler not in ["Euler a"]:
            scheduler_kwargs["use_karras_sigmas"] = use_karras

        pipeline_kwargs = {
            "torch_dtype": torch.float16,  # defaults to fp32
            "safety_checker": None,
            "requires_safety_checker": False,
            "scheduler": Scheduler(**scheduler_kwargs),
        }

        # Single-file models don't need a variant
        if model not in Config.SINGLE_FILE_MODELS:
            pipeline_kwargs["variant"] = "fp16"
        else:
            pipeline_kwargs["variant"] = None

        # Prepare state for loading checks
        self.unload_all(
            pipeline_id,
            ip_adapter_model,
            model,
            controlnet_annotator,
            deepcache_interval,
            scale,
        )

        # Load controlnet model before pipeline
        if self.should_load_controlnet(pipeline_id):
            self.load_controlnet(controlnet_annotator)

        if self.should_load_pipeline(pipeline_id):
            self.load_pipeline(pipeline_id, model, **pipeline_kwargs)

        if self.should_load_scheduler(Scheduler, use_karras):
            self.load_scheduler(Scheduler, use_karras, **scheduler_kwargs)

        if self.should_load_deepcache(deepcache_interval):
            self.load_deepcache(deepcache_interval)

        if self.should_load_ip_adapter(ip_adapter_model):
            self.load_ip_adapter(ip_adapter_model)

        if self.should_load_upscaler(scale):
            self.load_upscaler(scale)


# Get a singleton or a new instance of the Loader
def get_loader(singleton=False):
    if not singleton:
        return Loader()
    else:
        if not hasattr(get_loader, "_instance"):
            get_loader._instance = Loader()
        assert isinstance(get_loader._instance, Loader)
        return get_loader._instance
