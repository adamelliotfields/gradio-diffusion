import os
from importlib import import_module

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers.loaders.single_file import (
    SINGLE_FILE_OPTIONAL_COMPONENTS,
    load_single_file_sub_model,
)
from diffusers.loaders.single_file_utils import fetch_diffusers_config, load_single_file_checkpoint
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.pipelines.pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    _get_pipeline_class,
    load_sub_model,
)
from diffusers.utils import logging
from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_hf_hub_args


class CustomDiffusionMixin:
    r"""
    Overrides DiffusionPipeline methods.
    """

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/pipelines/pipeline_utils.py#L480
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path, progress=None, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)
        variant = kwargs.pop("variant", None)
        token = kwargs.pop("token", None)

        # download the checkpoints and configs
        cached_folder = cls.download(
            pretrained_model_name_or_path,
            variant=variant,
            token=token,
            **kwargs,
        )

        # pop out "_ignore_files" as it is only needed for download
        config_dict = cls.load_config(cached_folder)
        config_dict.pop("_ignore_files", None)

        # Define which model components should load variants.
        # We retrieve the information by matching whether variant model checkpoints exist in the subfolders.
        # Example: `diffusion_pytorch_model.safetensors` -> `diffusion_pytorch_model.fp16.safetensors` with variant being `"fp16"`.
        model_variants = {}
        if variant is not None:
            for folder in os.listdir(cached_folder):
                folder_path = os.path.join(cached_folder, folder)
                is_folder = os.path.isdir(folder_path) and folder in config_dict
                variant_exists = is_folder and any(
                    p.split(".")[1].startswith(variant) for p in os.listdir(folder_path)
                )
                if variant_exists:
                    model_variants[folder] = variant

        # load the pipeline class
        pipeline_class = _get_pipeline_class(cls, config=config_dict)

        # define expected modules given pipeline signature and define non-None initialized modules (=`init_kwargs`)
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict, _, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}
        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # load each module in the pipeline
        pipelines = import_module("diffusers.pipelines")
        tqdm = logging.tqdm if progress is None else progress.tqdm
        for name, (library_name, class_name) in tqdm(
            sorted(init_dict.items()),
            desc="Loading pipeline components",
        ):
            # use passed sub model or load class_name from library_name
            loaded_sub_model = None
            if name in passed_class_obj:
                # passed as an argument like "scheduler"
                loaded_sub_model = passed_class_obj[name]
            else:
                loaded_sub_model = load_sub_model(
                    library_name=library_name,
                    class_name=class_name,
                    importable_classes=ALL_IMPORTABLE_CLASSES,
                    pipelines=pipelines,
                    is_pipeline_module=hasattr(pipelines, library_name),
                    pipeline_class=pipeline_class,
                    torch_dtype=torch_dtype,
                    provider=None,
                    sess_options=None,
                    device_map=None,
                    max_memory=None,
                    offload_folder=None,
                    offload_state_dict=False,
                    model_variants=model_variants,
                    name=name,
                    from_flax=False,
                    variant=variant,
                    low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT,
                    cached_folder=cached_folder,
                )
            init_kwargs[name] = loaded_sub_model

        # potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        if len(missing_modules) > 0:
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)

        # instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        return model

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/loaders/single_file.py#L270
    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, progress=None, **kwargs):
        token = kwargs.pop("token", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        # load the pipeline class
        pipeline_class = _get_pipeline_class(cls, config=None)
        checkpoint = load_single_file_checkpoint(pretrained_model_link_or_path, token=token)

        config = fetch_diffusers_config(checkpoint)
        default_pretrained_model_config_name = config["pretrained_model_name_or_path"]

        # attempt to download the config files for the pipeline
        cached_model_config_path = snapshot_download(
            default_pretrained_model_config_name,
            token=token,
            allow_patterns=["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"],
        )

        # pop out "_ignore_files" as it is only needed for download
        config_dict = pipeline_class.load_config(cached_model_config_path)
        config_dict.pop("_ignore_files", None)

        # define expected modules given pipeline signature and define non-None initialized modules (=`init_kwargs`)
        expected_modules, optional_kwargs = pipeline_class._get_signature_keys(cls)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            if name in SINGLE_FILE_OPTIONAL_COMPONENTS:
                return False
            return True

        init_dict, _, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)
        init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}
        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # load each module in the pipeline
        pipelines = import_module("diffusers.pipelines")
        tqdm = logging.tqdm if progress is None else progress.tqdm
        for name, (library_name, class_name) in tqdm(
            sorted(init_dict.items()),
            desc="Loading pipeline components",
        ):
            # use passed sub model or load class_name from library_name
            loaded_sub_model = None
            if name in passed_class_obj:
                # passed as an argument like "scheduler"
                loaded_sub_model = passed_class_obj[name]
            else:
                loaded_sub_model = load_single_file_sub_model(
                    library_name=library_name,
                    class_name=class_name,
                    name=name,
                    checkpoint=checkpoint,
                    is_pipeline_module=hasattr(pipelines, library_name),
                    cached_model_config_path=cached_model_config_path,
                    pipelines=pipelines,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )
            init_kwargs[name] = loaded_sub_model

        # potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        if len(missing_modules) > 0:
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)

        # instantiate the pipeline
        pipe = pipeline_class(**init_kwargs)

        # save where the model was instantiated from
        pipe.register_to_config(_name_or_path=pretrained_model_link_or_path)
        return pipe


class CustomStableDiffusionPipeline(CustomDiffusionMixin, StableDiffusionPipeline):
    pass


class CustomStableDiffusionImg2ImgPipeline(CustomDiffusionMixin, StableDiffusionImg2ImgPipeline):
    pass
