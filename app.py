import argparse
import json
import os
import random

import gradio as gr

from lib import (
    Config,
    async_call,
    disable_progress_bars,
    download_civit_file,
    download_repo_files,
    generate,
    read_file,
)

# the CSS `content` attribute expects a string so we need to wrap the number in quotes
refresh_seed_js = """
() => {
    const n = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${n}"`);
    return n;
}
"""

seed_js = """
(seed) => {
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${seed}"`);
    return seed;
}
"""

aspect_ratio_js = """
(ar, w, h) => {
    if (!ar) return [w, h];
    const [width, height] = ar.split(",");
    return [parseInt(width), parseInt(height)];
}
"""


def image_prompt_fn(images, locked=False):
    if locked:
        return gr.Dropdown(
            choices=[("🔒", -2)],
            interactive=False,
            value=-2,
        )
    else:
        return gr.Dropdown(
            choices=[("None", -1)] + [(str(i + 1), i) for i, _ in enumerate(images or [])],
            interactive=True,
            value=-1,
        )


async def gallery_fn(images, image, control_image, ip_image):
    return (
        image_prompt_fn(images, locked=image is not None),
        image_prompt_fn(images, locked=control_image is not None),
        image_prompt_fn(images, locked=ip_image is not None),
    )


# Handle selecting an image from the gallery:
# * -2 is the lock icon
# * -1 is None
async def image_select_fn(images, image, i):
    if i == -2:
        return gr.Image(image)
    if i == -1:
        return gr.Image(None)
    return gr.Image(images[i][0]) if i > -1 else None


async def random_fn():
    prompts = read_file("data/prompts.json")
    prompts = json.loads(prompts)
    return gr.Textbox(value=random.choice(prompts))


async def generate_fn(*args, progress=gr.Progress(track_tqdm=True)):
    if len(args) > 0:
        prompt = args[0]
    else:
        prompt = None
    if prompt is None or prompt.strip() == "":
        raise gr.Error("You must enter a prompt")

    # always the last arguments
    DISABLE_IMAGE_PROMPT, DISABLE_CONTROL_IMAGE_PROMPT, DISABLE_IP_IMAGE_PROMPT = args[-3:]
    gen_args = list(args[:-3])

    # the first two arguments are the prompt and negative prompt
    if DISABLE_IMAGE_PROMPT:
        gen_args[2] = None
    if DISABLE_CONTROL_IMAGE_PROMPT:
        gen_args[3] = None
    if DISABLE_IP_IMAGE_PROMPT:
        gen_args[4] = None

    try:
        if Config.ZERO_GPU:
            progress((0, 100), desc="ZeroGPU init")

        # the remaining arguments are the alert handlers and progress bar
        images = await async_call(
            generate,
            *gen_args,
            Error=gr.Error,
            Info=gr.Info,
            progress=progress,
        )
    except RuntimeError:
        raise gr.Error("Error: Please try again")

    return images


with gr.Blocks(
    head=read_file("./partials/head.html"),
    css="./app.css",
    js="./app.js",
    theme=gr.themes.Default(
        # colors
        neutral_hue=gr.themes.colors.gray,
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        # sizing
        text_size=gr.themes.sizes.text_md,
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_md,
        # fonts
        font=[gr.themes.GoogleFont("Inter"), *Config.SANS_FONTS],
        font_mono=[gr.themes.GoogleFont("Ubuntu Mono"), *Config.MONO_FONTS],
    ).set(
        layout_gap="8px",
        block_shadow="0 0 #0000",
        block_shadow_dark="0 0 #0000",
        block_background_fill=gr.themes.colors.gray.c50,
        block_background_fill_dark=gr.themes.colors.gray.c900,
    ),
) as demo:
    # override image inputs without clearing them
    DISABLE_IMAGE_PROMPT = gr.State(False)
    DISABLE_IP_IMAGE_PROMPT = gr.State(False)
    DISABLE_CONTROL_IMAGE_PROMPT = gr.State(False)

    gr.HTML(read_file("./partials/intro.html"))

    with gr.Tabs():
        with gr.TabItem("🏠 Text"):
            with gr.Column():
                output_images = gr.Gallery(
                    elem_classes=["gallery"],
                    show_share_button=False,
                    object_fit="cover",
                    interactive=False,
                    show_label=False,
                    label="Output",
                    format="png",
                    columns=2,
                )
                prompt = gr.Textbox(
                    placeholder="What do you want to see?",
                    autoscroll=False,
                    show_label=False,
                    label="Prompt",
                    max_lines=3,
                    lines=3,
                )

                # Buttons
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    random_btn = gr.Button(
                        elem_classes=["icon-button", "popover"],
                        variant="secondary",
                        elem_id="random",
                        min_width=0,
                        value="🎲",
                    )
                    refresh_btn = gr.Button(
                        elem_classes=["icon-button", "popover"],
                        variant="secondary",
                        elem_id="refresh",
                        min_width=0,
                        value="🔄",
                    )
                    clear_btn = gr.ClearButton(
                        elem_classes=["icon-button", "popover"],
                        components=[output_images],
                        variant="secondary",
                        elem_id="clear",
                        min_width=0,
                        value="🗑️",
                    )

        # img2img tab
        with gr.TabItem("🖼️ Image"):
            with gr.Row():
                image_prompt = gr.Image(
                    show_share_button=False,
                    label="Initial Image",
                    min_width=640,
                    format="png",
                    type="pil",
                )
            with gr.Row():
                control_image_prompt = gr.Image(
                    show_share_button=False,
                    label="Control Image",
                    min_width=320,
                    format="png",
                    type="pil",
                )
                ip_image_prompt = gr.Image(
                    show_share_button=False,
                    label="IP-Adapter Image",
                    min_width=320,
                    format="png",
                    type="pil",
                )

            with gr.Row():
                image_select = gr.Dropdown(
                    info="Use a gallery image for initial latents",
                    choices=[("None", -1)],
                    label="Initial Image",
                    interactive=True,
                    filterable=False,
                    min_width=100,
                    value=-1,
                )
                control_image_select = gr.Dropdown(
                    info="Use a gallery image for ControlNet",
                    label="ControlNet Image",
                    choices=[("None", -1)],
                    interactive=True,
                    filterable=False,
                    min_width=100,
                    value=-1,
                )
                ip_image_select = gr.Dropdown(
                    info="Use a gallery image for IP-Adapter",
                    label="IP-Adapter Image",
                    choices=[("None", -1)],
                    interactive=True,
                    filterable=False,
                    min_width=100,
                    value=-1,
                )

            with gr.Row():
                denoising_strength = gr.Slider(
                    label="Initial Image Strength",
                    value=Config.DENOISING_STRENGTH,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                )
                control_annotator = gr.Dropdown(
                    label="ControlNet Annotator",
                    # TODO: annotators should be in config with names
                    choices=[("Canny", "canny")],
                    value=Config.ANNOTATOR,
                    filterable=False,
                )

            with gr.Row():
                disable_image = gr.Checkbox(
                    label="Disable Initial Image",
                    elem_classes=["checkbox"],
                    value=False,
                )
                disable_control_image = gr.Checkbox(
                    label="Disable ControlNet Image",
                    elem_classes=["checkbox"],
                    value=False,
                )
                disable_ip_image = gr.Checkbox(
                    label="Disable IP-Adapter Image",
                    elem_classes=["checkbox"],
                    value=False,
                )
                use_ip_face = gr.Checkbox(
                    label="Use IP-Adapter Face",
                    elem_classes=["checkbox"],
                    value=False,
                )

        with gr.TabItem("⚙️ Menu"):
            with gr.Group():
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="nsfw+",
                    lines=2,
                )

                with gr.Row():
                    model = gr.Dropdown(
                        choices=Config.MODELS,
                        value=Config.MODEL,
                        filterable=False,
                        label="Model",
                        min_width=240,
                    )
                    scheduler = gr.Dropdown(
                        choices=Config.SCHEDULERS.keys(),
                        value=Config.SCHEDULER,
                        elem_id="scheduler",
                        label="Scheduler",
                        filterable=False,
                    )

                with gr.Row():
                    styles = json.loads(read_file("data/styles.json"))
                    style_ids = list(styles.keys())
                    style_ids = [sid for sid in style_ids if not sid.startswith("_")]
                    style = gr.Dropdown(
                        value=Config.STYLE,
                        label="Style",
                        min_width=240,
                        choices=[("None", "none")]
                        + [(styles[sid]["name"], sid) for sid in style_ids],
                    )
                    embeddings = gr.Dropdown(
                        elem_id="embeddings",
                        label="Embeddings",
                        choices=[(f"<{e}>", e) for e in Config.EMBEDDINGS],
                        multiselect=True,
                        value=[Config.EMBEDDING],
                        min_width=240,
                    )

                with gr.Row():
                    with gr.Group(elem_classes=["gap-0"]):
                        lora_1 = gr.Dropdown(
                            min_width=240,
                            label="LoRA #1",
                            value="none",
                            choices=[("None", "none")]
                            + [
                                (lora["name"], lora_id)
                                for lora_id, lora in Config.CIVIT_LORAS.items()
                            ],
                        )
                        lora_1_weight = gr.Slider(
                            value=0.0,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            show_label=False,
                        )
                    with gr.Group(elem_classes=["gap-0"]):
                        lora_2 = gr.Dropdown(
                            min_width=240,
                            label="LoRA #2",
                            value="none",
                            choices=[("None", "none")]
                            + [
                                (lora["name"], lora_id)
                                for lora_id, lora in Config.CIVIT_LORAS.items()
                            ],
                        )
                        lora_2_weight = gr.Slider(
                            value=0.0,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            show_label=False,
                        )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        value=Config.GUIDANCE_SCALE,
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=15.0,
                        step=0.1,
                    )
                    inference_steps = gr.Slider(
                        value=Config.INFERENCE_STEPS,
                        label="Inference Steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                    )
                    deepcache_interval = gr.Slider(
                        value=Config.DEEPCACHE_INTERVAL,
                        label="DeepCache",
                        minimum=1,
                        maximum=4,
                        step=1,
                    )

                with gr.Row():
                    width = gr.Slider(
                        value=Config.WIDTH,
                        label="Width",
                        minimum=256,
                        maximum=768,
                        step=32,
                    )
                    height = gr.Slider(
                        value=Config.HEIGHT,
                        label="Height",
                        minimum=256,
                        maximum=768,
                        step=32,
                    )
                    aspect_ratio = gr.Dropdown(
                        value=f"{Config.WIDTH},{Config.HEIGHT}",
                        label="Aspect Ratio",
                        filterable=False,
                        choices=[
                            ("Custom", None),
                            ("4:7 (384x672)", "384,672"),
                            ("7:9 (448x576)", "448,576"),
                            ("1:1 (512x512)", "512,512"),
                            ("9:7 (576x448)", "576,448"),
                            ("7:4 (672x384)", "672,384"),
                        ],
                    )

                with gr.Row():
                    file_format = gr.Dropdown(
                        choices=["png", "jpeg", "webp"],
                        label="File Format",
                        filterable=False,
                        value="png",
                    )
                    num_images = gr.Dropdown(
                        choices=list(range(1, 5)),
                        value=Config.NUM_IMAGES,
                        filterable=False,
                        label="Images",
                    )
                    scale = gr.Dropdown(
                        choices=[(f"{s}x", s) for s in Config.SCALES],
                        filterable=False,
                        value=Config.SCALE,
                        label="Scale",
                    )
                    seed = gr.Number(
                        value=Config.SEED,
                        label="Seed",
                        minimum=-1,
                        maximum=(2**64) - 1,
                    )

                with gr.Row():
                    use_karras = gr.Checkbox(
                        elem_classes=["checkbox"],
                        label="Karras σ",
                        value=True,
                    )
                    use_taesd = gr.Checkbox(
                        elem_classes=["checkbox"],
                        label="Tiny VAE",
                        value=False,
                    )
                    use_freeu = gr.Checkbox(
                        elem_classes=["checkbox"],
                        label="FreeU",
                        value=False,
                    )
                    use_clip_skip = gr.Checkbox(
                        elem_classes=["checkbox"],
                        label="Clip skip",
                        value=False,
                    )

    random_btn.click(random_fn, inputs=[], outputs=[prompt], show_api=False)

    refresh_btn.click(None, inputs=[], outputs=[seed], js=refresh_seed_js)

    seed.change(None, inputs=[seed], outputs=[], js=seed_js)

    # input events are only user input; change events are both user and programmatic
    aspect_ratio.input(
        None,
        inputs=[aspect_ratio, width, height],
        outputs=[width, height],
        js=aspect_ratio_js,
    )

    file_format.change(
        lambda f: (
            gr.Gallery(format=f),
            gr.Image(format=f),
            gr.Image(format=f),
            gr.Image(format=f),
        ),
        inputs=[file_format],
        outputs=[output_images, image_prompt, control_image_prompt, ip_image_prompt],
        show_api=False,
    )

    # lock the input images so you don't lose them when the gallery updates
    output_images.change(
        gallery_fn,
        inputs=[output_images, image_prompt, control_image_prompt, ip_image_prompt],
        outputs=[image_select, control_image_select, ip_image_select],
        show_api=False,
    )

    # show the selected image in the image input
    image_select.change(
        image_select_fn,
        inputs=[output_images, image_prompt, image_select],
        outputs=[image_prompt],
        show_api=False,
    )
    control_image_select.change(
        image_select_fn,
        inputs=[output_images, control_image_prompt, control_image_select],
        outputs=[control_image_prompt],
        show_api=False,
    )
    ip_image_select.change(
        image_select_fn,
        inputs=[output_images, ip_image_prompt, ip_image_select],
        outputs=[ip_image_prompt],
        show_api=False,
    )

    # reset the dropdown on clear
    image_prompt.clear(
        image_prompt_fn,
        inputs=[output_images],
        outputs=[image_select],
        show_api=False,
    )
    control_image_prompt.clear(
        image_prompt_fn,
        inputs=[output_images],
        outputs=[control_image_select],
        show_api=False,
    )
    ip_image_prompt.clear(
        image_prompt_fn,
        inputs=[output_images],
        outputs=[ip_image_select],
        show_api=False,
    )

    # show "Custom" aspect ratio when manually changing width or height
    gr.on(
        triggers=[width.input, height.input],
        fn=None,
        inputs=[],
        outputs=[aspect_ratio],
        js="() => { return null; }",
    )

    # toggle image prompts by updating session state
    gr.on(
        triggers=[disable_image.input, disable_control_image.input, disable_ip_image.input],
        fn=lambda disable_image, disable_control_image, disable_ip_image: (
            disable_image,
            disable_control_image,
            disable_ip_image,
        ),
        inputs=[disable_image, disable_control_image, disable_ip_image],
        outputs=[DISABLE_IMAGE_PROMPT, DISABLE_CONTROL_IMAGE_PROMPT, DISABLE_IP_IMAGE_PROMPT],
    )

    # generate images
    gr.on(
        triggers=[generate_btn.click, prompt.submit],
        fn=generate_fn,
        api_name="generate",
        outputs=[output_images],
        inputs=[
            prompt,
            negative_prompt,
            image_prompt,
            control_image_prompt,
            ip_image_prompt,
            lora_1,
            lora_1_weight,
            lora_2,
            lora_2_weight,
            embeddings,
            style,
            seed,
            model,
            scheduler,
            control_annotator,
            width,
            height,
            guidance_scale,
            inference_steps,
            denoising_strength,
            deepcache_interval,
            scale,
            num_images,
            use_karras,
            use_taesd,
            use_freeu,
            use_clip_skip,
            use_ip_face,
            DISABLE_IMAGE_PROMPT,
            DISABLE_CONTROL_IMAGE_PROMPT,
            DISABLE_IP_IMAGE_PROMPT,
        ],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-s", "--server", type=str, metavar="STR", default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, metavar="INT", default=7860)
    args = parser.parse_args()

    disable_progress_bars()
    for repo_id, allow_patterns in Config.HF_MODELS.items():
        download_repo_files(repo_id, allow_patterns, token=Config.HF_TOKEN)

    # download civit loras
    for lora_id, lora in Config.CIVIT_LORAS.items():
        file_path = os.path.join(os.path.dirname(__file__), "loras")
        download_civit_file(
            lora_id,
            lora["model_version_id"],
            file_path=file_path,
            token=Config.CIVIT_TOKEN,
        )

    # https://www.gradio.app/docs/gradio/interface#interface-queue
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server,
        server_port=args.port,
    )
