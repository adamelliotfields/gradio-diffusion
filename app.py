import argparse
import json
import random

import gradio as gr

from lib import Config, generate

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


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def random_fn():
    prompts = read_file("data/prompts.json")
    prompts = json.loads(prompts)
    return gr.Textbox(value=random.choice(prompts))


# can't toggle interactive in JS
def gallery_fn(images, image):
    if image is not None:
        return gr.Dropdown(
            choices=[("🔒", -1)],
            interactive=False,
            value=-1,
        )

    return gr.Dropdown(
        choices=[("None", -1)]
        + [(str(i + 1), i) for i, _ in enumerate(images if images is not None else [])],
        interactive=True,
        value=-1,
    )


def image_prompt_fn(images):
    return gallery_fn(images, None)


# can't use image input in JS
def image_select_fn(images, image, i):
    if image is not None and i == -1:
        return gr.Image(value=image)
    return gr.Image(value=images[i][0]) if i > -1 else None


def generate_fn(*args):
    if len(args) > 0:
        prompt = args[0]
    else:
        prompt = None
    if prompt is None or prompt.strip() == "":
        raise gr.Error("You must enter a prompt")
    try:
        images = generate(*args, Info=gr.Info, Error=gr.Error)
    except RuntimeError:
        raise gr.Error("RuntimeError: Please try again")
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
    gr.HTML(read_file("./partials/intro.html"))

    with gr.Accordion(
        elem_classes=["accordion"],
        elem_id="menu",
        label="Show menu",
        open=False,
    ):
        with gr.Tabs():
            with gr.TabItem("⚙️ Settings"):
                with gr.Group():
                    negative_prompt = gr.Textbox(
                        value=None,
                        label="Negative Prompt",
                        placeholder="ugly, bad",
                        lines=2,
                    )

                    with gr.Row():
                        model = gr.Dropdown(
                            choices=Config.MODELS,
                            filterable=False,
                            value=Config.MODEL,
                            label="Model",
                            min_width=240,
                        )
                        scheduler = gr.Dropdown(
                            choices=Config.SCHEDULERS,
                            value=Config.SCHEDULER,
                            elem_id="scheduler",
                            label="Scheduler",
                            filterable=False,
                        )

                    with gr.Row():
                        styles = json.loads(read_file("data/styles.json"))
                        style = gr.Dropdown(
                            value=Config.STYLE,
                            label="Style",
                            choices=[("None", None)] + [(s["name"], s["id"]) for s in styles],
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
                        seed = gr.Number(
                            value=Config.SEED,
                            label="Seed",
                            minimum=-1,
                            maximum=(2**64) - 1,
                        )

                    with gr.Row():
                        width = gr.Slider(
                            value=Config.WIDTH,
                            label="Width",
                            minimum=320,
                            maximum=768,
                            step=16,
                        )
                        height = gr.Slider(
                            value=Config.HEIGHT,
                            label="Height",
                            minimum=320,
                            maximum=768,
                            step=16,
                        )
                        aspect_ratio = gr.Dropdown(
                            choices=[
                                ("Custom", None),
                                ("7:9 (448x576)", "448,576"),
                                ("3:4 (432x576)", "432,576"),
                                ("1:1 (512x512)", "512,512"),
                                ("4:3 (576x432)", "576,432"),
                                ("9:7 (576x448)", "576,448"),
                            ],
                            value="448,576",
                            filterable=False,
                            label="Aspect Ratio",
                        )
                        scale = gr.Dropdown(
                            choices=[(f"{s}x", s) for s in Config.SCALES],
                            filterable=False,
                            value=Config.SCALE,
                            label="Scale",
                        )

                    with gr.Row():
                        num_images = gr.Dropdown(
                            choices=list(range(1, 5)),
                            value=Config.NUM_IMAGES,
                            filterable=False,
                            label="Images",
                        )
                        file_format = gr.Dropdown(
                            choices=["png", "jpeg", "webp"],
                            label="File Format",
                            filterable=False,
                            value="png",
                        )
                        deepcache_interval = gr.Slider(
                            value=Config.DEEPCACHE_INTERVAL,
                            label="DeepCache",
                            minimum=1,
                            maximum=4,
                            step=1,
                        )
                        tome_ratio = gr.Slider(
                            value=Config.TOME_RATIO,
                            label="ToMe Ratio",
                            minimum=0.0,
                            maximum=0.5,
                            step=0.01,
                        )

                    with gr.Row():
                        increment_seed = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Autoincrement",
                            value=True,
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
                        truncate_prompts = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Truncate prompts",
                            value=False,
                        )

            # img2img tab
            with gr.TabItem("🖼️ Image"):
                with gr.Row():
                    image_prompt = gr.Image(
                        show_label=False,
                        min_width=320,
                        format="png",
                        type="pil",
                        scale=0,
                    )

                with gr.Row():
                    image_select = gr.Dropdown(
                        choices=[("None", -1)],
                        label="Load from Gallery",
                        interactive=True,
                        filterable=False,
                        value=-1,
                    )
                    denoising_strength = gr.Slider(
                        value=Config.DENOISING_STRENGTH,
                        label="Denoising Strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                    )

            with gr.TabItem("ℹ️ Usage"):
                gr.Markdown(read_file("usage.md"), elem_classes=["markdown"])

    # Main content
    with gr.Column(elem_id="content"):
        with gr.Group():
            output_images = gr.Gallery(
                elem_classes=["gallery"],
                show_share_button=False,
                interactive=False,
                show_label=False,
                object_fit="cover",
                label="Output",
                format="png",
                columns=2,
            )
            prompt = gr.Textbox(
                placeholder="corgi, beach, 8k",
                show_label=False,
                label="Prompt",
                value=None,
                lines=2,
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

    random_btn.click(random_fn, inputs=[], outputs=[prompt], show_api=False)

    refresh_btn.click(None, inputs=[], outputs=[seed], js=refresh_seed_js)

    seed.change(None, inputs=[seed], outputs=[], js=seed_js)

    file_format.change(
        lambda f: (gr.Gallery(format=f), gr.Image(format=f)),
        inputs=[file_format],
        outputs=[output_images, image_prompt],
        show_api=False,
    )

    # input events are only user input; change events are both user and programmatic
    aspect_ratio.input(
        None,
        inputs=[aspect_ratio, width, height],
        outputs=[width, height],
        js=aspect_ratio_js,
    )

    # lock the input image so you don't lose it when the gallery updates
    output_images.change(
        gallery_fn,
        inputs=[output_images, image_prompt],
        outputs=[image_select],
        show_api=False,
    )

    # show the selected image in the image input
    image_select.change(
        image_select_fn,
        inputs=[output_images, image_prompt, image_select],
        outputs=[image_prompt],
        show_api=False,
    )

    # reset the dropdown on clear
    image_prompt.clear(
        image_prompt_fn,
        inputs=[output_images],
        outputs=[image_select],
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

    gr.on(
        triggers=[generate_btn.click, prompt.submit],
        fn=generate_fn,
        api_name="generate",
        concurrency_limit=5,
        outputs=[output_images],
        inputs=[
            prompt,
            negative_prompt,
            image_prompt,
            embeddings,
            style,
            seed,
            model,
            scheduler,
            width,
            height,
            guidance_scale,
            inference_steps,
            denoising_strength,
            num_images,
            use_karras,
            use_taesd,
            use_freeu,
            use_clip_skip,
            truncate_prompts,
            increment_seed,
            deepcache_interval,
            tome_ratio,
            scale,
        ],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-s", "--server", type=str, metavar="STR", default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, metavar="INT", default=7860)
    args = parser.parse_args()

    # https://www.gradio.app/docs/gradio/interface#interface-queue
    demo.queue().launch(
        server_name=args.server,
        server_port=args.port,
    )
