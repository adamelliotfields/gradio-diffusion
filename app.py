import argparse
import json

import gradio as gr

import config as cfg
from generate import generate

# the CSS `content` attribute expects a string so we need to wrap the number in quotes
random_seed_js = """
() => {
    const n = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    const button = document.getElementById("random");
    button.style.setProperty("--seed", `"${n}"`);
    return n;
}
"""

seed_js = """
(seed) => {
    const button = document.getElementById("random");
    button.style.setProperty("--seed", `"${seed}"`);
    return seed;
}
"""


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def handle_generate(*args):
    if len(args) > 0:
        prompt = args[0]
    else:
        prompt = None
    if prompt is None or prompt.strip() == "":
        raise gr.Error("You must enter a prompt")
    try:
        images = generate(*args, log=gr.Info, Error=gr.Error)
    except RuntimeError:
        raise gr.Error("RuntimeError: Please try again")
    return images


with open("./styles/twri.json", "r") as f:
    styles = json.load(f)

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
        font=[gr.themes.GoogleFont("Inter"), *cfg.SANS_FONTS],
        font_mono=[gr.themes.GoogleFont("Ubuntu Mono"), *cfg.MONO_FONTS],
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
        label="Open menu",
        open=False,
    ):
        with gr.Tabs():
            with gr.TabItem("⚙️ Settings"):
                with gr.Group():
                    negative_prompt = gr.Textbox(
                        value=cfg.NEGATIVE_PROMPT,
                        label="Negative Prompt",
                        placeholder="ugly, bad",
                        lines=2,
                    )

                    model = gr.Dropdown(
                        value=cfg.MODEL,
                        filterable=False,
                        label="Model",
                        choices=cfg.MODELS,
                    )

                    with gr.Row():
                        style = gr.Dropdown(
                            value=cfg.STYLE,
                            label="Style",
                            choices=[("None", None)]
                            + [(style["name"], style["id"]) for style in styles],
                        )
                        scheduler = gr.Dropdown(
                            value=cfg.SCHEDULER,
                            elem_id="scheduler",
                            label="Scheduler",
                            filterable=False,
                            min_width=200,
                            choices=cfg.SCHEDULERS,
                        )

                    with gr.Row():
                        guidance_scale = gr.Slider(
                            value=cfg.GUIDANCE_SCALE,
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=15.0,
                            step=0.1,
                        )
                        inference_steps = gr.Slider(
                            value=cfg.INFERENCE_STEPS,
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                        )
                        seed = gr.Number(
                            value=cfg.SEED,
                            label="Seed",
                            minimum=-1,
                            maximum=(2**64) - 1,
                        )

                    with gr.Row():
                        width = gr.Slider(
                            value=cfg.WIDTH,
                            label="Width",
                            minimum=320,
                            maximum=768,
                            step=32,
                        )
                        height = gr.Slider(
                            value=cfg.HEIGHT,
                            label="Height",
                            minimum=320,
                            maximum=768,
                            step=32,
                        )
                        num_images = gr.Dropdown(
                            choices=list(range(1, 5)),
                            value=cfg.NUM_IMAGES,
                            filterable=False,
                            label="Images",
                        )

                    with gr.Row():
                        use_karras = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Karras σ",
                            value=True,
                            scale=1,
                        )
                        increment_seed = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Autoincrement",
                            value=True,
                            scale=1,
                        )
                        upscale_4x = gr.Checkbox(
                            interactive=cfg.NUM_IMAGES == 1,
                            elem_classes=["checkbox"],
                            label="Upscale 4x",
                            value=False,
                            scale=3,
                        )

            with gr.TabItem("🛠️ Advanced"):
                with gr.Group():
                    with gr.Row():
                        file_format = gr.Dropdown(
                            choices=["png", "jpeg", "webp"],
                            label="File Format",
                            filterable=False,
                            value="png",
                        )
                        deepcache_interval = gr.Slider(
                            value=cfg.DEEPCACHE_INTERVAL,
                            label="DeepCache Interval",
                            minimum=1,
                            maximum=4,
                            step=1,
                        )
                        tome_ratio = gr.Slider(
                            value=cfg.TOME_RATIO,
                            label="ToMe Ratio",
                            minimum=0.0,
                            maximum=0.5,
                            step=0.01,
                        )

                    with gr.Row():
                        use_taesd = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Tiny VAE",
                            value=False,
                        )
                        use_clip_skip = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Clip skip",
                            value=False,
                        )
                        truncate_prompts = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Truncate prompts",
                            value=False,
                        )

            with gr.TabItem("ℹ️ Usage"):
                gr.Markdown(read_file("usage.md"), elem_classes=["markdown"])

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
            placeholder="corgi, at the beach, cute, 8k",
            show_label=False,
            label="Prompt",
            value=None,
            lines=2,
        )

    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary", scale=6, elem_classes=[])
        random_btn = gr.Button(
            elem_classes=["icon-button", "popover"],
            variant="secondary",
            elem_id="random",
            min_width=0,
            value="🎲",
            scale=1,
        )
        clear_btn = gr.ClearButton(
            elem_classes=["icon-button", "popover"],
            components=[output_images],
            variant="secondary",
            elem_id="clear",
            min_width=0,
            value="🗑️",
            scale=1,
        )

    # update the seed using JavaScript
    random_btn.click(None, outputs=[seed], js=random_seed_js)

    seed.change(
        None,
        inputs=[seed],
        outputs=[],
        js=seed_js,
    )

    num_images.change(
        lambda n, upscale: gr.Checkbox(interactive=n == 1, value=upscale if n == 1 else False),
        inputs=[num_images, upscale_4x],
        outputs=[upscale_4x],
    )

    file_format.change(
        lambda f: gr.Gallery(format=f),
        inputs=[file_format],
        outputs=[output_images],
    )

    gr.on(
        triggers=[generate_btn.click, prompt.submit],
        fn=handle_generate,
        api_name="api",
        concurrency_limit=5,
        outputs=[output_images],
        inputs=[
            prompt,
            negative_prompt,
            style,
            seed,
            model,
            scheduler,
            width,
            height,
            guidance_scale,
            inference_steps,
            num_images,
            use_karras,
            use_taesd,
            use_clip_skip,
            truncate_prompts,
            increment_seed,
            deepcache_interval,
            tome_ratio,
            upscale_4x,
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
