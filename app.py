import time

import gradio as gr

from generate import generate

DEFAULT_NEGATIVE_PROMPT = "<fast_negative>"

# base font stacks
MONO_FONTS = ["monospace"]
SANS_FONTS = [
    "sans-serif",
    "Apple Color Emoji",
    "Segoe UI Emoji",
    "Segoe UI Symbol",
    "Noto Color Emoji",
]


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# don't request a GPU if input is bad
def generate_btn_click(*args, **kwargs):
    start = time.perf_counter()

    if "prompt" in kwargs:
        prompt = kwargs.get("prompt")
    elif len(args) > 0:
        prompt = args[0]
    else:
        prompt = None

    if prompt is None or prompt.strip() == "":
        raise gr.Error("You must enter a prompt")

    images = generate(*args, **kwargs, Error=gr.Error)
    end = time.perf_counter()
    diff = end - start
    gr.Info(f"Generated {len(images)} images in {diff:.2f}s")
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
        font=[gr.themes.GoogleFont("Inter"), *SANS_FONTS],
        font_mono=[gr.themes.GoogleFont("Ubuntu Mono"), *MONO_FONTS],
    ).set(
        layout_gap="8px",
        block_shadow="0 0 #0000",
        block_shadow_dark="0 0 #0000",
        block_background_fill=gr.themes.colors.gray.c50,
        block_background_fill_dark=gr.themes.colors.gray.c900,
    ),
) as demo:
    gr.HTML(read_file("./partials/intro.html"))

    with gr.Group():
        output_images = gr.Gallery(
            elem_classes=["gallery"],
            show_share_button=False,
            interactive=False,
            show_label=False,
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
            elem_classes=["icon-button"],
            variant="secondary",
            elem_id="random",
            min_width=0,
            value="🎲",
            scale=1,
        )
        clear_btn = gr.ClearButton(
            elem_classes=["icon-button"],
            components=[output_images],
            variant="secondary",
            elem_id="clear",
            min_width=0,
            value="🗑️",
            scale=1,
        )

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
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        placeholder="",
                        lines=2,
                    )

                    with gr.Row():
                        num_images = gr.Dropdown(
                            choices=list(range(1, 9)),
                            filterable=False,
                            label="Images",
                            value=1,
                            scale=1,
                        )
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=1024,
                            value=448,
                            step=32,
                            scale=2,
                        )
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1024,
                            value=576,
                            step=32,
                            scale=2,
                        )

                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=15.0,
                            value=7,
                            step=0.1,
                        )
                        inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            value=30,
                            step=1,
                        )

                    with gr.Row():
                        model = gr.Dropdown(
                            value="Lykon/dreamshaper-8",
                            label="Model",
                            scale=2,
                            choices=[
                                "fluently/Fluently-v4",
                                "Linaqruf/anything-v3-1",
                                "Lykon/dreamshaper-8",
                                "prompthero/openjourney-v4",
                                "runwayml/stable-diffusion-v1-5",
                                "SG161222/Realistic_Vision_V5.1_Novae",
                            ],
                        )
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value="DEIS 2M",
                            scale=2,
                            choices=[
                                "DEIS 2M",
                                "DPM++ 2M",
                                "DPM2 a",
                                "Euler a",
                                "Heun",
                                "LMS",
                                "PNDM",
                            ],
                        )
                        seed = gr.Number(label="Seed", value=42, scale=1)

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
                            scale=4,
                        )

            with gr.TabItem("🛠️ Advanced"):
                with gr.Group():
                    with gr.Row():
                        deepcache_interval = gr.Slider(
                            label="DeepCache Interval",
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                        )
                        tgate_step = gr.Slider(
                            label="T-GATE Step",
                            minimum=0,
                            maximum=50,
                            value=20,
                            step=1,
                        )
                        tome_ratio = gr.Slider(
                            label="ToMe Ratio",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.01,
                        )

                    with gr.Row():
                        use_taesd = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Tiny VAE",
                            value=False,
                            scale=1,
                        )
                        use_clip_skip = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Clip skip",
                            value=False,
                            scale=1,
                        )
                        truncate_prompts = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Truncate prompts",
                            value=False,
                            scale=3,
                        )

            with gr.TabItem("ℹ️ Info"):
                gr.Markdown(read_file("info.md"), elem_classes=["markdown"])

    # update the random seed using JavaScript
    random_btn.click(None, outputs=[seed], js="() => Math.floor(Math.random() * 2**32)")

    # ensure correct argument order
    generate_btn.click(
        generate_btn_click,
        api_name="generate",
        concurrency_limit=5,
        outputs=[output_images],
        inputs=[
            prompt,
            negative_prompt,
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
            tgate_step,
            tome_ratio,
        ],
    )

# https://www.gradio.app/docs/gradio/interface#interface-queue
demo.queue().launch(
    {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }
)
