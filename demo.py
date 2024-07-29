import time

import gradio as gr

from generate import generate

# base font stacks
mono_fonts = ["monospace"]
sans_fonts = [
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
    head=read_file("head.html"),
    css="./demo.css",
    js="./demo.js",
    theme=gr.themes.Default(
        # colors
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        # sizing
        text_size=gr.themes.sizes.text_md,
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_sm,
        # fonts
        font=[gr.themes.GoogleFont("Inter"), *sans_fonts],
        font_mono=[gr.themes.GoogleFont("Ubuntu Mono"), *mono_fonts],
    ).set(
        block_background_fill=gr.themes.colors.gray.c50,
        block_background_fill_dark=gr.themes.colors.gray.c900,
        block_shadow="0 0 #0000",
        block_shadow_dark="0 0 #0000",
    ),
) as demo:
    gr.HTML(read_file("intro.html"))
    output_images = gr.Gallery(
        label="Output",
        show_label=False,
        columns=1,
        interactive=False,
        show_share_button=False,
        elem_id="gallery",
    )
    prompt = gr.Textbox(
        label="Prompt",
        show_label=False,
        lines=2,
        placeholder="corgi, at the beach, cute",
        value=None,
    )
    generate_btn = gr.Button("Generate", variant="primary", elem_classes=[])

    with gr.Accordion(
        label="Menu",
        open=False,
        elem_id="menu",
        elem_classes=["accordion"],
    ):
        with gr.Tabs():
            with gr.TabItem("⚙️ Settings"):
                with gr.Group():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        lines=1,
                        placeholder="ugly",
                        value="",
                    )

                    with gr.Row():
                        num_images = gr.Dropdown(
                            label="Images",
                            choices=[1, 2, 3, 4],
                            value=1,
                            filterable=False,
                        )
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=["1:1", "4:3", "3:4", "16:9", "9:16"],
                            value="1:1",
                            filterable=False,
                        )
                        seed = gr.Number(label="Seed", value=0)

                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=15.0,
                            step=0.1,
                            value=7.5,
                        )
                        inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=30,
                        )

                    with gr.Row():
                        model = gr.Dropdown(
                            label="Model",
                            choices=[
                                "fluently/Fluently-v4",
                                "Linaqruf/anything-v3-1",
                                "Lykon/dreamshaper-8",
                                "prompthero/openjourney-v4",
                                "runwayml/stable-diffusion-v1-5",
                                "SG161222/Realistic_Vision_V5.1_Novae",
                            ],
                            value="Lykon/dreamshaper-8",
                        )
                        scheduler = gr.Dropdown(
                            label="Scheduler",
                            choices=[
                                "DEIS 2M",
                                "DPM++ 2M",
                                "DPM2 a",
                                "Euler a",
                                "Heun",
                                "LMS",
                                "PNDM",
                            ],
                            value="DEIS 2M",
                            elem_id="scheduler",
                        )

                    with gr.Row():
                        use_karras = gr.Checkbox(
                            label="Use Karras σ",
                            value=True,
                            elem_classes=["checkbox"],
                            scale=2,
                        )
                        increment_seed = gr.Checkbox(
                            label="Autoincrement seed",
                            value=True,
                            elem_classes=["checkbox"],
                            elem_id="increment-seed",
                            scale=2,
                        )
                        random_seed_btn = gr.Button(
                            "🎲 Random seed",
                            variant="secondary",
                            size="sm",
                            scale=1,
                        )

            with gr.TabItem("🛠️ Advanced"):
                gr.Markdown("_Coming soon..._", elem_classes=["markdown"])

            with gr.TabItem("ℹ️ Info"):
                gr.Markdown(read_file("info.md"), elem_classes=["markdown"])

    # change gallery columns when num_images changes
    num_images.change(
        lambda n: gr.Gallery(columns=n),
        inputs=[num_images],
        outputs=[output_images],
    )

    # update the random seed using JavaScript
    random_seed_btn.click(None, outputs=[seed], js="() => Math.floor(Math.random() * 2**32)")

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
            aspect_ratio,
            guidance_scale,
            inference_steps,
            use_karras,
            num_images,
            increment_seed,
        ],
    )

# https://www.gradio.app/docs/gradio/interface#interface-queue
demo.queue().launch(
    {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }
)
