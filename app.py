import argparse
import os
from importlib.util import find_spec

# Improved GPU handling and progress bars
os.environ["ZEROGPU_V2"] = "1"

# Use Rust-based downloader
if find_spec("hf_transfer"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gradio as gr
from huggingface_hub._snapshot_download import snapshot_download

from lib import (
    Config,
    generate,
    read_file,
    read_json,
)

# Update refresh button hover text
seed_js = """
(seed) => {
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${seed}"`);
    return seed;
}
"""

# The CSS `content` attribute expects a string so we need to wrap the number in quotes
refresh_seed_js = """
() => {
    const n = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${n}"`);
    return n;
}
"""

# Update width and height on aspect ratio change
aspect_ratio_js = """
(ar, w, h) => {
    if (!ar) return [w, h];
    const [width, height] = ar.split(",");
    return [parseInt(width), parseInt(height)];
}
"""

# Show "Custom" aspect ratio when manually changing width or height, or one of the predefined ones
custom_aspect_ratio_js = """
(w, h) => {
    if (w === 384 && h === 672) return "384,672";
    if (w === 448 && h === 576) return "448,576";
    if (w === 512 && h === 512) return "512,512"; 
    if (w === 576 && h === 448) return "576,448";
    if (w === 672 && h === 384) return "672,384";
    return null;
}
"""

random_prompt_js = f"""
(prompt) => {{
    const prompts = {read_json("data/prompts.json")};
    const filtered = prompts.filter(p => p !== prompt);
    return filtered[Math.floor(Math.random() * filtered.length)];
}}
"""

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
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Ubuntu Mono"), "monospace"],
    ).set(
        layout_gap="8px",
        block_shadow="0 0 #0000",
        block_shadow_dark="0 0 #0000",
        block_background_fill=gr.themes.colors.gray.c50,
        block_background_fill_dark=gr.themes.colors.gray.c900,
    ),
) as demo:
    gr.HTML(read_file("./partials/intro.html"))

    with gr.Tabs():
        with gr.TabItem("🏠 Home"):
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
                positive_prompt = gr.Textbox(
                    placeholder="What do you want to see?",
                    autoscroll=False,
                    show_label=False,
                    label="Prompt",
                    max_lines=3,
                    lines=3,
                )
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

        with gr.TabItem("⚙️ Settings", elem_id="settings"):
            # Prompt settings
            gr.HTML("<h3>Prompt</h3>")
            with gr.Row():
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="nsfw, <fast_negative>",
                    lines=1,
                )

            # Model settings
            gr.HTML("<h3>Model</h3>")
            with gr.Row():
                model = gr.Dropdown(
                    choices=Config.MODELS,
                    value=Config.MODEL,
                    filterable=False,
                    label="Checkpoint",
                    min_width=240,
                )
                scheduler = gr.Dropdown(
                    choices=Config.SCHEDULERS.keys(),
                    value=Config.SCHEDULER,
                    elem_id="scheduler",
                    label="Scheduler",
                    filterable=False,
                )

            # Generation settings
            gr.HTML("<h3>Generation</h3>")
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
                    value=-1,
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

            # Image-to-Image settings
            gr.HTML("<h3>Image-to-Image</h3>")
            with gr.Row():
                image_input = gr.Image(
                    show_share_button=False,
                    label="Initial Image",
                    min_width=640,
                    format="png",
                    type="pil",
                )
            with gr.Row():
                controlnet_input = gr.Image(
                    show_share_button=False,
                    label="Control Image",
                    min_width=320,
                    format="png",
                    type="pil",
                )
                ip_adapter_input = gr.Image(
                    show_share_button=False,
                    label="IP-Adapter Image",
                    min_width=320,
                    format="png",
                    type="pil",
                )
            with gr.Row():
                denoising_strength = gr.Slider(
                    label="Initial Image Strength",
                    value=Config.DENOISING_STRENGTH,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                )
                controlnet_annotator = gr.Dropdown(
                    label="ControlNet Annotator",
                    # TODO: annotators should be in config with names
                    choices=[("Canny", "canny")],
                    value=Config.ANNOTATOR,
                    filterable=False,
                )
            with gr.Row():
                use_ip_adapter_face = gr.Checkbox(
                    label="Use IP-Adapter Face",
                    elem_classes=["checkbox"],
                    value=False,
                )

        with gr.TabItem("ℹ️ Info"):
            gr.Markdown(read_file("DOCS.md"))

    # Random prompt on click
    random_btn.click(
        None, inputs=[positive_prompt], outputs=[positive_prompt], js=random_prompt_js
    )

    # Update seed on click
    refresh_btn.click(None, inputs=[], outputs=[seed], js=refresh_seed_js)

    # Update seed button hover text
    seed.change(None, inputs=[seed], outputs=[], js=seed_js)

    # Update width and height on aspect ratio change
    aspect_ratio.input(
        None,
        inputs=[aspect_ratio, width, height],
        outputs=[width, height],
        js=aspect_ratio_js,
    )

    # Show "Custom" aspect ratio when manually changing width or height
    gr.on(
        triggers=[width.input, height.input],
        fn=None,
        inputs=[width, height],
        outputs=[aspect_ratio],
        js=custom_aspect_ratio_js,
    )

    # Generate images
    gr.on(
        triggers=[generate_btn.click, positive_prompt.submit],
        fn=generate,
        api_name="generate",
        outputs=[output_images],
        inputs=[
            positive_prompt,
            negative_prompt,
            image_input,
            controlnet_input,
            ip_adapter_input,
            seed,
            model,
            scheduler,
            controlnet_annotator,
            width,
            height,
            guidance_scale,
            inference_steps,
            denoising_strength,
            deepcache_interval,
            scale,
            num_images,
            use_karras,
            use_ip_adapter_face,
        ],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-s", "--server", type=str, metavar="STR", default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, metavar="INT", default=7860)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", None)
    for repo_id, allow_patterns in Config.HF_REPOS.items():
        snapshot_download(
            repo_id,
            repo_type="model",
            revision="main",
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=None,
        )

    # https://www.gradio.app/docs/gradio/interface#interface-queue
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server,
        server_port=args.port,
    )
