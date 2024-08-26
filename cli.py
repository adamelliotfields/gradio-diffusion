# CLI
# usage: python cli.py 'colorful calico cat artstation'
import argparse
import asyncio

from lib import Config, async_call, generate


def save_images(images, filename="image.png"):
    for i, (img, _) in enumerate(images):
        name, ext = filename.rsplit(".", 1)
        img.save(f"{name}.{ext}" if len(images) == 1 else f"{name}_{i}.{ext}")


async def main():
    # fmt: off
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("prompt", type=str, metavar="PROMPT")
    parser.add_argument("-n", "--negative", type=str, metavar="STR", default="")
    parser.add_argument("-e", "--embedding", type=str, metavar="STR", default=[], action="append")
    parser.add_argument("-s", "--seed", type=int, metavar="INT", default=Config.SEED)
    parser.add_argument("-i", "--images", type=int, metavar="INT", default=1)
    parser.add_argument("-f", "--filename", type=str, metavar="STR", default="image.png")
    parser.add_argument("-w", "--width", type=int, metavar="INT", default=Config.WIDTH)
    parser.add_argument("-h", "--height", type=int, metavar="INT", default=Config.HEIGHT)
    parser.add_argument("-m", "--model", type=str, metavar="STR", default=Config.MODEL)
    parser.add_argument("-d", "--deepcache", type=int, metavar="INT", default=Config.DEEPCACHE_INTERVAL)
    parser.add_argument("--scale", type=int, metavar="INT", choices=Config.SCALES, default=Config.SCALE)
    parser.add_argument("--style", type=str, metavar="STR", default=Config.STYLE)
    parser.add_argument("--scheduler", type=str, metavar="STR", default=Config.SCHEDULER)
    parser.add_argument("--guidance", type=float, metavar="FLOAT", default=Config.GUIDANCE_SCALE)
    parser.add_argument("--steps", type=int, metavar="INT", default=Config.INFERENCE_STEPS)
    parser.add_argument("--strength", type=float, metavar="FLOAT", default=Config.DENOISING_STRENGTH)
    parser.add_argument("--image", type=str, metavar="STR")
    parser.add_argument("--ip-image", type=str, metavar="STR")
    parser.add_argument("--ip-face", action="store_true")
    parser.add_argument("--taesd", action="store_true")
    parser.add_argument("--clip-skip", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--karras", action="store_true")
    parser.add_argument("--freeu", action="store_true")
    parser.add_argument("--no-increment", action="store_false")
    # fmt: on

    args = parser.parse_args()
    images = await async_call(
        generate,
        args.prompt,
        args.negative,
        args.image,
        args.ip_image,
        args.ip_face,
        args.embedding,
        args.style,
        args.seed,
        args.model,
        args.scheduler,
        args.width,
        args.height,
        args.guidance,
        args.steps,
        args.strength,
        args.images,
        args.karras,
        args.taesd,
        args.freeu,
        args.clip_skip,
        args.truncate,
        args.no_increment,
        args.deepcache,
        args.scale,
    )
    await async_call(save_images, images, args.filename)


if __name__ == "__main__":
    asyncio.run(main())
