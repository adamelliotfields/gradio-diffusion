import argparse

from generate import generate


def save_images(images, filename="image.png"):
    for i, (img, _) in enumerate(images):
        name, ext = filename.rsplit(".", 1)
        img.save(f"{name}.{ext}" if len(images) == 1 else f"{name}_{i}.{ext}")


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("prompt", type=str, metavar="PROMPT")
    parser.add_argument("-n", "--negative", type=str, metavar="STR", default="<fast_negative>")
    parser.add_argument("-s", "--seed", type=int, metavar="INT")
    parser.add_argument("-i", "--images", type=int, metavar="INT", default=1)
    parser.add_argument("-f", "--filename", type=str, metavar="STR", default="image.png")
    parser.add_argument("-w", "--width", type=int, metavar="INT", default=448)
    parser.add_argument("-h", "--height", type=int, metavar="INT", default=576)
    parser.add_argument("-m", "--model", type=str, metavar="STR", default="Lykon/dreamshaper-8")
    parser.add_argument("-d", "--deepcache", type=int, metavar="INT", default=2)
    parser.add_argument("-t", "--tgate", type=int, metavar="INT", default=20)
    parser.add_argument("--scheduler", type=str, metavar="STR", default="DEIS 2M")
    parser.add_argument("--guidance", type=float, metavar="FLOAT", default=7)
    parser.add_argument("--steps", type=int, metavar="INT", default=30)
    parser.add_argument("--tome", type=float, metavar="FLOAT", default=0.0)
    parser.add_argument("--taesd", action="store_true")
    parser.add_argument("--clip-skip", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--no-karras", action="store_false")
    parser.add_argument("--no-increment", action="store_false")

    args = parser.parse_args()
    images = generate(
        args.prompt,
        args.negative,
        args.seed,
        args.model,
        args.scheduler,
        args.width,
        args.height,
        args.guidance,
        args.steps,
        args.images,
        args.no_karras,
        args.taesd,
        args.clip_skip,
        args.truncate,
        args.no_increment,
        args.deepcache,
        args.tgate,
        args.tome,
    )
    save_images(images, args.filename)


if __name__ == "__main__":
    main()
