import os
import sys
import argparse

from Augmentation import is_jpeg

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        nargs="?",
        type=str,
        help="path to image to predict",
    )
    parser.add_argument(
        "--batch",
        type=str,
        nargs=1,
        default=None,
        help="path to batch of images to predict"
    )
    args = parser.parse_args(sys.argv[1::])

    if args.image is None and args.batch is None:
        raise ValueError("Please specify an image or a batch directory.")
    elif args.image and args.batch:
        raise ValueError(
            "[-src, -path] Image and batch cannot be \
                specified at the same time."
        )

    if args.image and (
        not os.path.isfile(args.image) or not is_jpeg(args.image)
    ):
        raise ValueError("Image to predict must be a valid jpg file.")

    return args