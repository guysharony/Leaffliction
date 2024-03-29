import os
import sys
import pickle
import argparse

from Augmentation import is_jpeg


def find_labels_from_model(model):
    model_filename = os.path.basename(model)
    model_category = os.path.splitext(model_filename)[0].split("_")[1]

    return f"./labels_{model_category}.pickle"


def filter_arguments(args):
    # Verify image and batch arguments
    image = args.image
    batch = args.batch[0] if args.batch else None

    if image is None and batch is None:
        raise ValueError("Please specify an image or a batch directory.")
    elif image and batch:
        raise ValueError(
            "[-src, -path] Image and batch cannot be \
                specified at the same time."
        )

    # Verify image type
    if image and (
        not os.path.isfile(image) or not is_jpeg(image)
    ):
        raise ValueError("[image] Image to predict must be a valid jpg file.")

    # Batch size
    batch_size = args.batch_size[0] if args.batch_size else None
    if image and batch_size is not None:
        raise ValueError(
            "[--batch_size] Batch size can only be used with --batch."
        )

    if batch_size < 100:
        raise ValueError("[--batch_size] Must be at least 100.")

    if batch and batch_size is None:
        batch_size = 100

    # Verify model
    model = args.model[0] if args.model else None
    if model is None:
        raise ValueError("[--model] Model must be specified")

    # Determine labels
    labels_path = args.labels[0] if args.labels else None
    if labels_path is None:
        labels_path = find_labels_from_model(model)

    try:
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)
    except Exception:
        raise ValueError("[--labels] Failed to load class labels.")

    return {
        'image': image,
        'batch': batch,
        'batch_size': batch_size,
        'model': model,
        'labels': labels
    }


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        nargs="?",
        type=str,
        help="Path to image to predict.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        nargs=1,
        default=None,
        help="Path to batch of images to predict."
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs=1,
        default=None,
        help="Path to the model."
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs=1,
        default=None,
        help="Path to the class labals. If not specified, the program \
            will determine the class labels from model filename."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs=1,
        default=None,
        help="Number of images in batch test."
    )

    args = parser.parse_args(sys.argv[1::])

    return filter_arguments(args)
