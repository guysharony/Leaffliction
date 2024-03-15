import sys
import matplotlib.pyplot as plt

from keras.utils import image_dataset_from_directory
from keras import models, Sequential, layers

import tensorflow as ts


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python Training.py [images folder]")

    batch_size = 32
    img_height = 256
    img_width = 256
    data_dir = sys.argv[1]

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    num_classes = len(class_names)

    model = models.Sequential()
    model.layers.append(layers.Rescaling(1.0 / 255))
    model.layers.append(layers.Conv2D(32, ()))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
