import os
import sys
import datetime

from keras import callbacks

from src.training.build_model import build_model
from src.training.load_dataset import load_dataset
from src.training.plotting import plotting_evolution


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python Training.py [images folder]")

    # Dataset
    training_data, validation_data = load_dataset(sys.argv[1])

    # Model
    model = build_model(training_data, True)

    # TensorBoard
    log_dir = f"logs/filters/conv2D_32-64-128_regularize"
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=15,
        callbacks=[tensorboard_callback],
    )

    # Plotting evolution
    plotting_evolution(history)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
