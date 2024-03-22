import sys

from src.train.save_model import save_model
from src.train.build_model import build_model
from src.train.load_dataset import load_dataset
from src.train.plotting import plotting_evolution


def main():
    """
    Entry point of the program. Trains a model on the provided dataset.

    Raises:
        ValueError: If the number of command-line arguments is not 2.

    Usage:
        python train.py [images folder]

    Args:
        sys.argv: Command-line arguments passed to the program.
    """
    if len(sys.argv) != 2:
        raise ValueError("usage: python train.py [images folder]")

    # Dataset
    training_data, validation_data = load_dataset(sys.argv[1])

    # Model
    model = build_model(training_data, True)

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=5,
    )

    # Plotting evolution
    plotting_evolution(history)

    # Saving model
    save_model(model)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
