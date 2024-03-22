import sys

from src.training.build_model import build_model
from src.training.load_dataset import load_dataset
from src.training.plotting import plotting_evolution


def main():
    """
    Entry point of the program. Trains a model on the provided dataset.

    Raises:
        ValueError: If the number of command-line arguments is not 2.

    Usage:
        python Training.py [images folder]

    Args:
        sys.argv: Command-line arguments passed to the program.
    """
    if len(sys.argv) != 2:
        raise ValueError("usage: python Training.py [images folder]")

    # Dataset
    training_data, validation_data = load_dataset(sys.argv[1])

    # Model
    model = build_model(training_data, True)

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=15,
    )

    # Plotting evolution
    plotting_evolution(history)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
