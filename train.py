import sys
from src.train.save_learnings import save_learnings
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

    # Check categories
    class_names = training_data.class_names

    if len(class_names) < 2:
        raise ValueError("At least 2 class names are required.")

    dataset_category = class_names[0].split("_")[0].lower()
    for category in class_names[1:]:
        if category.split("_")[0].lower() != dataset_category:
            raise ValueError(
                "Class names must all belong to the same category."
            )

    # Model
    model = build_model(class_names, True)

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=15,
    )

    # Plotting evolution
    plotting_evolution(history)

    # Saving model
    save_learnings(model, dataset_category, class_names)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
