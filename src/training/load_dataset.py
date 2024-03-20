from Augmentation import balance_dataset

from keras.utils import image_dataset_from_directory


def load_dataset(path):
    # Balance dataset
    balanced_source = balance_dataset(path)

    # Preparing dataset
    dataset = image_dataset_from_directory(
        balanced_source,
        validation_split=0.2,
        subset="both",
        shuffle=True,
        seed=42,
        image_size=(256, 256),
        batch_size=32,
    )

    training_data = dataset[0]
    validation_data = dataset[1]

    return training_data, validation_data
