from keras.preprocessing.image import ImageDataGenerator


def get_class_labels(directory_path: str) -> dict:
    """
    Retrieves class labels and their corresponding indices
    from the given directory path.

    Args:
        directory_path (str): The path to the directory
        containing subdirectories representing classes.

    Returns:
        dict: A dictionary mapping class names
        to their corresponding indices.
    """
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
        directory_path,
        class_mode="categorical",
    )
    return generator.class_indices
