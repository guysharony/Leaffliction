from keras import models
from keras import layers
from keras import losses
from keras import regularizers

from keras.optimizers import Adam


def build_model(class_names, summary=False):
    """
    Builds and compiles a convolutional neural network model for image
    classification.

    Args:
        class_names: Class names of dataset.
        summary (bool, optional): Whether to print the model summary.
            Defaults to False.

    Returns:
        keras.models.Sequential: A compiled convolutional neural network model.

    """
    number_classes = len(class_names)

    # Initialize model
    model = models.Sequential()

    # Rescaling layer
    model.add(layers.Rescaling(1.0 / 255))

    # Blocks 1
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Blocks 2
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Blocks 3
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.1))

    # Blocks 4
    model.add(
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.1))

    # Flatten
    model.add(layers.Flatten())

    # Dense layers with dropout
    model.add(
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.01)
        )
    )

    # Output layer
    model.add(layers.Dense(number_classes, activation="softmax"))

    # Compile
    model.compile(
        optimizer=Adam(0.0001),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.build(input_shape=(None, 256, 256, 3))

    if summary is True:
        model.summary()

    return model
