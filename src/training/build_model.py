from keras import models, layers, losses, callbacks, regularizers
from keras.optimizers import Adam, AdamW


def build_model(training_data, summary=False):
    number_classes = len(training_data.class_names)

    # Initialize model
    model = models.Sequential()

    # Rescaling layer
    model.add(layers.Rescaling(1.0 / 255))

    # Blocks 1-4
    for f in [32, 64, 64]:
        model.add(layers.Conv2D(filters=f, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.1))

    # Flatten
    model.add(layers.Flatten())

    # Dense layers with dropout
    model.add(layers.Dense(64, activation="relu"))

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
