import sys
import matplotlib.pyplot as plt

import tensorflow as ts

from keras import models, layers, losses, callbacks
from keras.utils import image_dataset_from_directory


def preparing_dataset(directory):
    dataset = image_dataset_from_directory(
        directory,
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


def build_model(number_classes):
    model = models.Sequential()
    model.layers.append(layers.Rescaling(1.0 / 255))

    # Blocks
    model.add(
        layers.Conv2D(
            filters=16, kernel_size=3, activation="relu", input_shape=(256, 256, 3)
        )
    )
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=2))

    # Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(number_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python Training.py [images folder]")

    # Dataset
    training_data, validation_data = preparing_dataset(sys.argv[1])

    # Model
    model = build_model(len(training_data.class_names))

    # Summary
    print(model.summary())

    # Training
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=3,
        callbacks=[early_stopping],
        use_multiprocessing=True,
    )
    train_accuracy = history.history["accuracy"][-1]
    validation_accuracy = history.history["val_accuracy"][-1]

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")

    # Prediction
    print(model.predict(validation_data))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)