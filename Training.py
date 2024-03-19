import sys
import matplotlib.pyplot as plt

import tensorflow as ts
import matplotlib.pyplot as plt

from keras import models, layers, losses, callbacks
from keras.utils import image_dataset_from_directory
from Augmentation import balance_dataset


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
    # Initialize model
    model = models.Sequential()

    # Rescaling layer
    model.add(layers.Rescaling(1.0 / 255))

    # Blocks 1-5
    for filters in [16, 32, 64, 128, 128]:
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(layers.Flatten())

    # Dense layers with dropout
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(number_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def plotting_evolution(history):
    # Loss history
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    # Accuracy history
    training_accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]

    # Figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation loss values
    ax1.plot(training_loss, label="Training Loss")
    ax1.plot(validation_loss, label="Validation Loss")
    ax1.set_title("Model Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Plot training & validation accuracy values
    ax2.plot(training_accuracy, label="Training Accuracy")
    ax2.plot(validation_accuracy, label="Validation Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python Training.py [images folder]")

    # Balance dataset
    balanced_source = balance_dataset(sys.argv[1])

    # Dataset
    training_data, validation_data = preparing_dataset(balanced_source)

    # Model
    model = build_model(len(training_data.class_names))
    model.build(input_shape=(None, 256, 256, 3))

    # Summary
    print(model.summary())

    # Training
    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5)

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=10,
        callbacks=[early_stopping, reduce_lr],
    )

    # Plotting evolution
    plotting_evolution(history)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
