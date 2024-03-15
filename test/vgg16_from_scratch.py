import cv2
import keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt


def main():
    # Load image
    img = cv2.imread("./datasets/images/Apple/Apple_healthy/image (1).JPG")
    img = cv2.resize(img, (224, 224))

    # VGG16
    model = keras.Sequential()

    # Block 1
    model.add(
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            input_shape=(img.shape[0], img.shape[1], 3),
        )
    )
    model.add(
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(
        layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(
        layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(
        layers.Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        layers.Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        layers.Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(
        layers.Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        layers.Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        layers.Conv2D(
            512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Top
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(
        layers.Dense(2, activation="softmax")
    )  # 2 is the number of categories to detect

    # Build
    model.build()
    model.summary()

    # Result
    result = model.predict(np.array([img]))
    print(result)

    # Displaying feature map
    # for i in range(512):
    #    feature_img = result[0, :, :, i]
    #    ax = plt.subplot(32, 16, i + 1)
    #    ax.set_xticks([])
    #    ax.set_yticks([])
    #    plt.imshow(feature_img, cmap="gray")
    # plt.show()


if __name__ == "__main__":
    main()
