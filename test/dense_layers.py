import cv2
import keras
import numpy as np
from keras import layers
import matplotlib.pyplot as plt


def main():
    img_size = 32
    img = cv2.imread("./datasets/images/Apple/Apple_healthy/image (1).JPG")
    img = cv2.resize(img, (img_size, img_size))

    model = keras.Sequential()
    model.add(
        layers.Conv2D(
            input_shape=(img_size, img_size, 3), filters=64, kernel_size=(3, 3)
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))  # (1, 29, 29, 64)
    model.add(layers.Flatten())  # (1, 53824) => 29 x 29 x 64
    model.add(layers.Dense(units=10))  # (1, 10)

    # Displaying results
    result = model.predict(np.array([img]))
    print(result.shape)

    # Display image
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
