import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from Augmentation import is_jpeg


def preprocess_image(image_path: str) -> 'np.ndarray':
    """
    Preprocess the image by loading it and converting it
    to a format suitable for prediction.
    - resize
    - normalize

    Args:
        image_path (str): path to image file
    Returns:
        'np.ndarray': Preprocessed image array
    """
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0


def load_and_predict(image_path: str,
                     model: 'tf.keras.engine.sequential.Sequential'):
    """
    Loads an image, preprocesses it, and predicts
    its class using the provided model.

    Args:
        image_path (str): The path to the image file.
        model (tf.keras.Model): The pre-trained model
                                to use for prediction.

    Returns:
        int: The predicted class index.
    """
    prediction = model.predict(preprocess_image(image_path))
    return np.argmax(prediction)


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


def display_prediction(image_path, predicted_class_label):
    """
    Displays the original and transformed images
    with the predicted class label.

    Args:
        image_path (str): path to the image file
        predicted_class_label (str): predicted class label
    """
    original_img = cv2.imread(image_path)
    transformed_img = cv2.imread(image_path)

    canvas_width = original_img.shape[1] * 2
    canvas_height = (
        max(original_img.shape[0], transformed_img.shape[0]) + 100
    )

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    original_img_start_x = 0
    original_img_end_x = original_img.shape[1]
    original_img_start_y = 0
    original_img_end_y = original_img_start_y + original_img.shape[0]

    transformed_img_start_x = original_img_end_x
    transformed_img_end_x = transformed_img_start_x + transformed_img.shape[1]
    transformed_img_start_y = 0
    transformed_img_end_y = transformed_img_start_y + transformed_img.shape[0]

    canvas[
        original_img_start_y:original_img_end_y,
        original_img_start_x:original_img_end_x
    ] = original_img

    canvas[
        transformed_img_start_y:transformed_img_end_y,
        transformed_img_start_x:transformed_img_end_x,
    ] = transformed_img

    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    font_thickness = 1

    # to display
    text = f"Predicted Class: {predicted_class_label}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # text location
    text_x = (canvas_width - text_size[0]) // 2
    text_y = canvas_height - 20

    cv2.putText(
        canvas,
        text,
        (text_x, text_y),
        font, font_scale,
        font_color,
        font_thickness
    )

    cv2.imshow("Prediction", canvas)

    while True:
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    try:
        assert len(sys.argv) == 2, "Only one argument required."
        assert os.path.isfile(sys.argv[1]) and is_jpeg(
            sys.argv[1]
        ), "Argument is not a valid .jpg file"

        image_path = sys.argv[1]
        sub_directory = os.path.dirname(image_path)
        main_directory = os.path.dirname(sub_directory)

        # load the saved model
        model = tf.keras.models.load_model("model.keras")

        class_labels = get_class_labels(main_directory)
        predicted_class = load_and_predict(image_path, model)

        for label, value in class_labels.items():
            if value == predicted_class:
                predicted_class_label = label
                break

        print("Predicted class:", predicted_class_label)
        display_prediction(image_path, predicted_class_label)

    except Exception as error:
        print(f"error: {error}")
