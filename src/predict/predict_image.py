import numpy as np
import tensorflow as tf
from src.predict.preprocess_image import preprocess_image


def predict_image(
    image_path: str,
    model: 'tf.keras.engine.sequential.Sequential'
):
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
