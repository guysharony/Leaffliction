import numpy as np
from keras import preprocessing


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
    img = preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)
