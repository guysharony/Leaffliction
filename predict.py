import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

from Augmentation import is_jpeg


def load_and_predict(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return predicted_class

def get_class_labels(directory_path: str) -> dict: 
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
    directory_path,
    class_mode='categorical',
    )
    return generator.class_indices

def display_prediction(image_path, predicted_class_label):
    original_img = cv2.imread(image_path)
    transformed_img = cv2.imread(image_path)

    canvas_width = original_img.shape[1] * 2
    canvas_height = max(original_img.shape[0], transformed_img.shape[0]) + 100  # Add space for text

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    original_img_start_x = 0
    original_img_end_x = original_img.shape[1]
    original_img_start_y = 0
    original_img_end_y = original_img_start_y + original_img.shape[0]

    transformed_img_start_x = original_img_end_x
    transformed_img_end_x = transformed_img_start_x + transformed_img.shape[1]
    transformed_img_start_y = 0
    transformed_img_end_y = transformed_img_start_y + transformed_img.shape[0]

    canvas[original_img_start_y:original_img_end_y, original_img_start_x:original_img_end_x] = original_img

    canvas[transformed_img_start_y:transformed_img_end_y, transformed_img_start_x:transformed_img_end_x] = transformed_img

    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 1
    text = f"Predicted Class: {predicted_class_label}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (canvas_width - text_size[0]) // 2
    text_y = canvas_height - 20
    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    cv2.imshow("Prediction", canvas)

    while True:
        if cv2.waitKey(1) == ord('q'):
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
        model = tf.keras.models.load_model('model.h5')

        class_labels = get_class_labels(main_directory)
        print(class_labels)
        predicted_class = load_and_predict(image_path, model)

        print('predicted class', predicted_class)
        print('class labels', class_labels)
        for label, value in class_labels.items():
            if value == predicted_class:
                predicted_class_label = label
                break
        print("Predicted class:", predicted_class_label)

        display_prediction(image_path, predicted_class_label)

    except Exception as error:
        print(f"error: {error}")
