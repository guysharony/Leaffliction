import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_predict(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Scale pixel values to [0, 1]

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return predicted_class


if __name__ == "__main__":
    # Load the saved model
    model = tf.keras.models.load_model('model.h5')

    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
    "./datasets/augmented_directory/images",
    class_mode='categorical',  # specify the class mode
    )

    image_path = './datasets/images/Apple/Apple_scab/image (1).JPG'
    predicted_class = load_and_predict(image_path, model)

    print('predicted class', predicted_class)
    class_labels = generator.class_indices
    print('class labels', class_labels)
    for label, value in class_labels.items():
        if value == predicted_class:
            predicted_class_label = label
            break
    print("Predicted class:", predicted_class_label)