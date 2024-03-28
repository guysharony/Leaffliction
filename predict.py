import os
import sys

from keras import models
from keras.preprocessing.image import ImageDataGenerator
from Augmentation import is_jpeg
from src.predict.arguments import arguments
from src.predict.predict_image import predict_image
from src.predict.get_class_labels import get_class_labels
from src.predict.display_prediction import display_prediction


def prediction_on_image(image_path):
    sub_directory = os.path.dirname(image_path)
    main_directory = os.path.dirname(sub_directory)

    # load the saved model
    class_labels = get_class_labels(main_directory)
    class_names = list(class_labels.keys())

    if len(class_names) < 2:
        raise ValueError("At least 2 class labels are required.")

    dataset_category = class_names[0].split("_")[0].lower()
    for category in class_names[1:]:
        if category.split("_")[0].lower() != dataset_category:
            raise ValueError(
                "Class labels must all belong to the same category."
            )

    model = models.load_model(f"model_{dataset_category}.keras")

    predicted_class = predict_image(image_path, model)

    for label, value in class_labels.items():
        if value == predicted_class:
            predicted_class_label = label
            break

    print("Predicted class:", predicted_class_label)
    display_prediction(image_path, predicted_class_label)


def prediction_on_batch(batch_path):
    


def main():
    args = arguments()

    if args.image:
        prediction_on_image(args.image)
    elif args.batch:
        prediction_on_batch(args.batch[0])

if __name__ == "__main__":
    #try:
    main()
    #except Exception as error:
    #    print(f"error: {error}")
