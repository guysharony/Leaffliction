import os
import sys

from keras import models
from Augmentation import is_jpeg
from src.predict.predict_image import predict_image
from src.predict.get_class_labels import get_class_labels
from src.predict.display_prediction import display_prediction


def main():
    assert len(sys.argv) == 2, "Only one argument required."
    assert os.path.isfile(sys.argv[1]) and is_jpeg(
        sys.argv[1]
    ), "Argument is not a valid .jpg file"

    image_path = sys.argv[1]
    sub_directory = os.path.dirname(image_path)
    main_directory = os.path.dirname(sub_directory)

    # load the saved model
    model = models.load_model("model.keras")

    class_labels = get_class_labels(main_directory)
    predicted_class = predict_image(image_path, model)

    for label, value in class_labels.items():
        if value == predicted_class:
            predicted_class_label = label
            break

    print("Predicted class:", predicted_class_label)
    display_prediction(image_path, predicted_class_label)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
