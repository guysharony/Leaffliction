import os
import random
from keras import models
from src.predict.arguments import arguments
from src.predict.predict_image import predict_image
from src.predict.display_prediction import display_prediction


def prediction_on_image(args):
    image = args['image']
    model = args['model']
    labels = args['labels']

    # Load model
    model = models.load_model(model)

    predicted_class = predict_image(image, model)
    predicted_class_label = labels[predicted_class]

    print("Predicted class:", predicted_class_label)
    display_prediction(image, predicted_class_label)


def prediction_on_batch(args):
    batch = args['batch']
    batch_size = args['batch_size']
    model = args['model']
    labels = args['labels']

    # Load model
    model = models.load_model(model)

    # List batch subdirectories
    label_directories = [os.path.join(batch, labeldir) for labeldir in os.listdir(batch) if os.path.isdir(os.path.join(batch, labeldir))]

    # Number of image per label
    images_per_label = int(batch_size / len(label_directories))

    # Message
    print(f"=> Evaluate on {batch_size} ({images_per_label} per label) images.")

    # Store random images
    random_image_paths = []

    for labeldir in label_directories:
        image_files = [os.path.join(labeldir, file) for file in os.listdir(labeldir) if file.lower().endswith(('.jpg'))]
        random_images = random.sample(image_files, min(images_per_label, len(image_files)))
        random_image_paths.extend(random_images)

    valid = 0

    for random_image in random_image_paths:
        label = random_image.split('/')[-2]

        predicted_class = predict_image(random_image, model)
        predicted_class_label = labels[predicted_class]

        if predicted_class_label == label:
            valid += 1

    accuracy = (valid / batch_size) * 100

    print(f'Predicted {valid} of {batch_size} images.')
    print(f'Accuracy of {"{:.2f}".format(accuracy)}%')

def main():
    args = arguments()

    if args['image']:
        prediction_on_image(args)
    elif args['batch']:
        prediction_on_batch(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}")
