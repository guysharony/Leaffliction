from keras import models
from src.predict.arguments import arguments
from src.predict.predict_image import predict_image
from src.predict.display_prediction import display_prediction


def prediction_on_image(args):
    image = args['image']
    model = args['model']
    labels = args['labels']

    # Load class labels
    model = models.load_model(model)

    predicted_class = predict_image(image, model)
    predicted_class_label = labels[predicted_class]

    print("Predicted class:", predicted_class_label)
    display_prediction(image, predicted_class_label)


def prediction_on_batch(batch_path):
    return True


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
