import matplotlib.pyplot as plt
from Transformation import Transformation


def display_prediction(image_path, predicted_class_label):
    """
    Displays the original and transformed images
    with the predicted class label.

    Args:
        image_path (str): path to the image file
        predicted_class_label (str): predicted class label
    """
    transformation = Transformation(image_path, "mask", None)
    transformation.transformations()

    original_img = transformation.image
    transformed_img = transformation._mask

    fig, axs = plt.subplots(1, 2)

    for ax in axs:
        ax.set_facecolor('black')

    fig.patch.set_facecolor('black')

    axs[0].imshow(original_img)
    axs[0].axis('off')

    axs[1].imshow(transformed_img)
    axs[1].axis('off')

    fig.suptitle(
        f'Class predicted: {predicted_class_label.lower()}',
        color='white'
    )

    plt.show()
