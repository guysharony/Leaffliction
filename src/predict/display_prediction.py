import cv2
import numpy as np
from Transformation import Transformation


def display_prediction(image_path, predicted_class_label):
    """
    Displays the original and transformed images
    with the predicted class label.

    Args:
        image_path (str): path to the image file
        predicted_class_label (str): predicted class label
    """

    # Getting original and transformed image
    original_img = cv2.imread(image_path)

    transformation = Transformation(image_path, "mask", None)
    transformation.transformations()

    transformed_img = transformation._mask

    # Compute canvas height
    canvas_width = original_img.shape[1] * 2 + 75
    canvas_height = (
        max(original_img.shape[0], transformed_img.shape[0]) + 200
    )

    # Initialize canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Defining padding
    padding = 25
    padding_text = 45

    # Computing images position
    original_img_start_x = padding
    original_img_end_x = original_img.shape[1] + padding
    original_img_start_y = padding
    original_img_end_y = original_img_start_y + original_img.shape[0]

    transformed_img_start_x = original_img_end_x + padding
    transformed_img_end_x = transformed_img_start_x + transformed_img.shape[1]
    transformed_img_start_y = padding
    transformed_img_end_y = transformed_img_start_y + transformed_img.shape[0]

    canvas[
        original_img_start_y:original_img_end_y,
        original_img_start_x:original_img_end_x
    ] = original_img

    canvas[
        transformed_img_start_y:transformed_img_end_y,
        transformed_img_start_x:transformed_img_end_x,
    ] = transformed_img

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 1

    # Title
    title_border_text = "==="
    title_border_text_size, _ = cv2.getTextSize(
        title_border_text,
        font,
        font_scale,
        font_thickness
    )

    title_text = "DL classification"
    title_text_size, _ = cv2.getTextSize(
        title_text,
        font,
        font_scale,
        font_thickness
    )

    y_title_position = (
        transformed_img_end_y + title_text_size[1] + padding_text
    )

    # Title position
    cv2.putText(
        canvas,
        title_border_text,
        (padding, y_title_position),
        font,
        font_scale,
        font_color,
        font_thickness
    )

    cv2.putText(
        canvas,
        title_text,
        (int((canvas_width / 2) - (title_text_size[0] / 2)), y_title_position),
        font,
        font_scale,
        font_color,
        font_thickness
    )

    cv2.putText(
        canvas,
        title_border_text,
        (canvas_width - title_border_text_size[0] - padding, y_title_position),
        font,
        font_scale,
        font_color,
        font_thickness
    )

    # Display class
    text_size1, _ = cv2.getTextSize(
        "Class predicted : ",
        font,
        font_scale,
        font_thickness
    )
    text_size2, _ = cv2.getTextSize(
        f"Class predicted : {predicted_class_label.lower()}",
        font,
        font_scale,
        font_thickness
    )

    # Computing words position
    x_text_size1_position = int((canvas_width / 2) - (text_size2[0] / 2))
    x_text_size2_position = x_text_size1_position + text_size1[0]

    y_class_label_position = y_title_position + text_size2[1] + padding_text

    cv2.putText(
        canvas,
        "Class predicted : ",
        (x_text_size1_position, y_class_label_position),
        font,
        font_scale,
        font_color,
        font_thickness
    )

    cv2.putText(
        canvas,
        predicted_class_label.lower(),
        (x_text_size2_position, y_class_label_position),
        font,
        font_scale,
        (0, 255, 0),
        font_thickness
    )

    cv2.imshow("Prediction", canvas)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
