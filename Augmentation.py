import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def flip(image):
    assert image is not None, "File could not be read"
    flipped_image = cv2.flip(image, 0)
    return flipped_image

def rotation(image):
    assert image is not None, "File could not be read"
    angle = np.random.randint(0, 360)
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((int(width/2), int(height/2)), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT)
    return rotated_image

def brightness(image):
    assert image is not None, "File could not be read"
    brightness_factor = np.random.uniform(1.5, 2.5)
    brightened_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    return brightened_image

def saturation(image):
    assert image is not None, "File could not be read"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_factor = np.random.uniform(1., 4.)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    saturated_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return saturated_image

def crop(image):
    assert image is not None, "File could not be read"
    rows, cols, _ = image.shape
    crop_percent = 0.2
    crop_x = np.random.randint(0, int(cols * crop_percent))
    crop_y = np.random.randint(0, int(rows * crop_percent))
    cropped_image = image[crop_y:rows-crop_y, crop_x:cols-crop_x]
    return cropped_image

def distortion(image):
    assert image is not None, "File could not be read"
    rows, cols, _ = image.shape
    distort_factor = np.random.uniform(0.5, 1.5)
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32([[0, 0], [int(cols * 0.9), int(rows * 0.1)], [int(cols * 0.1), int(rows * 0.9)], [int(cols * 0.9), int(rows * 0.9)]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    distorted_image = cv2.warpPerspective(image, matrix, (cols, rows))
    return distorted_image

def plot_data_augmentation(augmented_images: dict):
    plt.figure(figsize=(18, 3))
    for i, (title, img) in enumerate(augmented_images.items()):
        plt.subplot(1, 7, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()    

def main():
    try:
        assert len(sys.argv) == 2, "Only one argument required."

        image = cv2.imread(sys.argv[1])

        augmented_images = {
            'Original': image,
            'Flip': flip(image),
            'Rotate': rotation(image),
            'Crop': crop(image),
            'Distortion': distortion(image),
            'Brightness': brightness(image),
            'Saturation': saturation(image)
        }

        plot_data_augmentation(augmented_images)

    except Exception as error:
        print(f"error: {error}")


if __name__ == "__main__":
    main()