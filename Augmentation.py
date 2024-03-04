import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from Distribution import count_jpg_files

JPEG_SIGNATURE = b"\xFF\xD8"


def flip(image, axis=0):
    """
    Flips an image along a specified axis.

    Args:
        image ('numpy.ndarray'): input image to transform
        axis (int, optional): the axis along which to flip the image.
        - 0: horizontally
        - 1: vertically
        Defaults to 0.

    Returns:
        'numpy.ndarray': the flipped image
    """
    assert image is not None, "File could not be read"
    flipped_image = cv2.flip(image, axis)
    return flipped_image


def rotation(image):
    """
    Rotates an image by a random angle

    Args:
        image ('numpy.ndarray'): input image to transform

    Returns:
        numpy.ndarray: the rotated image
    """
    assert image is not None, "File could not be read"

    height, width = image.shape[:2]
    angle = np.random.randint(0, 360)

    rotation_matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2), angle, 1)

    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT
    )
    return rotated_image


def brightness(image):
    """
    Modifies brightness of the image by a random factor

    Args:
        image ('numpy.ndarray'): input image to transform

    Returns:
        numpy.ndarray: the brightened image
    """
    assert image is not None, "File could not be read"

    brightness_factor = np.random.uniform(1.5, 2.5)
    brightened_image = np.clip(
        image * brightness_factor, 0, 255).astype(np.uint8)

    return brightened_image


def saturation(image):
    """
    Modifies saturation of the image by a random factor

    Args:
        image ('numpy.ndarray'): input image to transform

    Returns:
        'numpy.ndarray': the brightened image
    """
    assert image is not None, "File could not be read"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_factor = np.random.uniform(1.0, 4.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    saturated_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return saturated_image


def crop(image):
    """
    Crops a random space from an image and resizes it to its original shape

    Args:
        image ('numpy.ndarray'): input image to transform

    Returns:
        'numpy.ndarray': the cropped image
    """
    assert image is not None, "File could not be read"
    height, width = image.shape[:2]
    start = np.random.randint(0, 100)

    cropped = image[start:height, start:width]
    resized = cv2.resize(
        cropped, (height, width), interpolation=cv2.INTER_AREA)

    return resized


def distortion(image):
    """
    Applies Barrel distorsion to an image

    Args:
        image ('numpy.ndarray'): input image to transform

    Returns:
        'numpy.ndarray': the distorted image
    """
    assert image is not None, "File could not be read"
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    k = 0.00001
    for y in range(height):
        for x in range(width):
            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            r_dist = r * (1 + k * r**2)
            theta = np.arctan2(y - center[1], x - center[0])
            map_x[y, x] = center[0] + r_dist * np.cos(theta)
            map_y[y, x] = center[1] + r_dist * np.sin(theta)
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return distorted


def plot_data_augmentation(augmented_images: dict):
    """
    Plots original image and its augmented versions

    Args:
        augmented_images (dict): dictionary containing
        augmented versions of the original image
    """
    plt.figure(figsize=(18, 3))
    for i, (title, img) in enumerate(augmented_images.items()):
        plt.subplot(1, 7, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def is_jpeg(file_path):
    """
    Verifies if file is .jpg format

    Args:
        file_path (str): path to the file to verify

    Returns:
        bool : true if .jpg format else false
    """
    with open(file_path, "rb") as f:
        header = f.read(2)
        return header == JPEG_SIGNATURE

def augment_category(category, current_count, target_count):
    category_prefix = category.split('_')[0]    
    file_list = os.listdir(f"./datasets/images/{category_prefix}/{category}")
    sorted_files = sorted(file_list, key=lambda x: int(x.split('(')[1].split(')')[0]))
    
    augmentation_functions = {
        "Flip": flip,
        "Rotate": rotation,
        "Crop": crop,
        "Distortion": distortion,
        "Brightness": brightness,
        "Saturation": saturation,
    }

    # number_iterations = target_count % current_count
    path = f"./datasets/images/{category_prefix}/{category}"
    for file in sorted_files:
        if current_count == target_count:
                return
        for augmentation_type, func in augmentation_functions.items():
            if current_count == target_count:
                return
            augmented_image = func(cv2.imread(f"{path}/{file}"))
            filename = file.split(".JPG")[0]
            plt.imsave(f"./datasets/augmented_directory/images/{category_prefix}/{category}/{filename}_{augmentation_type}.JPG", augmented_image)
            current_count += 1




def balance_dataset(dataset_path):
    augmented_directory = "./datasets/augmented_directory"
    if not os.path.exists(augmented_directory):
        src = './datasets/'
        dest = './datasets/augmented_directory'
        destination = shutil.copytree(src, dest)  

    category_counts = {}

    for root, dirs, files in os.walk(dataset_path):
        if len(dirs) == 0:
            count_jpg_files(category_counts, root)
    target_count = max(category_counts.values())
    
    for category, count in category_counts.items():
        if count != target_count:
            print(category, ":", count)
            augment_category(category, count, target_count)
            
    
    for root, dirs, files in os.walk("./datasets/augmented_directory/images/"):
        if len(dirs) == 0:
            count_jpg_files(category_counts, root)

    print(category_counts)

def main():
    try:
        assert len(sys.argv) == 2, "Only one argument required."
        assert os.path.isfile(sys.argv[1]) and is_jpeg(
            sys.argv[1]
        ), "Argument is not a valid .jpg file"
        image_path = sys.argv[1]
        
        balance_dataset('./datasets/images/')

        # image = cv2.imread(image_path)

        # augmented_images = {
        #     "Flip": flip(image),
        #     "Rotate": rotation(image),
        #     "Crop": crop(image),
        #     "Distortion": distortion(image),
        #     "Brightness": brightness(image),
        #     "Saturation": saturation(image),
        # }

        # print(image_path)
        # # save
        # for key, value in augmented_images.items():
        #     plt.imsave(image_path.split(".JPG")[0] + f"_{key}.JPG", value)

        # # display
        # updict = {"Original": image}
        # updict.update(augmented_images)
        # plot_data_augmentation(updict)

    except Exception as error:
        print(f"error: {error}")


if __name__ == "__main__":
    main()
