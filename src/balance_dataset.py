import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from Distribution import count_jpg_files

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
            break
    
    for root, dirs, files in os.walk("./datasets/augmented_directory/images/"):
        if len(dirs) == 0:
            count_jpg_files(category_counts, root)

    print(category_counts)