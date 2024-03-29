import os
import shutil
from sklearn.model_selection import train_test_split


def count_files(classes, subclasses, dataset_dir):
    """
    Count files for each subclass in a directory

    Args:
        classes (list): list of class labels
        subclasses (list): list of subclasses labels
        dataset_dir (str): dataset path

    Returns:
        dict: number of files in each subclass
    """
    count = {}
    for cls in classes:
        count[cls] = {}
        for subclass in subclasses[classes.index(cls)]:
            count[cls][subclass] = len(
                os.listdir(os.path.join(dataset_dir, cls, subclass))
            )
    return count


def split_dataset(dataset_dir):
    """
    Splits a dataset into training and test sets
    while maintaining the directory structure.

    Args:
        dataset_dir (str): Path to the original dataset directory.
    """
    # get list of classes
    classes = os.listdir(dataset_dir)
    classes = [cls for cls in classes
               if os.path.isdir(os.path.join(dataset_dir, cls))]

    # get list of subclasses
    subclasses_list = []
    for cls in classes:
        subclasses = [
            name
            for name in os.listdir(f"{dataset_dir}/{cls}")
            if os.path.isdir(os.path.join(f"{dataset_dir}/{cls}", name))
        ]
        subclasses_list.append(subclasses)

    # create path
    dataset_dir = os.path.abspath(dataset_dir)
    train_dir = os.path.join(dataset_dir, "training_set")
    test_dir = os.path.join(dataset_dir, "test_set")

    # create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(
        f"before split: {count_files(classes, subclasses_list, dataset_dir)}"
        )

    i = 0
    for cls in classes:
        train_class_dir = os.path.join(train_dir, cls)
        test_class_dir = os.path.join(test_dir, cls)

        for subclass in subclasses_list[i]:
            # create directories
            os.makedirs(f"{train_class_dir}/{subclass}", exist_ok=True)
            os.makedirs(f"{test_class_dir}/{subclass}", exist_ok=True)

            # create path for subclass
            subclass_dir = os.path.abspath(dataset_dir)
            subclass_dir = f"{subclass_dir}/{cls}/{subclass}"

            # get images dataset to split
            images = os.listdir(subclass_dir)
            images = [
                img for img in images
                if os.path.isfile(os.path.join(subclass_dir, img))
            ]

            # split dataset
            train_images, test_images = train_test_split(
                images, test_size=0.2, random_state=42
            )

            # move training dataset to training_set directory
            for img in train_images:
                src = os.path.join(subclass_dir, img)
                dst = os.path.join(f"{train_class_dir}/{subclass}", img)
                shutil.move(src, dst)

            # move test dataset to test_set directory
            for img in test_images:
                src = os.path.join(subclass_dir, img)
                dst = os.path.join(f"{test_class_dir}/{subclass}", img)
                shutil.move(src, dst)

        i += 1

    # delete old directories
    shutil.rmtree(f"{dataset_dir}/Apple")
    shutil.rmtree(f"{dataset_dir}/Grape")

    print(
        f"Training set: {count_files(classes, subclasses_list, train_dir)}"
        )

    print(
        f"Test set: {count_files(classes, subclasses_list, test_dir)}"
        )
