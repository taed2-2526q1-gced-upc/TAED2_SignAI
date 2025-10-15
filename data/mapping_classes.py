"""
Maps original classes to the four classes used in the yolov8 model
"""

import os
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

path_to_train_labels = ROOT_DIR / 'data/raw/train/labels'
path_to_test_labels = ROOT_DIR / 'data/raw/test/labels'
path_to_train_imgs = ROOT_DIR / 'data/raw/train/images'
path_to_test_imgs = ROOT_DIR / 'data/raw/test/images'

procesed_path_train = ROOT_DIR / 'data/processed/train'
procesed_path_test = ROOT_DIR / 'data/processed/test'

mapping = {
    "0": "0", "1": "0", "2": "0", "3": "0", "4": "0", "5": "0",
    "6": "0", "7": "0", "8": "0", "9": "0", "10": "0", "11": "0",
    "12": "3", "13": "3", "14": "1", "15": "0", "16": "0", "17": "0",
    "18": "2", "19": "2", "20": "2", "21": "2", "22": "2", "23": "2",
    "24": "2", "25": "2", "26": "2", "27": "2", "28": "2", "29": "2",
    "30": "2", "31": "2", "32": "3", "33": "1", "34": "1", "35": "1",
    "36": "1", "37": "1", "38": "1", "39": "1", "40": "1", "41": "3", "42": "3"
}


def map_label(path_to_labels, processed_path):
    """
    Map original labels to the four YOLOv8 categories and save them
    in the processed directory.
    """
    # Iterate over each label file within the directory
    for label_file in path_to_labels.glob('*.txt'):
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                continue
            parts = content.split()
            original_label = parts[0]

        # Map to new class using the mapping dictionary
        if original_label in mapping:
            new_label = mapping[original_label]
        else:
            print(f"Warning: No mapping found for label {original_label} in file {label_file}")
            continue

        # Create the output file path with the same filename
        output_file = processed_path / label_file.name

        # Write back the new label with the rest of the coordinates
        with open(output_file, 'w') as f:
            if len(parts) > 1:
                f.write(f"{new_label} " + " ".join(parts[1:]) + "\n")
            else:
                f.write(f"{new_label}\n")


# Process training labels
os.makedirs(procesed_path_train, exist_ok=True)
map_label(path_to_train_labels, procesed_path_train)

# Process testing labels
os.makedirs(procesed_path_test, exist_ok=True)
map_label(path_to_test_labels, procesed_path_test)

print("Label mapping completed.")

# Move all images to the processed folder
shutil.copytree(path_to_train_imgs, procesed_path_train, dirs_exist_ok=True)
shutil.copytree(path_to_test_imgs, procesed_path_test, dirs_exist_ok=True)
