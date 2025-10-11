#Checks amount of images per class and creates a random subset of the dataset with a balanced number of images per class.
import os
import random
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

processed_path_train = ROOT_DIR / 'data/processed/train'
processed_path_test = ROOT_DIR / 'data/processed/test'

#Classify images by class (read from the label files first word)
def classify_images_by_class(input_dir):
    class_dict = {}
    for label_file in input_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = first_line.split()[0]
                if class_id not in class_dict:
                    class_dict[class_id] = []
                class_dict[class_id].append(label_file)
    return class_dict

# Create a random subset with a balanced number of images per class
def create_random_subset(input_dir, output_dir, num_images):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    class_dict = classify_images_by_class(input_dir)
    print("Class distribution in the original dataset:")
    for class_id, images in class_dict.items():
        print(f"Class {class_id}: {len(images)} images")
        selected_images = random.sample(images, min(num_images, len(images)))
        for img in selected_images:
            shutil.copy(img.with_suffix('.png'), output_dir / img.with_suffix('.png').name)
            shutil.copy(img, output_dir / img.name)

# Define the number of images you want per class
num_images = 500  # Adjust this number as needed

# Create balanced subsets for training and testing
objective_train_dir = ROOT_DIR / f'data/processed_subset/train'
objective_test_dir = ROOT_DIR / f'data/processed_subset/test'

create_random_subset(processed_path_train, objective_train_dir, num_images)
create_random_subset(processed_path_test, objective_test_dir, num_images)
