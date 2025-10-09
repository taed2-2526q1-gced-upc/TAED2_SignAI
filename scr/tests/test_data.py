# Test the data with deepchecks train_test_validation suite
from deepchecks.tabular.suites import train_test_validation
from deepchecks import VisionData
import pandas as pd
import os
import matplotlib.pyplot as plt

def test_data():
    # Load the image (PNG) and label data (txt)
    train_image_path = 'scr/data/raw/train/images/' 
    train_label_path = 'scr/data/raw/train/labels/'
    test_image_path = 'scr/data/raw/test/images/' 
    test_label_path = 'scr/data/raw/test/labels/'
    train_image_paths = [os.path.join(train_image_path, fname) for fname in os.listdir(train_image_path)]
    test_image_paths = [os.path.join(test_image_path, fname) for fname in os.listdir(test_image_path)]

    train_images = []
    train_labels = []
    for img_path in train_image_paths:
        img = plt.imread(img_path)
        train_images.append(img)
        label_file = os.path.join(train_label_path, os.path.basename(img_path).replace('.png', '.txt'))
        with open(label_file, 'r') as f:
            #Grab first number in the txt file as label
            label = f.read().strip().split()[0]
            train_labels.append(label)

    test_images = []
    test_labels = []
    for img_path in test_image_paths:
        img = plt.imread(img_path)
        test_images.append(img)
        label_file = os.path.join(test_label_path, os.path.basename(img_path).replace('.png', '.txt'))
        with open(label_file, 'r') as f:
            #Grab first number in the txt file as label
            label = f.read().strip().split()[0]
            test_labels.append(label)

    # Create a Deepchecks Dataset
    train_ds = VisionData(train_images, label=train_labels)
    test_ds = VisionData(test_images, label=test_labels)

    # Run the train_test_validation suite
    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)

    result.save_as_html('deepchecks_test.html')

    assert result.passed, "Data validation failed. Check the report for details."