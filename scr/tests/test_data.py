# Test the data with deepchecks train_test_validation suite
from deepchecks.tabular.suites import train_test_validation
from deepchecks import VisionData
import pandas as pd
import os
import matplotlib.pyplot as plt

def test_data():
    # Load the image (PNG) and label data (txt)
    image_path = 'scr/data/raw/images/' 
    label_path = 'scr/data/raw/labels/'
    image_paths = [os.path.join(image_path, fname) for fname in os.listdir(image_path)]

    images = []
    labels = []
    for img_path in image_paths:
        img = plt.imread(img_path)
        images.append(img)
        label_file = os.path.join(label_path, os.path.basename(img_path).replace('.png', '.txt'))
        with open(label_file, 'r') as f:
            #Grab first number in the txt file as label
            label = f.read().strip().split()[0]
            labels.append(label)

    # Create a Deepchecks Dataset
    dataset = VisionData(images, label=labels)

    # Run the train_test_validation suite
    suite = train_test_validation()
    result = suite.run(dataset)

    result.save_as_html('deepchecks_test.html')

    assert result.passed, "Data validation failed. Check the report for details."