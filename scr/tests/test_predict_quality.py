"""
Asserts quality in input and output format of data and accuracy of the model at inference
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scr.config import MODELS_DIR, DATA_DIR

import pytest #type: ignore
from scr.modeling.predict import predict #type: ignore
from PIL import Image #type: ignore

def test_input_image_size(input_images_path):
    """Check input is an image of size < 1024x1024 pixels"""
    for image_path in os.listdir(input_images_path):
        if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            image_file = os.path.join(input_images_path, image_path)
            image = Image.open(image_file)
            width, height = image.size
            assert width <= 1024 and height <= 1024, f"Input image {image_path} exceeds size limit of 1024x1024 pixels"

def test_output_format(input_images_path: Path, output_path: Path):
    """Check output format is as expected (the label (0-3) followed by bounding box coordinates)"""
    prediction = predict(
        images_dir = Path(input_images_path),
        output_dir = Path(output_path),
        save_txt_on_repo = True)
    #Get image name from image
    for label_file in os.listdir(Path(output_path / "predict" / "labels")):
        if label_file.lower().endswith((".txt")):
            with open(os.path.join(output_path, label_file), 'r') as f:
                content = f.read().strip()
                assert len(content.split()) >= 5, f"Output file {label_file} does not have the expected format"
                assert content.split()[0] in ['0', '1', '2', '3'], f"Output label {content.split()[0]} in file {label_file} is not a valid class"

def test_model_metrics(input_images_path: Path, output_path: Path, IoU_threshold=0.5):
    """Check that overall mAP@IoU_threshold is above threshold (80%) and difference between classes is not too high (10%)"""
    TP_0, TN_0, FP_0, FN_0 = 0, 0, 0, 0
    TP_1, TN_1, FP_1, FN_1 = 0, 0, 0, 0
    TP_2, TN_2, FP_2, FN_2 = 0, 0, 0, 0
    TP_3, TN_3, FP_3, FN_3 = 0, 0, 0, 0
    #Test mAP from saved output files for test images
    for label_file in os.listdir(Path(output_path / "predict" / "labels")):
        with open(os.path.join(output_path, label_file), 'r') as f:
            with open(os.path.join(input_images_path, label_file), 'r') as f_true:
                content = f.read().strip()
                content_true = f_true.read().strip()
                IoU = iou(list(map(float, content.split()[1:5])), list(map(float, content_true.split()[1:5])))
                if content[0] == content_true[0] and IoU >= IoU_threshold:
                    if content_true[0] == "0": TP_0 += 1
                    if content_true[0] == "1": TP_1 += 1
                    if content_true[0] == "2": TP_2 += 1
                    if content_true[0] == "3": TP_3 += 1
                elif content[0] != content_true[0] and IoU >= IoU_threshold:
                    if content_true[0] == "0": FP_0 += 1
                    if content_true[0] == "1": FP_1 += 1
                    if content_true[0] == "2": FP_2 += 1
                    if content_true[0] == "3": FP_3 += 1
                elif content[0] == content_true[0] and IoU < IoU_threshold:
                    if content_true[0] == "0": FN_0 += 1
                    if content_true[0] == "1": FN_1 += 1
                    if content_true[0] == "2": FN_2 += 1
                    if content_true[0] == "3": FN_3 += 1
                else:
                    if content_true[0] == 0: TN_0 += 1
                    if content_true[0] == 1: TN_1 += 1
                    if content_true[0] == 2: TN_2 += 1
                    if content_true[0] == 3: TN_3 += 1

    #Calculate mAP for each class
    mAP_0 = calculate_map(TP_0, FP_0, FN_0)
    mAP_1 = calculate_map(TP_1, FP_1, FN_1)
    mAP_2 = calculate_map(TP_2, FP_2, FN_2)
    mAP_3 = calculate_map(TP_3, FP_3, FN_3)
    #Calculate overall mAP       
    mAP = calculate_map(TP_0+TP_1+TP_2+TP_3, FP_0+FP_1+FP_2+FP_3, FN_0+FN_1+FN_2+FN_3)
    assert mAP >= 0.8, f"Model mAP: {mAP} is below acceptable threshold of 0.8"
    #Test difference in accuracy between classes from saved output files for test images
    assert abs(mAP_0 - mAP_1) <= 0.1, f"Difference in mAP between class 0 and 1 is too high: |{mAP_0} - {mAP_1}| = {abs(mAP_0 - mAP_1)}"
    assert abs(mAP_0 - mAP_2) <= 0.1, f"Difference in mAP between class 0 and 2 is too high: |{mAP_0} - {mAP_2}| = {abs(mAP_0 - mAP_2)}"
    assert abs(mAP_0 - mAP_3) <= 0.1, f"Difference in mAP between class 0 and 3 is too high: |{mAP_0} - {mAP_3}| = {abs(mAP_0 - mAP_3)}"    
    assert abs(mAP_1 - mAP_2) <= 0.1, f"Difference in mAP between class 1 and 2 is too high: |{mAP_1} - {mAP_2}| = {abs(mAP_1 - mAP_2)}"
    assert abs(mAP_1 - mAP_3) <= 0.1, f"Difference in mAP between class 1 and 3 is too high: |{mAP_1} - {mAP_3}| = {abs(mAP_1 - mAP_3)}"
    assert abs(mAP_2 - mAP_3) <= 0.1, f"Difference in mAP between class 2 and 3 is too high: |{mAP_2} - {mAP_3}| = {abs(mAP_2 - mAP_3)}"

def calculate_map(TP, FP, FN):
        if TP + FP == 0 or TP + FN == 0:
            return 0.0
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

def iou(box1, box2):
    """Compute IoU between two boxes in YOLO normalized format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corners
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2

    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

# Run tests
test_input_image_size(Path(DATA_DIR / 'processed' / 'test'))
test_output_format(Path(DATA_DIR / 'processed' / 'test'), Path(DATA_DIR / 'processed' / 'test' ))
test_model_metrics(Path(DATA_DIR / 'processed' / 'test'), Path(DATA_DIR / 'processed' / 'test' ))

