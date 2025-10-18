"""Performs several behavioural tests on the model"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scr.config import DATA_DIR
from scr.modeling.predict import predict #type: ignore
from scr.tests.test_predict_quality import iou  # Reuse IoU function from quality tests


# Direct testing behavioural tests on the model


#Using a subset of images test for different modifications
def test_model_behaviour(path_to_test_images, img_id):
    """Test model behaviour on modified images compared to original images"""
    # Original images
    prediction_original = predict(
        images_dir = Path(path_to_test_images / img_id),
        output_dir = Path(path_to_test_images / img_id))
    
    # Modified images with occlusions, colour filters, noise (snow and rain), perspective distortions (Pov)
    prediction_occluded = predict(
        images_dir = Path(path_to_test_images / img_id / "modified"),
        output_dir = Path(path_to_test_images / img_id / "modified"))
    
    # Compare predictions
    with open(os.path.join(path_to_test_images/ img_id / "predict" / "labels", f"{img_id}.txt"), 'r') as f:
        original_content = f.read().strip()
    for mod_label_file in os.listdir(Path(path_to_test_images / img_id / "modified" / "predict" / "labels")):
        with open(os.path.join(path_to_test_images / img_id / "modified" / "predict" / "labels", mod_label_file), 'r') as f:
            modified_content = f.read().strip()
            
            if original_content.split()[0] != modified_content.split()[0]: 
                print(f"Label prediction changed for modified image {mod_label_file}")
                print(f"Original label: {original_content.split()[0]}, Modified label: {modified_content.split()[0]}")
            if iou(list(map(float, original_content.split()[1:5])), list(map(float, modified_content.split()[1:5]))) < 0.5:
                print(f"Significant bounding box change for modified image {mod_label_file}")
                print(f"IoU: {iou(list(map(float, original_content.split()[1:5])), list(map(float, modified_content.split()[1:5])))}")


# Run behavioural test on a two randomly selected test images
print("-----")
test_model_behaviour(Path(DATA_DIR / 'behavioural_testing'), img_id="00001")
print("-----")
print("Finished behavioural test on image 00001")
print("-----")
test_model_behaviour(Path(DATA_DIR / 'behavioural_testing'), img_id="07836")
print("-----")