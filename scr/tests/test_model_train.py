"""
Fast tests for the YOLOv8 training pipeline using the 4-image subset.
"""

from pathlib import Path
import pytest #type: ignore
import torch #type: ignore
from scr.config import DATA_DIR, MODELS_DIR, REPORTS_DIR

from scr.modeling import train


# ----- FIXTURES -----

@pytest.fixture(scope="session")
def yolo_model():
    """Load YOLO only when needed to avoid slow collection."""
    from ultralytics import YOLO # type: ignore
    model_path = MODELS_DIR / "yolov8n.pt"
    assert model_path.exists(), f"Model weights not found at {model_path}"
    return YOLO(model_path)


# ----- STRUCTURE AND PATH TESTS -----

def test_project_structure_integrity():
    """Ensure that essential project directories exist."""
    required_dirs = [DATA_DIR, MODELS_DIR, REPORTS_DIR]
    for path in required_dirs:
        assert path.exists(), f"Missing required folder: {path}"



def test_training_module_exists():
    """Check that the training script file exists within the modeling directory."""
    train_file = Path(__file__).resolve().parents[1] / "modeling" / "train.py"
    assert train_file.exists(), f"The training script file does not exist at {train_file}"



# ----- YOLO MODEL TEST -----

def test_yolo_model_loads_successfully(yolo_model):
    """Ensure that the YOLO model loads correctly from weights."""
    assert yolo_model is not None, "Failed to load YOLO model."
    assert hasattr(yolo_model, "train"), "Loaded model lacks the 'train' method."


def test_yolo_forward_pass(yolo_model):
    """Perform a dummy forward pass to confirm the model produces output."""
    dummy_input = torch.randn(1, 3, 64, 64)
    output = yolo_model(dummy_input)
    assert output is not None, "Model output is None."
    assert len(output) > 0, "Model output is empty."


# ----- SIMULATED TRAINING TEST -----

@pytest.mark.slow
def test_simulated_training_run(tmp_path, yolo_model):
    """
    Simulate a minimal YOLO training run using 4-image subset.
    Creates a temporary mini YAML to avoid large datasets.
    """
    test_images = DATA_DIR / "test_sample" / "images"
    test_labels = DATA_DIR / "test_sample" / "labels"
    assert test_images.exists()
    assert test_labels.exists()

    # Create a lightweight temporary YAML
    temp_yaml = tmp_path / "mini_data.yaml"
    temp_yaml.write_text(
        f"train: {test_images}\n"
        f"val: {test_images}\n"
        "nc: 4\n"
        "names: ['prohibition', 'danger', 'mandatory', 'other']\n"
    )

    output_dir = tmp_path / "mini_run"

    try:
        #Directly calling the function created in train.py
        model, results = train(
            DATA_YAML =  "data/data_test.yaml",
            IMGSZ = 64,
            EPOCHS = 1,
            BATCH = 2,
            OPTIMIZER = "Adam",
            LR = 0.001,
            DEVICE = "cpu",
        )
        assert results is not None, "Training did not return results."
        assert model is not None, "Training did not return a model."
        assert output_dir.exists(), "Output directory was not created."
    except Exception as e:
        pytest.fail(f"Simulated training failed: {e}")
