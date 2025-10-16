"""
Tests for the YOLOv8 prediction pipeline in the SignAI project.

These tests verify that the prediction script runs correctly,
produces valid outputs, and handles input images as expected.
"""


# ----- IMPORTS -----

from pathlib import Path
import pytest #type: ignore
from typer.testing import CliRunner #type: ignore
from ultralytics import YOLO #type: ignore

from scr.modeling import predict
from scr.config import DATA_DIR, MODELS_DIR
from src.modeling import predict # type: ignore


# ----- FIXTURES -----

@pytest.fixture(scope="session")
def runner():
    """Fixture to run the Typer CLI app."""
    return CliRunner()


@pytest.fixture(scope="session")
def yolo_model():
    """Fixture to load the YOLOv8 model for inference."""
    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        # Fallback to the default YOLO model if no trained weights exist
        model_path = MODELS_DIR / "yolov8n.pt"
    assert model_path.exists(), f"Model weights not found at {model_path}"
    return YOLO(model_path)


# ----- MODEL LOOAD TEST -----


def test_yolo_model_loads(yolo_model):
    """Ensure the YOLO model loads correctly and is ready for prediction."""
    assert yolo_model is not None, "Failed to load YOLO model."
    assert hasattr(yolo_model, "predict"), "Model object does not have a predict() method."


# ----- CLI EXECUTION TEST -----

def test_predict_cli_runs_successfully(runner, tmp_path):
    """
    Run the Typer CLI defined in predict.py to ensure it executes without errors.
    """
    output_dir = tmp_path / "predictions"

    result = runner.invoke(
        predict.app,
        [
            "--images-dir", str(DATA_DIR / "test_sample" / "images"),
            "--output-dir", str(output_dir),
            "--save-txt",
            "--save-conf",
        ],
    )

    # Check that the CLI exits cleanly
    assert result.exit_code == 0, f"Prediction CLI failed: {result.output}"
    # Check that the output directory was created
    assert output_dir.exists(), "Prediction output directory was not created."


# ----- OUTPUT VALIDATION TEST -----

def test_predictions_generate_output_files(yolo_model, tmp_path):
    """
    Ensure that predictions produce output files in the directory.
    Even if no labels are generated, the run should complete successfully.
    """
    test_images_dir = DATA_DIR / "test_sample"
    assert test_images_dir.exists(), "Test images directory not found."

    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = predict(
        images_dir = test_images_dir,
        output_dir = output_dir,
        save_txt=True,
        save_conf=False,
    )

    assert results is not None, "No results returned from model prediction."

    # Check that the labels folder was created, even if empty
    labels_dir = output_dir / "predict" / "labels"
    assert labels_dir.exists(), "Labels folder was not created."


# ----- FILE STRUCTURE TEST -----

def test_project_structure_for_prediction():
    """Ensure that all required directories for prediction exist."""
    required_dirs = [
        DATA_DIR / "test_sample" / "images",
        MODELS_DIR,
        Path(__file__).resolve().parents[1] / "modeling",
    ]

    for path in required_dirs:
        assert path.exists(), f"Missing required folder for prediction: {path}"
