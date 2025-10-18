"""
Unit tests for scr/app/utils.py
These tests validate the reusable functions independently from the API.
"""

import io
from PIL import Image
import pytest
from scr.app.utils import load_model, run_inference
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"


def test_load_model_success():
    """Test that the YOLO model loads correctly."""
    model = load_model(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")


def test_run_inference_with_valid_image():
    """Test inference function using a simple in-memory image."""
    model = load_model(MODEL_PATH)
    image = Image.new("RGB", (32, 32), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    results = run_inference(model, buffer.getvalue())
    assert isinstance(results, list)


def test_run_inference_invalid_image():
    """Ensure invalid images raise ValueError."""
    model = load_model(MODEL_PATH)
    with pytest.raises(ValueError):
        run_inference(model, b"not an image")
