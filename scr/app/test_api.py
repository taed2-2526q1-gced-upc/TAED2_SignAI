"""
Tests for the SignAI FastAPI API (YOLOv8 traffic sign detection).
Compatible with modular API (utils.py + api.py separation).
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from scr.app.api import app
from scr.app.utils import load_model  # ✅ nou import

client = TestClient(app)


# ---------- BASIC ENDPOINT TESTS ----------

def test_root_endpoint():
    """Check that the API root is reachable."""
    response = client.get("/")
    assert response.status_code == 200
    assert "SignAI" in response.json()["message"]


# ---------- PREDICTION TESTS ----------

def test_predict_valid_image():
    """Upload a valid image and expect a 200 response."""
    image = Image.new("RGB", (64, 64), color="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    files = {"file": ("test.jpg", buffer, "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_predict_invalid_file():
    """Send a non-image file and check that the API returns a 400 error."""
    files = {"file": ("fake.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400  # ✅ actualitzat
    assert "Invalid image file" in response.json()["detail"]


# ---------- FAILURE / EDGE CASE TESTS ----------

@pytest.fixture
def disable_model(monkeypatch):
    """Temporarily disable the model to simulate a missing model."""
    from scr.app import api
    original_model = api.model
    monkeypatch.setattr(api, "model", None)
    yield
    # Restore using utils.load_model()
    api.model = load_model(api.MODEL_PATH)


def test_model_not_loaded(disable_model):
    """Simulate missing model and expect a 500 error."""
    response = client.get("/model/info")
    assert response.status_code == 500
    assert "not loaded" in response.json()["detail"].lower()
