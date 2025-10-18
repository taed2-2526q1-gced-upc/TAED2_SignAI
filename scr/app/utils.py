"""
Utility functions for the SignAI YOLOv8 model.

This module handles:
- Loading the YOLO model.
- Running inference on images.
- Parsing model outputs into JSON-serializable format.
"""

from ultralytics import YOLO
from PIL import Image
import io


def load_model(model_path):
    """
    Load a YOLO model from the given path.

    Parameters
    ----------
    model_path : str or Path
        Path to the YOLO model (.pt file).

    Returns
    -------
    YOLO
        The loaded YOLO model instance.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")


def run_inference(model, image_bytes):
    """
    Run object detection on an image using a YOLO model.

    Parameters
    ----------
    model : YOLO
        The YOLO model instance to use for prediction.
    image_bytes : bytes
        The image data (as bytes).

    Returns
    -------
    list
        List of predictions, each containing:
        - class (str)
        - confidence (float)
        - bbox (list of floats)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    try:
        results = model.predict(source=image, save=False, verbose=False, device="cpu")

        predictions = [
            {
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
            }
            for box in results[0].boxes
        ]
        return predictions
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
