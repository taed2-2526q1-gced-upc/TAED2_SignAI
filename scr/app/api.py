"""
FastAPI application for serving the YOLOv8 traffic sign detection model.

This API provides an endpoint that receives an image and returns
the model’s predictions, including detected classes and bounding boxes.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io

# ----- INITIALIZATION -----

import sys
from pathlib import Path

# Add project root to Python path (2 levels up from this file)
sys.path.append(str(Path(__file__).resolve().parents[2]))



app = FastAPI(
    title="SignAI - YOLOv8 API",
    description="API for traffic sign detection using the trained YOLOv8 model.",
    version="1.0.0",
)

# Allow local testing and integration from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the model (adjust to your project structure)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"

# Load YOLO model once when the API starts
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")


# ----- ENDPOINTS -----

@app.get("/")
def root():
    """Root endpoint to verify that the API is running."""
    return {"message": "SignAI YOLOv8 API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Perform object detection on an uploaded image using the YOLOv8 model.

    Parameters
    ----------
    file : UploadFile
        The image file uploaded by the user.

    Returns
    -------
    JSON
        A list of detected objects, each with class name, confidence, and bounding box coordinates.
    """
    try:
        # Read and open the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run prediction
        results = model.predict(source=image, save=False, verbose=False, device="cpu")

        # Parse predictions
        predictions = []
        for box in results[0].boxes:
            predictions.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()],
            })

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/model/info")
def model_info():
    """
    Return basic information about the loaded model.
    """
    return {
        "model_name": str(MODEL_PATH.name),
        "classes": model.names,
        "version": "YOLOv8",
    }
