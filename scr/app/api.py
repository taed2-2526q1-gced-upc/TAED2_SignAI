"""
FastAPI application for serving the SignAI YOLOv8 traffic sign detection model.

This API provides endpoints for:
- Health check (`/`)
- Model information (`/model/info`)
- Prediction (`/predict`)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Import modularized functions
from scr.app.utils import load_model, run_inference

# ----- INITIALIZATION -----

# Add project root to Python path (2 levels up)
sys.path.append(str(Path(__file__).resolve().parents[2]))

# FastAPI instance
app = FastAPI(
    title="SignAI - YOLOv8 API",
    description="API for traffic sign detection using the trained YOLOv8 model.",
    version="1.0.0",
)

# Allow local testing and integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to model
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.pt"

# Load YOLO model (reusable function)
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[WARNING] {e}")


# ----- ENDPOINTS -----

@app.get("/")
def root():
    """Root endpoint to verify that the API is running."""
    return {"message": "SignAI YOLOv8 API is running."}


@app.get("/model/info")
def model_info():
    """Return basic information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded or unavailable.")

    return {
        "model_name": "SignAI - Traffic Sign Detection",
        "classes": model.names,
        "framework": "YOLOv8",
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Perform object detection on an uploaded image using the YOLOv8 model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded or unavailable.")

    try:
        image_bytes = await file.read()
        predictions = run_inference(model, image_bytes)
        return JSONResponse(content={"predictions": predictions})

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

