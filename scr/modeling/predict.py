"""
To run prediction, execute this script **from the project root** using:
    python -m scr.modeling.predict
"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from ultralytics import YOLO

from scr.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # --- DEFAULT PATHS ---
    images_dir: Path = PROCESSED_DATA_DIR / "test",
    model_path: Path = MODELS_DIR / "best.pt",  # Pending confirmation of best model
    output_dir: Path = MODELS_DIR / "predictions",
    save_txt: bool = True,
    save_conf: bool = True,
):
    """
    Run inference using a trained YOLOv8 model.
    """
    logger.info(f"Starting inference with model: {model_path}")

    # Validation checks
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(model_path)

    if not images_dir.exists():
        logger.error(f"Image directory not found at {images_dir}")
        raise FileNotFoundError(images_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model = YOLO(model_path)
    logger.info("Model loaded successfully.")

    # List of images to predict
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not images:
        logger.warning("No images found for prediction.")
        return

    logger.info(f"Processing {len(images)} images...")

    # Inference with progress bar
    for img_path in tqdm(images, desc="Predicting"):
        results = model.predict(
            source=img_path,
            save=True,                # Save images with bounding boxes
            save_txt=save_txt,        # Save coordinates to .txt
            save_conf=save_conf,      # Save confidence scores
            project=str(output_dir),  # Directory to store results
            exist_ok=True
        )
        # You can access predictions via results[0].boxes.data
        logger.debug(f"Processed: {img_path.name}")

    logger.success(f"Inference completed. Results saved in: {output_dir}")

if __name__ == "__main__":
    app()
