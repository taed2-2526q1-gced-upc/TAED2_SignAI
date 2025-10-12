# pip install ultralytics mlflow codecarbon python-dotenv dagshub pandas
import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv  # type: ignore
from ultralytics import YOLO  # type: ignore
from codecarbon import EmissionsTracker  # type: ignore
import mlflow

import dagshub  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[2]

dagshub.init(repo_owner='laia.villagrasa', repo_name='TAED2_SignAI', mlflow=True)

print("Loading environment variables...")
try:
    load_dotenv()
except Exception:
    pass

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT_DIR}/mlruns"))
mlflow.set_experiment("traffic-signs-yolov8")

DATA_YAML = ROOT_DIR / "data/data.yaml"
MODEL = ROOT_DIR / "models/yolov8n.pt"
IMGSZ = 250
EPOCHS = 20
BATCH = 8
OPTIMIZER = "Adam"
LR = 0.001
DEVICE = "cpu"
RUN_NAME = f"yolov8-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

print(f"Using model {MODEL} with images of {IMGSZ}x{IMGSZ}, batch {BATCH}, epochs {EPOCHS}")
# Load the YOLO model
model = YOLO(MODEL)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("model", MODEL)
    mlflow.log_param("imgsz", IMGSZ)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch", BATCH)
    mlflow.log_param("data_yaml", DATA_YAML)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("learning_rate", LR)

    tracker = EmissionsTracker(
        project_name="traffic-signs-yolov8",
        output_dir=str(ROOT_DIR / "reports/codecarbon_out"),
        save_to_file=True
    )
    tracker.start()

    # Training
    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        optimizer=OPTIMIZER,
        lr0 = LR,
        project="runs",
        name=RUN_NAME,
        exist_ok=True
    )

    emissions = tracker.stop()
    mlflow.log_metric("emissions_kg", float(emissions))

    # Save important artifacts
    run_dir = ROOT_DIR / "models/runs" / RUN_NAME
    for p in [
        run_dir / "results.csv",
        run_dir / "results.yaml",
        run_dir / "weights" / "best.pt",
    ]:
        if p.exists():
            mlflow.log_artifact(str(p))

    # Parse metrics from the CSV
    try:
        import pandas as pd
        last = pd.read_csv(run_dir / "results.csv").iloc[-1]
        for k in ["metrics/precision(B)", "metrics/recall(B)",
                  "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if k in last:
                mlflow.log_metric(k.replace("(B)", ""), float(last[k]))
    except Exception as e:
        print("Could not parse metrics:", e)

print("Training completed using CPU.")