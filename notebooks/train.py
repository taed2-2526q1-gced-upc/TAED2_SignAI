# pip install ultralytics mlflow codecarbon python-dotenv dagshub
import os
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from codecarbon import EmissionsTracker
import mlflow

# Si usas .env (opcional)
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

# Asegura que apuntas a DagsHub (o deja que lo tome de las env vars)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("traffic-signs-yolov8")

DATA_YAML = "data/dataset.yaml"   # ajusta la ruta
MODEL = "yolov8n.pt"
IMGSZ = 250
EPOCHS = 50
BATCH = 16
DEVICE = 0
RUN_NAME = f"yolov8-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

model = YOLO(MODEL)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("model", MODEL)
    mlflow.log_param("imgsz", IMGSZ)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch", BATCH)
    mlflow.log_param("data_yaml", DATA_YAML)

    tracker = EmissionsTracker(project_name="traffic-signs-yolov8",
                               output_dir="codecarbon_out", save_to_file=True)
    tracker.start()

    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project="runs",
        name=RUN_NAME,
        exist_ok=True
    )

    emissions = tracker.stop()
    mlflow.log_metric("emissions_kg", float(emissions))

    # Sube artefactos claves a MLflow/DagsHub
    run_dir = Path("runs") / "detect" / RUN_NAME
    for p in [
        run_dir / "results.csv",
        run_dir / "results.yaml",
        run_dir / "weights" / "best.pt",
    ]:
        if p.exists():
            mlflow.log_artifact(str(p))

    # (Opcional) parsear y loguear métricas desde results.csv
    try:
        import pandas as pd
        last = pd.read_csv(run_dir / "results.csv").iloc[-1]
        for k in ["metrics/precision(B)", "metrics/recall(B)",
                  "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if k in last:
                mlflow.log_metric(k.replace("(B)", ""), float(last[k]))
    except Exception as e:
        print("No se pudieron parsear métricas:", e)

