# pip install ultralytics mlflow codecarbon python-dotenv dagshub pandas
import os
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from codecarbon import EmissionsTracker
import mlflow

# Si usas .env (opcional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Si no tienes definida la variable en tu entorno, usa MLflow localmente
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("traffic-signs-yolov8")

# Configuración
DATA_YAML = "data/dataset.yaml"  # Asegúrate de tenerlo bien configurado
MODEL = "yolov8n.pt"             # Modelo base pequeño (va mejor en CPU)
IMGSZ = 250
EPOCHS = 10                      # Empieza con pocos epochs (entrenar en CPU es lento)
BATCH = 8                        # Reduce el batch para no saturar la RAM
DEVICE = "cpu"                   # usa CPU, no device=0
RUN_NAME = f"yolov8-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Carga el modelo YOLO
model = YOLO(MODEL)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("model", MODEL)
    mlflow.log_param("imgsz", IMGSZ)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch", BATCH)
    mlflow.log_param("data_yaml", DATA_YAML)

    tracker = EmissionsTracker(
        project_name="traffic-signs-yolov8",
        output_dir="codecarbon_out",
        save_to_file=True
    )
    tracker.start()

    # Entrenamiento
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

    # Guarda artefactos importantes
    run_dir = Path("runs") / "detect" / RUN_NAME
    for p in [
        run_dir / "results.csv",
        run_dir / "results.yaml",
        run_dir / "weights" / "best.pt",
    ]:
        if p.exists():
            mlflow.log_artifact(str(p))

    # parsear métricas del CSV
    try:
        import pandas as pd
        last = pd.read_csv(run_dir / "results.csv").iloc[-1]
        for k in ["metrics/precision(B)", "metrics/recall(B)",
                  "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if k in last:
                mlflow.log_metric(k.replace("(B)", ""), float(last[k]))
    except Exception as e:
        print("No se pudieron parsear métricas:", e)

print("Entrenamiento completado con CPU.")
