from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

from ultralytics import YOLO

from scr.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # --- RUTAS POR DEFECTO  ---
    images_dir: Path = PROCESSED_DATA_DIR / "test_images",
    model_path: Path = MODELS_DIR / "best.pt",  # Pendiente confirmar mejor modelo
    output_dir: Path = PROCESSED_DATA_DIR / "predictions",
    save_txt: bool = True,
    save_conf: bool = True,
):
    """
    Ejecuta inferencia usando un modelo YOLOv8 entrenado.
    Deja pendiente la selección definitiva del modelo (best.pt o último).
    """
    logger.info(f"Iniciando inferencia con el modelo: {model_path}")

    # Verificaciones
    if not model_path.exists():
        logger.error(f"No se encontró el modelo en {model_path}")
        raise FileNotFoundError(model_path)

    if not images_dir.exists():
        logger.error(f"No se encontraron imágenes en {images_dir}")
        raise FileNotFoundError(images_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Carga el modelo YOLO
    model = YOLO(model_path)
    logger.info("Modelo cargado correctamente.")

    # Lista de imágenes para predecir
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not images:
        logger.warning("No se encontraron imágenes para procesar.")
        return

    logger.info(f"Procesando {len(images)} imágenes...")

    # Inferencia con barra de progreso
    for img_path in tqdm(images, desc="Prediciendo"):
        results = model.predict(
            source=img_path,
            save=True,                # Guarda las imágenes con bounding boxes
            save_txt=save_txt,        # Guarda coordenadas en .txt
            save_conf=save_conf,      # Guarda la confianza
            project=str(output_dir),  # Dónde guardar resultados
            exist_ok=True
        )
        # Puedes acceder a las predicciones con results[0].boxes.data
        logger.debug(f"Procesada: {img_path.name}")

    logger.success(f"Inferencia completada. Resultados en: {output_dir}")

if __name__ == "__main__":
    app()

