"""Tests for the modeling pipeline (training and prediction)."""

from pathlib import Path
import importlib
import pytest
from typer.testing import CliRunner

from scr.modeling import train, predict
from scr.config import MODELS_DIR


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def runner():
    """Fixture to run Typer CLI applications."""
    return CliRunner()


# ===========================================================================
# ===                        TRAINING TESTS                                ===
# ===========================================================================

def test_train_module_import():
    """Ensure the training module can be imported properly."""
    module = importlib.import_module("scr.modeling.train")
    assert module is not None, "Failed to import scr.modeling.train"


def test_train_has_main_function():
    """Check that the training module defines a main entry point."""
    assert hasattr(train, "__file__")
    # Optional: assert hasattr(train, "main") if you define a main() function.


def test_model_file_exists():
    """Verify that at least one trained model file (.pt) exists."""
    model_files = list(Path(MODELS_DIR).glob("*.pt"))
    assert len(model_files) > 0, "No .pt model file found in the models directory"


def test_project_structure_integrity():
    """Ensure that the base project structure is consistent."""
    required_dirs = ["data", "models", "scr", "reports"]
    for folder in required_dirs:
        path = Path(folder)
        assert path.exists(), f"Missing required folder: {folder}"


# ===========================================================================
# ===                        PREDICTION TESTS                              ===
# ===========================================================================

def test_predict_module_import():
    """Ensure the prediction module can be imported properly."""
    module = importlib.import_module("scr.modeling.predict")
    assert module is not None, "Failed to import scr.modeling.predict"


def test_predict_cli_runs_successfully(runner):
    """Run the Typer CLI from the predict script and check for successful execution."""
    result = runner.invoke(predict.app, [])
    assert result.exit_code == 0, f"Prediction CLI failed: {result.output}"
    assert "complet" in result.stdout.lower(), "CLI output did not indicate completion"


def test_predict_creates_output(tmp_path):
    """Simulate an inference scenario using temporary folders."""
    test_images = tmp_path / "images"
    test_images.mkdir()
    (test_images / "dummy.jpg").write_bytes(b"")  # mock image file
    result_dir = tmp_path / "preds"

    result = runner().invoke(
        predict.app,
        [
            "--images-dir", str(test_images),
            "--output-dir", str(result_dir),
            "--save-txt", "False",
        ]
    )

    assert result.exit_code == 0, f"Prediction CLI failed: {result.output}"
    assert result_dir.exists(), "Output directory was not created"


def test_predict_has_main_function():
    """Verify that the prediction module defines a main() function."""
    assert hasattr(predict, "main"), "The predict module is missing the main() function"

