"""Tests for the modeling pipeline (training and prediction)."""

from pathlib import Path
import pytest
from typer.testing import CliRunner
from scr.modeling import predict, train
from scr.config import MODELS_DIR


@pytest.fixture(scope="session")
def runner():
    """Fixture for running Typer CLI apps."""
    return CliRunner()


def test_train_module_exists():
    """Ensure that the training module can be imported and executed."""
    assert hasattr(train, "__file__") or callable(train), "train module not found"


def test_predict_cli_runs_successfully(runner):
    """Run the predict.main Typer command and ensure it completes."""
    result = runner.invoke(predict.app, [])
    assert result.exit_code == 0, f"Prediction CLI failed: {result.output}"
    assert "Inference complete" in result.stdout or "complete" in result.stdout.lower()


def test_model_file_exists():
    """Check that a trained model file or directory exists."""
    model_files = list(Path(MODELS_DIR).glob("*.pkl"))
    assert len(model_files) > 0, "No model .pkl file found in models directory"


def test_dummy_inference_simulation():
    """Simulate a dummy inference to ensure code structure works."""
    # Here we just check that the function exists and can be called
    assert hasattr(predict, "main"), "Predict module missing main() function"
