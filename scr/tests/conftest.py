# scr/tests/conftest.py
import sys
from pathlib import Path

# Add the project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
