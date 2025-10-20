"""ProPainter packaged interface."""
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_ROOT / "results"
WEIGHTS_DIR = PACKAGE_ROOT / "weights"

if str(PACKAGE_ROOT) not in sys.path:
	sys.path.insert(0, str(PACKAGE_ROOT))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
