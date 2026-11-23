"""Launcher:  python Model_Training/app/main.py [dataset.csv]"""

import sys
from pathlib import Path

# The existing code uses package-relative imports (``from ..utils import ...``), so the folder that
# CONTAINS ``Model_Training`` must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    from Model_Training.app.gui import run
    sys.exit(run())
