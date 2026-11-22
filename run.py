#!/usr/bin/env python3
"""Start the EMG Robotic Arm app:   python3 run.py   (macOS)    py run.py   (Windows)

The first run creates a private Python environment in ./.venv and installs the libraries
(a few minutes, needs internet). Later runs start immediately."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emg_arm import bootstrap  # noqa: E402

bootstrap.ensure()

from emg_arm.app import main  # noqa: E402

if __name__ == "__main__":
    main()
