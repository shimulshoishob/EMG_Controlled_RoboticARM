"""Where things live. Works from a source checkout on macOS, Windows and Linux."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "arm_config.json"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
FIRMWARE_DIR = PROJECT_ROOT / "firmware" / "EMG_IMU_WIFI_SETUP_V2"


def user_settings_dir():
    """Per-user folder for settings (access key, last device). Never inside the repo."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    folder = base / "EMG Robotic Arm"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
