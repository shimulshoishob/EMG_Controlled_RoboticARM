"""Small per-user settings file (access key, last armband, theme, last model).

Stored as JSON in the user's application-data folder - never inside the project, so the access key
cannot be committed by accident. The file is created readable by the current user only.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .paths import user_settings_dir

DEFAULTS = {
    "access_key": "",
    "device_ip": "",
    "device_id": "",
    "connection": "wifi",          # wifi | usb | virtual
    "usb_port": "",
    "theme": "auto",               # auto | light | dark
    "last_model": "",
    "model_files": [],             # .joblib files the user uploaded to the model list
    "wifi_ssid": "",
    "setup_done": False,
    "control_mode": "hybrid",      # hybrid | gestures | keyboard
    "min_confidence": 0.0,         # 0 = use config/arm_config.json
    "tuning": {},                  # motion-smoothing overrides (see ui/pages/data_page.py)
    "servo_ranges": {},            # {servo id: [min, max]}, only ever narrower than the shipped limits
    "home_overrides": {},          # {servo id: position}
}


class Settings:
    def __init__(self, path=None):
        self.path = path or (user_settings_dir() / "settings.json")
        self.data = dict(DEFAULTS)
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update({k: v for k, v in loaded.items() if k in DEFAULTS})
        except (OSError, ValueError):
            pass

    def get(self, key: str) -> Any:
        return self.data.get(key, DEFAULTS.get(key))

    def set(self, key: str, value: Any):
        if self.data.get(key) == value:
            return
        self.data[key] = value
        self.save()

    def save(self):
        try:
            tmp = str(self.path) + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            pass
