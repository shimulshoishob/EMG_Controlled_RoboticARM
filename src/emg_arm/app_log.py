"""Quiet background log (``logs/app.log``) for support. Nothing here is shown in the application window."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .paths import LOG_DIR

_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
_log = logging.getLogger("emg_arm")


def record(level: str, source: str, message: str) -> None:
    """Write one line to the log file. Safe to call from any thread."""
    _log.log(_LEVELS.get(level, logging.INFO), "[%s] %s", source, message)


def setup_file_logging(folder: Path = LOG_DIR) -> Optional[Path]:
    """Rotating app.log (1 MB x 3). Failure to create it must never stop the application."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "app.log"
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        _log.setLevel(logging.INFO)
        if not any(isinstance(h, RotatingFileHandler) for h in _log.handlers):
            _log.addHandler(handler)
        return path
    except OSError:
        return None
