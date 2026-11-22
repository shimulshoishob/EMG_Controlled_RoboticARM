"""In-memory event log for the Diagnostics page, mirrored to the standard ``logging`` module.

Thread-safe: links and workers may call ``add`` from any thread; the GUI polls ``since``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Deque, List, Optional

from .paths import LOG_DIR

LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
_log = logging.getLogger("emg_arm")


@dataclass(frozen=True)
class LogEntry:
    index: int
    wall_time: float
    level: str        # info | warning | error
    source: str       # link | arm | gesture | keyboard | app
    message: str

    @property
    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.wall_time))


class DiagnosticsLog:
    def __init__(self, capacity: int = 1000):
        self._entries: Deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._counter = 0
        self.counts = {"info": 0, "warning": 0, "error": 0}

    def add(self, level: str, source: str, message: str) -> LogEntry:
        level = level if level in LEVELS else "info"
        with self._lock:
            self._counter += 1
            entry = LogEntry(self._counter, time.time(), level, source, message)
            self._entries.append(entry)
            self.counts[level] += 1
        _log.log(LEVELS[level], "[%s] %s", source, message)
        return entry

    def since(self, index: int) -> List[LogEntry]:
        with self._lock:
            return [e for e in self._entries if e.index > index]

    def recent(self, n: int = 8, min_level: str = "info") -> List[LogEntry]:
        order = ["info", "warning", "error"]
        with self._lock:
            items = [e for e in self._entries if order.index(e.level) >= order.index(min_level)]
        return items[-n:]

    def errors(self) -> List[LogEntry]:
        with self._lock:
            return [e for e in self._entries if e.level == "error"]

    def text(self) -> str:
        with self._lock:
            return "\n".join(f"{e.clock}  {e.level.upper():7s} {e.source:9s} {e.message}" for e in self._entries)


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
