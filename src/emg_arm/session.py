"""Research logging: the control-session CSV and the raw EMG/IMU recorder."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .paths import LOG_DIR, RECORDINGS_DIR


class SessionLog:
    """One CSV row per prediction / motion / arm event (for the thesis analysis)."""

    FIELDS = ["time_s", "event", "label", "confidence", "gesture", "motion", "source", "note", "targets"]

    def __init__(self, folder: Path = LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._t0 = time.monotonic()

    def write(self, event: str, targets=None, **fields):
        row = {"time_s": f"{time.monotonic() - self._t0:.3f}", "event": event,
               "targets": json.dumps(targets) if targets else ""}
        row.update(fields)
        self._writer.writerow(row)

    def close(self):
        try:
            self._file.close()
        except OSError:
            pass


class RawRecorder:
    """Writes every received frame (8 EMG counts + roll/pitch/yaw) with its packet sequence."""

    HEADER = ["host_time_s", "packet_seq", "frame_id"] + [f"emg{i}" for i in range(1, 9)] + ["roll", "pitch", "yaw"]

    def __init__(self, folder: Path = RECORDINGS_DIR, label: str = ""):
        folder.mkdir(parents=True, exist_ok=True)
        tag = f"_{label}" if label else ""
        self.path = folder / f"emg_{time.strftime('%Y%m%d_%H%M%S')}{tag}.csv"
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self.frames = 0

    def add(self, packet):
        """``packet`` is a protocol.Packet."""
        now = time.monotonic() - self._t0
        step = 1.0 / 500.0
        with self._lock:
            n = packet.n_frames
            for i in range(n):
                row = [f"{now - (n - 1 - i) * step:.4f}", packet.sequence, int(packet.frame_ids[i])]
                row += [int(v) for v in packet.frames[i, :8]]
                row += [f"{v:.2f}" for v in packet.frames[i, 8:]]
                self._writer.writerow(row)
            self.frames += n

    def close(self):
        with self._lock:
            try:
                self._file.close()
            except OSError:
                pass
