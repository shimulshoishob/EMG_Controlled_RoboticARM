"""Random Forest gesture model: loading and the background prediction worker."""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .features import extract_window_features

try:  # models saved with another scikit-learn version still work; don't nag
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

CHANNELS_BY_FEATURES = {70: 4, 156: 8, 231: 11}


@dataclass
class LoadedModel:
    path: str
    model: object
    class_names: List[str]
    input_channels: int
    window_samples: int
    stride_samples: int
    n_features: int
    summary: str = ""


class ModelError(RuntimeError):
    pass


def load_model(path) -> LoadedModel:
    import joblib
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise ModelError(f"Could not read the model file: {exc}") from exc
    if not isinstance(artifact, dict) or "model" not in artifact or "class_names" not in artifact:
        raise ModelError("This file is not a BioWave model (expected 'model' and 'class_names').")
    model = artifact["model"]
    n_features = int(getattr(model, "n_features_in_", 0))
    channels = artifact.get("input_channels") or CHANNELS_BY_FEATURES.get(n_features, 4)
    return LoadedModel(
        path=str(path), model=model, class_names=[str(c) for c in artifact["class_names"]],
        input_channels=int(channels), window_samples=int(artifact.get("window_samples", 100)),
        stride_samples=int(artifact.get("stride_samples", 25)), n_features=n_features)


def find_models(*roots) -> List[Path]:
    """All ``*.joblib`` models below the given folders, newest first."""
    found = []
    for root in roots:
        root = Path(root)
        if root.is_dir():
            found += [p for p in root.rglob("*.joblib") if p.is_file()]
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


class InferenceWorker(QThread):
    """Extracts features from the newest window and predicts. Older windows are skipped, not queued."""
    prediction_ready = pyqtSignal(str, float)
    error = pyqtSignal(str)

    def __init__(self, sample_rate=500):
        super().__init__()
        self.sample_rate = sample_rate
        self.model = None
        self.class_names: List[str] = []
        self._window = None
        self._running = True
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._reported = False

    def load_model(self, model, class_names):
        with self._lock:
            self.model, self.class_names = model, list(class_names)

    def submit_window(self, window):
        with self._lock:
            self._window = window
        self._event.set()

    def run(self):
        while self._running:
            self._event.wait(0.1)
            if not self._running:
                break
            if not self._event.is_set():
                continue
            self._event.clear()
            with self._lock:
                win, model, classes, self._window = self._window, self.model, self.class_names, None
            if win is None or model is None:
                continue
            try:
                feats = extract_window_features(win, sample_rate=self.sample_rate).reshape(1, -1)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feats)[0]
                    idx = int(np.argmax(proba))
                    conf = float(proba[idx])
                    best = list(getattr(model, "classes_", range(len(proba))))[idx]
                else:
                    best, conf = model.predict(feats)[0], 1.0
                label = classes[best] if isinstance(best, (int, np.integer)) and 0 <= best < len(classes) else str(best)
                self.prediction_ready.emit(label, conf)
            except Exception as exc:                      # report once, keep running
                if not self._reported:
                    self._reported = True
                    self.error.emit(f"Prediction failed: {exc}")

    def stop(self):
        self._running = False
        self._event.set()
        self.wait()
