"""Random Forest gesture model: loading and the background prediction worker."""

from __future__ import annotations

import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .features import extract_window_features
from ..paths import PROJECT_ROOT
from ..telemetry import Timing

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
    train_windows: dict = None          # {class: number of training windows}, from the sidecar .json when present
    test_accuracy: float = None


class ModelError(RuntimeError):
    pass


# Libraries a Model_Training bundle may need to be unpickled; they are optional in the app.
_OPTIONAL_LIBS = {"catboost": "catboost", "xgboost": "xgboost", "lightgbm": "lightgbm", "torch": "torch<2.10"}


def _read_failure(exc: Exception) -> ModelError:
    missing = getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else None
    root = (missing or "").split(".")[0]
    if root in _OPTIONAL_LIBS:
        return ModelError(
            f"This model was trained with '{root}', which is not installed for this app. Install it with:  "
            f"\"{sys.executable}\" -m pip install \"{_OPTIONAL_LIBS[root]}\"  and restart the app, "
            f"or pick a model that needs no extra library (Random Forest, Extra Trees, LDA, SVM, k-NN, MLP, ...).")
    return ModelError(f"Could not read the model file: {exc}")


# Decisions are taken at least this often (50 ms at 500 Hz) whatever stride the model was trained with: a model
# trained with a 100 ms stride can still be asked every 50 ms, which halves the delay before the arm reacts.
MAX_STRIDE_SAMPLES = 25


def _sidecar_info(path):
    """Training window counts and test accuracy from ``<model>.json`` next to the model (both optional)."""
    import json
    try:
        info = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        windows = info["dataset"]["class_windows"]["train"]
        accuracy = info.get("metrics", {}).get("accuracy")
        return {str(k): int(v) for k, v in windows.items()}, (float(accuracy) if accuracy is not None else None)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None, None


def _load_gesture_bundle(artifact) -> LoadedModel:
    """A bundle written by Model_Training: the classifier does filtering, features and scaling itself."""
    clf = artifact["classifier"]
    return LoadedModel(
        path="", model=clf, class_names=[str(c) for c in clf.classes], input_channels=int(clf.n_channels),
        window_samples=int(clf.window_samples), stride_samples=min(int(clf.stride_samples), MAX_STRIDE_SAMPLES),
        n_features=int(artifact.get("metadata", {}).get("n_features", 0) or 0))


def load_model(path) -> LoadedModel:
    import joblib
    if str(PROJECT_ROOT) not in sys.path:          # bundles from Model_Training reference its classes
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise _read_failure(exc) from exc
    if isinstance(artifact, dict) and artifact.get("format") == "emg_gesture_classifier":
        loaded = _load_gesture_bundle(artifact)
        loaded.path = str(path)
        loaded.train_windows, loaded.test_accuracy = _sidecar_info(path)
        return loaded
    if isinstance(artifact, dict) and "pipeline" in artifact:
        raise ModelError("This is a table-data model (one row per sample, no EMG windows), so it cannot drive the "
                         "arm. Train a gesture model in Model_Training with the EMG recording option instead.")
    if not isinstance(artifact, dict) or "model" not in artifact or "class_names" not in artifact:
        raise ModelError("This file is not a gesture model (expected a Model_Training EMG bundle, or 'model' and "
                         "'class_names').")
    model = artifact["model"]
    n_features = int(getattr(model, "n_features_in_", 0))
    channels = artifact.get("input_channels") or CHANNELS_BY_FEATURES.get(n_features, 4)
    return LoadedModel(
        path=str(path), model=model, class_names=[str(c) for c in artifact["class_names"]],
        input_channels=int(channels), window_samples=int(artifact.get("window_samples", 100)),
        stride_samples=int(artifact.get("stride_samples", 25)), n_features=n_features)


class InferenceWorker(QThread):
    """Extracts features from the newest window and predicts. Older windows are skipped, not queued."""
    prediction_ready = pyqtSignal(str, float, object)     # label, confidence, telemetry.Timing
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
        self._submitted_at = 0.0
        self._signal = (0.0, None)        # (arrival of the newest samples, seconds since the previous window)
        self.last_latency_s = -1.0        # newest window handed in -> prediction ready (measured)

    def load_model(self, model, class_names):
        with self._lock:
            self.model, self.class_names = model, list(class_names)

    def submit_window(self, window, arrived=None, since_previous=None):
        """``arrived``: time.perf_counter() when the newest samples reached this computer."""
        with self._lock:
            self._window = window
            self._submitted_at = time.monotonic()
            self._signal = (time.perf_counter() if arrived is None else arrived, since_previous)
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
                t_submit = self._submitted_at
                arrived, since_previous = self._signal
            if win is None or model is None:
                continue
            try:
                start, prepared = time.perf_counter(), None
                if all(hasattr(model, a) for a in ("transform", "classes", "model")):
                    # Model_Training bundle: the same two steps as model.predict_window, timed separately
                    features = model.transform(win.T)
                    prepared = time.perf_counter()
                    proba = model.model.predict_proba(features)[0]
                    idx = int(np.argmax(proba))
                    label, conf = str(model.classes[idx]), float(proba[idx])
                elif hasattr(model, "predict_window"):        # raw (samples, channels) in, no split available
                    label, conf = model.predict_window(win.T)
                    label, conf = str(label), float(conf)
                else:
                    feats = extract_window_features(win, sample_rate=self.sample_rate).reshape(1, -1)
                    prepared = time.perf_counter()
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(feats)[0]
                        idx = int(np.argmax(proba))
                        conf = float(proba[idx])
                        best = list(getattr(model, "classes_", range(len(proba))))[idx]
                    else:
                        best, conf = model.predict(feats)[0], 1.0
                    label = classes[best] if isinstance(best, (int, np.integer)) and 0 <= best < len(classes) \
                        else str(best)
                timing = Timing(arrived, start, prepared, time.perf_counter(), since_previous)
                self.last_latency_s = time.monotonic() - t_submit
                self.prediction_ready.emit(label, conf, timing)
            except Exception as exc:                      # report once, keep running
                if not self._reported:
                    self._reported = True
                    self.error.emit(f"Prediction failed: {exc}")

    def stop(self):
        self._running = False
        self._event.set()
        self.wait()
