"""Common inference interface for deployment.

A saved model directory is self-contained::

    meta.json      classes, sampling rate, window/stride, preprocessing + feature settings, library versions
    preproc.npz    normalisation statistics fitted on the training data
    model.joblib | model.pt (+ rf.joblib)   the classifier

Usage in the real-time pipeline (drop-in for the current classifier call)::

    from Model_Training.inference import GestureClassifier
    clf = GestureClassifier.load("Model_Training/results/experiment_001/models/svm_rbf")
    label, confidence = clf.predict_window(window)       # window: (samples, channels) raw ADC counts

``window`` must hold ``clf.window_samples`` samples of the same channels, in the same units as the
training recordings.  Filtering, DC removal, feature extraction, normalisation and the label mapping are
all applied internally, exactly as during training.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

if __package__ in (None, ""):                          # allow `python inference.py` as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Model_Training"

from .config import FeatureConfig, PreprocessConfig
from .features import FeatureExtractor
from .models.ml_models import SklearnModel
from .models.torch_wrapper import HybridCNNRF, TorchModel
from .preprocessing import WindowPreprocessor
from .utils import environment_info

_KINDS = {"sklearn": SklearnModel, "torch": TorchModel, "hybrid_cnn_rf": HybridCNNRF}


class GestureClassifier:
    def __init__(self, model, classes: List[str], fs: float, n_channels: int, window_samples: int,
                 stride_samples: int, preprocessor: WindowPreprocessor, extractor, norm: dict, meta: dict):
        self.model, self.classes = model, list(classes)
        self.fs, self.n_channels = float(fs), int(n_channels)
        self.window_samples, self.stride_samples = int(window_samples), int(stride_samples)
        self.preprocessor, self.extractor, self.norm, self.meta = preprocessor, extractor, norm, meta
        self.input_kind = model.input_kind

    # ------------------------------------------------------------------ representation
    def _as_batch(self, windows) -> np.ndarray:
        w = np.asarray(windows, dtype=np.float32)
        if w.ndim == 2:
            w = w[None]
        if w.ndim != 3:
            raise ValueError("windows must be (samples, channels) or (N, samples, channels)")
        if w.shape[1] == self.n_channels and w.shape[2] == self.window_samples and self.n_channels != self.window_samples:
            w = w.transpose(0, 2, 1)                        # given as (N, channels, samples)
        if w.shape[1] != self.window_samples or w.shape[2] != self.n_channels:
            raise ValueError(f"expected windows of {self.window_samples} samples x {self.n_channels} channels, "
                             f"got {tuple(w.shape[1:])}")
        return w

    def transform(self, windows) -> np.ndarray:
        """Raw windows -> the exact model input (preprocessing, features / normalisation)."""
        proc = self.preprocessor(self._as_batch(windows))
        if self.input_kind == "features":
            f = self.extractor.transform(proc)
            return ((f - self.norm["feat_mean"]) / self.norm["feat_scale"]).astype(np.float32)
        x = (proc - self.norm["raw_mean"]) / self.norm["raw_scale"]
        return np.ascontiguousarray(x.transpose(0, 2, 1).astype(np.float32))

    # ------------------------------------------------------------------ prediction
    def predict_proba(self, windows) -> np.ndarray:
        return self.model.predict_proba(self.transform(windows))

    def predict(self, windows) -> List[str]:
        return [self.classes[i] for i in self.predict_proba(windows).argmax(axis=1)]

    def predict_window(self, window) -> Tuple[str, float]:
        """One window -> (gesture label, confidence); the same shape as the app's prediction signal."""
        p = self.predict_proba(window)[0]
        i = int(p.argmax())
        return self.classes[i], float(p[i])

    def set_threads(self, n: int) -> None:
        self.model.set_inference_threads(n)

    # ------------------------------------------------------------------ persistence
    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        model_path = self.model.save(d)
        np.savez(d / "preproc.npz", **{k: np.asarray(v) for k, v in self.norm.items()})
        meta = dict(self.meta)
        meta.update({"classes": self.classes, "fs": self.fs, "n_channels": self.n_channels,
                     "window_samples": self.window_samples, "stride_samples": self.stride_samples,
                     "model_kind": self.model.kind_name, "input_kind": self.input_kind,
                     "environment": environment_info()["libraries"]})
        with open(d / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return model_path

    @classmethod
    def load(cls, directory) -> "GestureClassifier":
        d = Path(directory)
        with open(d / "meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        model = _KINDS[meta["model_kind"]].load(d)
        with np.load(d / "preproc.npz") as z:
            norm = {k: z[k] for k in z.files}
        pre = WindowPreprocessor(meta["fs"], PreprocessConfig(**meta["preprocess"]))
        ext = None
        if meta["input_kind"] == "features":
            ext = FeatureExtractor(meta["fs"], meta["n_channels"], meta["window_samples"], FeatureConfig(**meta["features"]))
        return cls(model, meta["classes"], meta["fs"], meta["n_channels"], meta["window_samples"],
                   meta["stride_samples"], pre, ext, norm, meta)

    @classmethod
    def from_training(cls, model, prepared, cfg, spec_key: str) -> "GestureClassifier":
        norm = {"raw_mean": prepared.raw_norm.mean, "raw_scale": prepared.raw_norm.scale}
        if prepared.feat_scaler is not None:
            norm.update(feat_mean=prepared.feat_scaler.mean, feat_scale=prepared.feat_scaler.scale)
        meta = {"model_key": spec_key, "preprocess": asdict(cfg.preprocess), "features": asdict(cfg.features),
                "feature_names": prepared.extractor.names if prepared.extractor else None}
        return cls(model, prepared.classes, prepared.fs, prepared.n_channels, prepared.n_samples,
                   cfg.stride_samples, prepared.preprocessor, prepared.extractor, norm, meta)


if __name__ == "__main__":                            # tiny CLI: python inference.py <model_dir> <recording.csv>
    import pandas as pd
    if len(sys.argv) < 3:
        sys.exit("usage: python Model_Training/inference.py <model_dir> <recording.csv>")
    clf = GestureClassifier.load(sys.argv[1])
    df = pd.read_csv(sys.argv[2])
    cols = [c for c in df.columns if c.lower().startswith("emg")][: clf.n_channels]
    sig = df[cols].to_numpy(np.float32)
    for s in range(0, len(sig) - clf.window_samples + 1, clf.stride_samples * 4):
        print(s / clf.fs, *clf.predict_window(sig[s:s + clf.window_samples]))
