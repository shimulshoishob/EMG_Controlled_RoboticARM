"""Load an exported ``.joblib`` bundle and predict on new raw data.

    from Model_Training.app.inference import load_bundle, predict
    bundle = load_bundle("Model_Training/outputs/trained_models/RandomForest_20261123_101500.joblib")
    labels = predict(bundle, pandas.read_csv("new_rows.csv"))

The bundle is a plain dict of scikit-learn objects (preprocessing + model in one ``Pipeline``), so it also
loads with a bare ``joblib.load`` - ``bundle["pipeline"].predict(df)`` - as long as the libraries listed in
``bundle["metadata"]["versions"]`` are installed.  Deep-learning bundles additionally need ``torch`` and the
``Model_Training`` package on ``sys.path`` (``load_bundle`` adds it).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

_PROJECT_PARENT = str(Path(__file__).resolve().parents[2])


def load_bundle(path: Union[str, Path]) -> dict:
    import joblib
    if _PROJECT_PARENT not in sys.path:
        sys.path.insert(0, _PROJECT_PARENT)
    return joblib.load(path)


def predict(bundle: dict, df: pd.DataFrame, return_proba: bool = False):
    """Predict original labels (classification) or values (regression) for raw rows of ``df``."""
    from .preprocess import coerce_columns
    X = coerce_columns(df, bundle["numeric"], bundle["categorical"])
    pipe = bundle["pipeline"]
    if bundle["task"] == "regression":
        return pipe.predict(X)
    if return_proba:
        return pipe.predict_proba(X)
    pred = np.asarray(pipe.predict(X)).astype(int)
    return bundle["label_encoder"].inverse_transform(pred)


def load_gesture_classifier(path):
    """Load an EMG ``.joblib`` bundle written by the application and return its ``GestureClassifier``.

        clf = load_gesture_classifier("Model_Training/outputs/trained_models/SVMRBF_20261123_101500.joblib")
        label, confidence = clf.predict_window(window)       # window: (clf.window_samples, n_channels) raw ADC counts
        clf.classes, clf.fs, clf.window_samples, clf.stride_samples      # everything needed to stream
    The full training metadata (settings, split, metrics, versions) is in ``bundle["metadata"]`` / the sidecar ``.json``.
    """
    bundle = load_bundle(path)
    if bundle.get("format") != "emg_gesture_classifier":
        raise ValueError("not an EMG gesture-classifier bundle (use load_bundle for tabular models)")
    return bundle["classifier"]
