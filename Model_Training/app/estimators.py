"""Adapter that lets the existing PyTorch models (``models/torch_wrapper.py``) sit in a scikit-learn Pipeline.

Tabular rows are fed to the sequence networks as a 1-channel signal of length n_features, i.e.
``(N, F) -> (N, 1, F)`` (the 2-D CNN gets a padded 4 x T grid, see ``layout_for``).  Unpickling a bundle that contains one of these needs ``torch`` and the
``Model_Training`` package importable (``app/inference.py::load_bundle`` takes care of the path).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def layout_for(key: str, n_features: int):
    """(channels, length) a tabular row is reshaped to.  Sequence nets see 1 channel of length F;
    the 2-D CNN needs a real 2-D grid, so its row is zero-padded into 4 x ceil(F/4) (min width 4)."""
    if key == "cnn2d":
        return 4, max(4, -(-n_features // 4))
    return 1, n_features


def to_layout(X, layout):
    X = np.asarray(X, dtype=np.float32)
    c, t = layout
    if t == 0:
        t = X.shape[1]
    if c * t != X.shape[1]:
        X = np.pad(X, ((0, 0), (0, c * t - X.shape[1])))
    return np.ascontiguousarray(X.reshape(len(X), c, t))


class TabularTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, n_classes: int = 2, layout=(1, 0)):
        self.model = model
        self.n_classes = n_classes
        self.layout = layout                      # (channels, length); length 0 = n_features, no padding

    def __sklearn_is_fitted__(self):
        return True

    @property
    def classes_(self):
        return np.arange(self.n_classes)

    def fit(self, X, y=None):                         # already trained by the application
        return self

    def _x(self, X):
        return to_layout(X, self.layout)

    def predict_proba(self, X):
        return self.model.predict_proba(self._x(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
