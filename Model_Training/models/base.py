"""Common model interface used by training, evaluation, benchmarking and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ModelContext:
    """Everything a model factory needs to know about the data and the run."""
    n_classes: int
    n_features: int
    n_channels: int
    n_samples: int
    seed: int
    device: str
    train_cfg: Any            # config.TrainConfig
    n_train: int = 0


class BaseModel:
    """fit / predict_proba / save / load for every model family.

    ``input_kind`` tells the pipeline which representation the model consumes:
    ``"features"`` -> (N, F) standardised engineered features, ``"raw"`` -> (N, C, T) normalised windows.
    Labels are integer class indices 0..K-1 and ``predict_proba`` always returns (N, K).
    """

    key: str = ""
    family: str = ""
    input_kind: str = "features"
    kind_name: str = "base"          # key used by the loader in inference.py
    history: Dict[str, Any]

    def __init__(self):
        self.history = {}

    def fit(self, train: dict, val: Optional[dict] = None) -> "BaseModel":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, directory: Path) -> Path:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> "BaseModel":
        raise NotImplementedError

    def set_inference_threads(self, n: int) -> None:
        pass

    def n_params(self) -> Optional[int]:
        return None
