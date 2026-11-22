"""Model construction helpers, group-aware cross-validation and random-search hyper-parameter tuning."""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import ExperimentConfig
from .data_io import Recording
from .models.base import ModelContext
from .models.registry import ModelSpec
from .pipeline import Prepared, prepare
from .splits import Piece, make_folds
from .utils import LOG, seed_everything


def make_ctx(prepared: Prepared, cfg: ExperimentConfig, device: str, n_train: int) -> ModelContext:
    return ModelContext(n_classes=len(prepared.classes), n_features=prepared.n_features,
                        n_channels=prepared.n_channels, n_samples=prepared.n_samples,
                        seed=cfg.train.seed, device=device, train_cfg=cfg.train, n_train=n_train)


def build_model(spec: ModelSpec, prepared: Prepared, cfg: ExperimentConfig, device: str, params: dict):
    merged = dict(cfg.train.model_params.get(spec.key, {}))
    merged.update(params or {})
    n_train = len(prepared.parts["train"].y)
    return spec.factory(make_ctx(prepared, cfg, device, n_train), merged)


class FoldCache:
    """Builds (and caches) per-fold prepared data. Every fold fits its own normalisers on its own training part."""

    def __init__(self, recs: Sequence[Recording], classes: List[str], cfg: ExperimentConfig, mode: str):
        self.recs, self.classes, self.cfg, self.mode = recs, classes, cfg, mode
        self._cache: Dict[str, List[Prepared]] = {}

    def get(self, name: str, pieces: Sequence[Piece], k: int, need_features: bool = True) -> List[Prepared]:
        key = f"{name}:{need_features}"
        if key not in self._cache:
            out = []
            for tr, va in make_folds(pieces, self.mode, k, self.cfg.split.seed):
                if {p.seg.label for p in tr} != set(self.classes):
                    LOG.warning("  fold skipped: a gesture is missing from its training part")
                    continue
                try:
                    out.append(prepare(self.recs, {"train": tr, "val": va}, self.classes, self.cfg, need_features))
                except ValueError as exc:
                    LOG.warning("  fold skipped: %s", exc)
            self._cache[key] = out
        return self._cache[key]


def _score_fold(spec, fold: Prepared, cfg, device, params) -> Optional[dict]:
    if "val" not in fold.parts:
        return None
    seed_everything(cfg.train.seed)
    model = build_model(spec, fold, cfg, device, params)
    model.fit(fold.parts["train"].data(spec.input_kind), None)       # no validation set: fold-val stays untouched
    va = fold.parts["val"]
    pred = model.predict_proba(va.data(spec.input_kind)["X"]).argmax(1)
    return {"accuracy": float(accuracy_score(va.y, pred)),
            "f1_macro": float(f1_score(va.y, pred, average="macro", zero_division=0))}


def cross_validate(spec: ModelSpec, folds: List[Prepared], cfg: ExperimentConfig, device: str, params: dict) -> dict:
    scores = [s for s in (_score_fold(spec, f, cfg, device, params) for f in folds) if s]
    if not scores:
        return {}
    f1 = np.array([s["f1_macro"] for s in scores])
    acc = np.array([s["accuracy"] for s in scores])
    return {"cv_folds": len(scores), "cv_f1_mean": float(f1.mean()), "cv_f1_std": float(f1.std()),
            "cv_acc_mean": float(acc.mean()), "cv_acc_std": float(acc.std())}


def random_search(spec: ModelSpec, folds: List[Prepared], cfg: ExperimentConfig, device: str):
    """Random search scored by group-aware CV macro-F1 on the training pool. The default setting is always a candidate."""
    if not spec.space or not folds:
        return {}, []
    keys = sorted(spec.space)
    grid = [dict(zip(keys, vals)) for vals in itertools.product(*(spec.space[k] for k in keys))]
    rng = np.random.default_rng(cfg.train.seed)
    picks = [grid[i] for i in rng.permutation(len(grid))[: cfg.tuning.n_iter]]
    candidates = [{}] + picks
    trials = []
    for params in candidates:
        res = cross_validate(spec, folds, cfg, device, params)
        score = res.get("cv_f1_mean", -1.0)
        trials.append({"params": params, "cv_f1_mean": score})
        LOG.info("    tune %-16s %-60s cv macro-F1 %.3f", spec.key, str(params or "defaults"), score)
    best = max(trials, key=lambda t: t["cv_f1_mean"])
    return best["params"], trials
