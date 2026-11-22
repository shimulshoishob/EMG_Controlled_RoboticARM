"""Classification metrics: overall, per gesture, per held-out group, confusion matrix, ROC-AUC."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def _auc_per_class(y, proba) -> np.ndarray:
    K = proba.shape[1]
    out = np.full(K, np.nan)
    for k in range(K):
        pos = (y == k)
        if 0 < pos.sum() < len(y):
            try:
                out[k] = roc_auc_score(pos, proba[:, k])
            except ValueError:
                pass
    return out


def evaluate(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str],
             groups: Optional[Dict[str, np.ndarray]] = None, overlap_factor: float = 1.0) -> dict:
    """Score predictions.

    ``groups`` maps a grouping name (e.g. ``"subject"``) to an array of group ids per window; per-group
    scores are reported to show spread across held-out people/sessions.
    ``overlap_factor`` = stride / window: windows overlap, so the effective number of independent samples
    is roughly ``n * overlap_factor`` (used for the accuracy confidence half-width).
    """
    K = len(classes)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = proba.argmax(axis=1)
    present = sorted(set(y_true.tolist()))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="weighted", zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    n_eff = max(1.0, len(y_true) * min(1.0, overlap_factor))
    auc = _auc_per_class(y_true, proba)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_m), "precision_weighted": float(p_w),
        "recall_macro": float(r_m), "recall_weighted": float(r_w),
        "f1_macro": float(f_m), "f1_weighted": float(f_w),
        "roc_auc_macro": float(np.nanmean(auc)) if np.isfinite(auc).any() else None,
        "accuracy_ci95_halfwidth": float(1.96 * np.sqrt(max(acc * (1 - acc), 1e-12) / n_eff)),
        "n_windows": int(len(y_true)),
    }
    per_class = [{"gesture": classes[k], "precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                  "support": int(s[k]), "roc_auc": None if np.isnan(auc[k]) else float(auc[k])} for k in range(K)]
    by_group: Dict[str, Dict[str, dict]] = {}
    for gname, garr in (groups or {}).items():
        garr = np.asarray(garr)
        res = {}
        for g in sorted(set(garr.tolist())):
            m = garr == g
            res[str(g)] = {"accuracy": float(accuracy_score(y_true[m], y_pred[m])),
                           "f1_macro": float(f1_score(y_true[m], y_pred[m], average="macro", zero_division=0)),
                           "n_windows": int(m.sum())}
        by_group[gname] = res
    return {"metrics": metrics, "confusion_matrix": cm, "per_class": per_class, "by_group": by_group}


def group_spread(by_group: Dict[str, dict]) -> dict:
    """Mean / std / min of macro-F1 across held-out groups (subjects or sessions)."""
    f = np.array([v["f1_macro"] for v in by_group.values()], dtype=float)
    if f.size == 0:
        return {"group_f1_mean": None, "group_f1_std": None, "group_f1_min": None, "n_groups": 0}
    return {"group_f1_mean": float(f.mean()), "group_f1_std": float(f.std()), "group_f1_min": float(f.min()),
            "n_groups": int(f.size)}
