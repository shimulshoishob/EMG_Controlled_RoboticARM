"""One training task = one OS process running ``run_task``.

The task is fully independent: it loads the shared train/test split from disk, fits the shared
preprocessing on the training rows, builds the model through the EXISTING factories
(``models.registry.SPECS`` / ``app.regression``), trains, evaluates, and exports a joblib bundle.
Everything is reported to the parent through a ``multiprocessing.Queue`` as small dict messages:

    {"type": "status"|"progress"|"log"|"result"|"failed", "key": <model key>, ...}
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)")


class _QueueHandler(logging.Handler):
    """Forwards the existing models' log lines to the GUI and turns 'epoch 5/40' into progress."""

    def __init__(self, send: Callable, key: str):
        super().__init__(logging.INFO)
        self.send, self.key = send, key

    def emit(self, record):
        try:
            msg = record.getMessage().strip()
            self.send("log", level=record.levelname, msg=msg)
            m = _EPOCH_RE.search(msg)
            if m:
                e, n = int(m.group(1)), int(m.group(2))
                self.send("progress", pct=30 + 55.0 * e / max(n, 1), msg=f"epoch {e}/{n}", indeterminate=False)
        except Exception:
            pass


def run_task(payload: dict, q) -> None:
    key = payload["key"]
    import warnings
    warnings.filterwarnings("ignore")                     # library noise would otherwise flood the console

    def send(kind: str, **kw):
        q.put({"type": kind, "key": key, **kw})

    from ..utils import LOG
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    LOG.addHandler(_QueueHandler(send, key))
    try:
        _train(payload, send)
    except BaseException as exc:                         # noqa: BLE001 - the parent must always hear back
        send("failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    from ..utils import lib_version
    for m in ("numpy", "pandas", "sklearn", "joblib", "torch", "xgboost", "lightgbm", "catboost"):
        v = lib_version(m)
        if v:
            out[m] = v
    return out


def _simple_params(est) -> dict:
    try:
        params = est.get_params(deep=False)
    except Exception:
        return {}
    out = {}
    for k, v in params.items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)[:120]
    return out


def _train(p: dict, send: Callable) -> None:
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from threadpoolctl import threadpool_limits

    from ..config import TrainConfig
    from ..evaluation import evaluate
    from ..models.base import ModelContext
    from ..models.registry import SPECS
    from ..utils import seed_everything
    from .estimators import TabularTorchClassifier, layout_for, to_layout
    from .preprocess import PreprocessOptions, feature_names, fit_preprocessing
    from .regression import build_regressor, fit_regressor, regression_metrics

    key, task, seed = p["key"], p["task"], int(p["seed"])
    opts = PreprocessOptions(**p["options"])
    n_jobs = max(1, int(p["n_jobs"]))
    send("status", state="training")
    send("progress", pct=3, msg="loading split", indeterminate=False)
    seed_everything(seed)

    t_total = time.perf_counter()
    split = joblib.load(p["split_path"])
    X_tr, X_te, y_tr, y_te = split["X_train"], split["X_test"], split["y_train"], split["y_test"]
    numeric, categorical, classes = split["numeric"], split["categorical"], split["classes"]

    with threadpool_limits(limits=n_jobs):
        # ---- preprocessing: fitted on TRAIN only -------------------------------------------------
        send("progress", pct=10, msg="preprocessing (fit on train)", indeterminate=False)
        t0 = time.perf_counter()
        pre, Xt_tr = fit_preprocessing(opts, numeric, categorical, X_tr, y_tr, task)
        Xt_te = np.asarray(pre.transform(X_te), dtype=np.float64)
        prep_time = time.perf_counter() - t0
        n_features = Xt_tr.shape[1]

        # ---- model from the existing implementations ------------------------------------------------
        cfg = TrainConfig(seed=seed, device="cpu", n_jobs=n_jobs, torch_threads=n_jobs, epochs=opts.dl_epochs,
                          class_weight=opts.class_weight, max_train_seconds=float(p.get("max_train_seconds", 900)))
        layout = layout_for(key, n_features)
        ctx = ModelContext(n_classes=max(len(classes), 1), n_features=n_features, n_channels=layout[0],
                           n_samples=layout[1], seed=seed, device="cpu", train_cfg=cfg, n_train=len(Xt_tr))
        family, wrapped, est, booster, raw_input = "ml", None, None, None, False
        if task == "classification":
            spec = SPECS[key]
            family = spec.family
            params = dict(p.get("params") or {})
            if key == "xgboost" and len(classes) == 2:
                params.setdefault("eval_metric", "logloss")      # registry default 'mlogloss' is multiclass-only
            wrapped = spec.factory(ctx, params)
            booster = getattr(wrapped, "booster", None)
            raw_input = wrapped.input_kind == "raw"
        else:
            est, booster = build_regressor(key, ctx, p.get("params"))

        def shape(X):                                   # (N, F) -> (N, C, T) float32 for the deep networks
            return to_layout(X, layout) if raw_input else X

        # ---- optional validation split (early stopping) taken from TRAIN rows ---------------------------
        Xfit, yfit, val = Xt_tr, y_tr, None
        if (booster or raw_input) and len(Xt_tr) >= 60:
            strat = y_tr if task == "classification" and np.bincount(y_tr).min() >= 2 else None
            try:
                a, b, ya, yb = train_test_split(Xt_tr, y_tr, test_size=0.15, random_state=seed, stratify=strat)
            except ValueError:
                a, b, ya, yb = train_test_split(Xt_tr, y_tr, test_size=0.15, random_state=seed)
            Xfit, yfit, val = a, ya, (b, yb)

        # ---- train ------------------------------------------------------------------------------------
        send("progress", pct=30, msg="training", indeterminate=not raw_input)
        t0 = time.perf_counter()
        if task == "classification":
            wrapped.fit({"X": shape(Xfit), "y": yfit}, {"X": shape(val[0]), "y": val[1]} if val else None)
        else:
            fit_regressor(est, booster, Xfit, yfit, val)
        fit_time = time.perf_counter() - t0

        # ---- evaluate on the untouched test rows --------------------------------------------------------
        send("progress", pct=88, msg="evaluating", indeterminate=False)
        t0 = time.perf_counter()
        if task == "classification":
            proba = wrapped.predict_proba(shape(Xt_te))
            predict_time = time.perf_counter() - t0
            ev = evaluate(y_te, proba, classes)
            metrics = ev["metrics"]
            extra = {"confusion_matrix": ev["confusion_matrix"].tolist(), "per_class": ev["per_class"],
                     "classes": classes}
            final_est = wrapped.est if hasattr(wrapped, "est") else TabularTorchClassifier(wrapped, len(classes), layout)
            if metrics.get("roc_auc_macro") is not None and hasattr(wrapped, "est") \
                    and not hasattr(wrapped.est, "predict_proba"):
                extra["roc_auc_note"] = "computed from SVM decision scores (softmax), not calibrated probabilities"
            primary = "f1_macro"
        else:
            pred = est.predict(Xt_te)
            predict_time = time.perf_counter() - t0
            metrics = regression_metrics(y_te, pred)
            idx = np.random.default_rng(seed).permutation(len(y_te))[:400]
            extra = {"y_true_sample": y_te[idx].tolist(), "y_pred_sample": np.asarray(pred)[idx].tolist()}
            final_est = est
            primary = "r2"

    train_time = prep_time + fit_time

    # ---- export: preprocessing + model in ONE pipeline --------------------------------------------------
    send("progress", pct=94, msg="exporting", indeterminate=False)
    pipeline = Pipeline(list(pre.steps) + [("model", final_est)])
    run_id = p["run_id"]
    label = p["label"]
    path = Path(p["trained_dir"]) / f"{p['file_stem']}_{run_id}.joblib"
    meta = {
        "model_key": key, "model_name": label, "family": family, "task": task, "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "dataset": p.get("dataset", {}), "target": p["target"], "numeric_features": numeric,
        "categorical_features": categorical, "n_features_after_preprocessing": int(n_features),
        "feature_names_after_preprocessing": feature_names(pre)[:500],
        "preprocessing": p["options"], "model_params": _simple_params(final_est),
        "split": {"train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                  "validation_rows_from_train": int(len(val[1])) if val else 0},
        "timing_s": {"preprocessing": round(prep_time, 4), "fit": round(fit_time, 4),
                     "predict_test": round(predict_time, 4)},
        "metrics": metrics, "versions": _versions(),
        "usage": "from Model_Training.app.inference import load_bundle, predict; b = load_bundle(path); predict(b, dataframe)",
    }
    bundle = {"format_version": 1, "task": task, "pipeline": pipeline, "label_encoder": split["label_encoder"],
              "classes": classes, "target": p["target"], "numeric": numeric, "categorical": categorical,
              "feature_columns": numeric + categorical, "metadata": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    size = path.stat().st_size

    # ---- prove the file can be loaded and reproduces the predictions -------------------------------------
    send("progress", pct=97, msg="verifying export", indeterminate=False)
    loaded = joblib.load(path)
    sample = X_te.iloc[:min(25, len(X_te))]
    a, b = pipeline.predict(sample), loaded["pipeline"].predict(sample)
    if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True):
        raise RuntimeError("Exported model does not reproduce its predictions after reloading.")
    meta["artifact"] = {"path": str(path), "size_bytes": int(size), "reload_check": "ok"}
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    result = {"model_key": key, "label": label, "family": family, "task": task, "metrics": metrics,
              "primary_metric": primary, "train_time_s": float(train_time), "fit_time_s": float(fit_time),
              "predict_time_s": float(predict_time), "total_time_s": float(time.perf_counter() - t_total),
              "model_size_bytes": int(size), "artifact": str(path), "metadata_file": str(meta_path),
              "n_features": int(n_features), **extra}
    send("progress", pct=100, msg="done", indeterminate=False)
    send("result", result=result)
