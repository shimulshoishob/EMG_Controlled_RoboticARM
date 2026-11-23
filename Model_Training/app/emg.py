"""EMG session-folder training for the GUI: options -> ``ExperimentConfig``, run preparation and the per-model worker.

Everything model-related is the EXISTING code: ``session_io`` (loading), ``splits`` (leakage-safe split before windowing),
``pipeline.prepare`` (windows, filtering, features, train-fitted normalisers), ``models.registry`` (all 25 models),
``evaluation.evaluate``, ``benchmark.benchmark_classifier`` (CPU latency of the deployed pipeline) and
``inference.GestureClassifier`` (the real-time classifier).  This module only adds what the GUI needs on top:

* ``build_emg_run``   - load/split/window ONCE, save the arrays as ``.npy`` so every worker memory-maps the same data;
* ``run_emg_task``    - one model per process: train (Adam / AdamW), evaluate on held-out data, simulate the real-time
                        decision stream (causal soft-vote smoothing), benchmark latency, export a ``.joblib`` bundle.
"""

from __future__ import annotations

import json
import logging
import platform
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import ExperimentConfig
from ..models.registry import SPECS
from ..utils import LOG
from .engine import RunPlan, new_run_id
from .paths import Dirs

SPLIT_MODES = {
    "auto": "Automatic (strictest the data allows)",
    "subject": "New people (subject-independent)",
    "trial": "Same people, later repetitions (personal model)",
    "session": "New sessions",
    "recording": "New recordings",
    "temporal": "Interleaved time blocks (optimistic)",
}


@dataclass
class EmgOptions:
    """Everything the EMG part of the *Configure* tab controls (JSON-serialisable, stored with every model)."""
    paths: List[str] = field(default_factory=list)
    # ---- windows / filtering (applied identically in training and in the real-time classifier)
    window_ms: float = 200.0
    stride_ms: float = 100.0
    bandpass_lo: float = 20.0
    bandpass_hi: float = 220.0
    notch_hz: Optional[float] = None         # 50 / 60 / None - measured not to help on dry-electrode data, off by default
    notch_harmonics: int = 1
    # ---- what part of each session is used
    perform_trim_s: List[float] = field(default_factory=lambda: [0.5, 0.25])
    rest_trim_s: List[float] = field(default_factory=lambda: [1.0, 0.25])
    rest_cap_ratio: Optional[float] = 2.0
    min_valid_ratio: float = 0.90
    # ---- evaluation protocol
    split_mode: str = "auto"
    test_size: float = 0.2
    val_size: float = 0.15
    seed: int = 42
    # ---- training
    optimizer: str = "adam"                  # adam | adamw
    epochs: int = 40
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 8
    class_weight: bool = True
    max_train_seconds: float = 600.0         # wall-clock cap per deep model
    # ---- real-time evaluation
    smoothing_k: int = 5                     # causal soft-vote over the last k windows
    benchmark: bool = True
    budget_ms: float = 50.0                  # real-time decision interval the latency is judged against

    def to_dict(self) -> dict:
        return asdict(self)


def to_config(o: EmgOptions) -> ExperimentConfig:
    c = ExperimentConfig()
    d = c.data
    d.paths, d.layout = list(o.paths), "sessions"
    d.perform_trim_s, d.rest_trim_s = list(o.perform_trim_s), list(o.rest_trim_s)
    d.rest_cap_ratio, d.min_valid_ratio = o.rest_cap_ratio, o.min_valid_ratio
    c.preprocess.window_ms, c.preprocess.stride_ms = o.window_ms, o.stride_ms
    c.preprocess.bandpass_hz = [o.bandpass_lo, o.bandpass_hi] if (o.bandpass_lo or o.bandpass_hi) else None
    c.preprocess.notch_hz, c.preprocess.notch_harmonics = o.notch_hz, o.notch_harmonics
    c.split.mode, c.split.test_size, c.split.val_size, c.split.seed = o.split_mode, o.test_size, o.val_size, o.seed
    t = c.train
    t.seed, t.optimizer, t.epochs, t.batch_size, t.lr = o.seed, o.optimizer, o.epochs, o.batch_size, o.lr
    t.weight_decay, t.patience, t.class_weight, t.max_train_seconds = o.weight_decay, o.patience, o.class_weight, o.max_train_seconds
    c.cv.enabled, c.tuning.enabled = False, False
    b = c.benchmark
    b.enabled, b.threads, b.n_single, b.warmup, b.load_repeats = o.benchmark, 1, 100, 10, 2
    b.budget_ms = o.budget_ms
    return c


# --------------------------------------------------------------------------------------- preparation
def _save_part(work: Path, name: str, part) -> None:
    np.save(work / f"{name}_y.npy", part.y)
    np.save(work / f"{name}_raw.npy", np.ascontiguousarray(part.raw, dtype=np.float32))
    if part.feat is not None:
        np.save(work / f"{name}_feat.npy", np.ascontiguousarray(part.feat, dtype=np.float32))
    for k, v in part.meta.items():
        np.save(work / f"{name}_meta_{k}.npy", np.asarray(v).astype(str))
    np.save(work / f"{name}_seg.npy", np.asarray(part.seg))
    if part.windows is not None:
        np.save(work / f"{name}_windows.npy", np.ascontiguousarray(part.windows, dtype=np.float32))


def build_emg_run(recs, load_report: dict, opts: EmgOptions, model_keys: Sequence[str], dirs: Optional[Dirs] = None,
                  max_parallel: Optional[int] = None, cpu_budget: Optional[int] = None,
                  progress: Callable[[str], None] = lambda s: None) -> RunPlan:
    """Split (before windowing), window, preprocess, fit normalisers on TRAIN only, save arrays for the workers."""
    import joblib
    from ..pipeline import filter_rare_classes, prepare
    from ..splits import split_pieces
    from ..data_io import summarize_dataset

    dirs = (dirs or Dirs()).ensure()
    cfg = to_config(opts)
    warn: List[str] = []
    progress("checking classes")
    recs = filter_rare_classes(list(recs), cfg)
    classes = sorted({lab for r in recs for _, _, lab in r.runs})
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 gesture classes, found {classes}")
    summary = summarize_dataset(recs)
    progress("splitting (before windowing, so no window is shared between train and test)")
    pieces, split_info = split_pieces(recs, cfg.split, cfg.window_samples)
    keys = list(model_keys)
    need_feat = any(SPECS[k].input_kind == "features" for k in keys)
    progress("windowing, filtering, extracting features (normalisers are fitted on train only)")
    prepared = prepare(recs, pieces, classes, cfg, need_feat, keep_windows=("test",))
    if "test" not in prepared.parts or "val" not in prepared.parts:
        raise ValueError("The split left the validation or test set empty - use more recordings or another protocol.")
    counts = {k: int(len(p.y)) for k, p in prepared.parts.items()}
    missing = [classes[c] for c in range(len(classes)) if (prepared.parts["test"].y == c).sum() == 0]
    if missing:
        warn.append(f"Classes absent from the test set (metrics cover the rest): {missing}")
    if opts.split_mode not in ("auto", split_info["mode"]):
        warn.append(f"Requested split '{opts.split_mode}' but used '{split_info['mode']}'.")
    if split_info["mode"] == "temporal":
        warn.append("Interleaved time blocks share trials between train and test: scores are optimistic.")
    n_skipped = len(load_report.get("skipped", []))
    if n_skipped:
        warn.append(f"{n_skipped} folder(s) were skipped while loading (legacy format / raw-only / too many invalid samples).")

    run_id = new_run_id(dirs)
    work = dirs.work / run_id
    work.mkdir(parents=True, exist_ok=True)
    (dirs.reports / run_id).mkdir(parents=True, exist_ok=True)
    progress("saving arrays for the training processes")
    for name, part in prepared.parts.items():
        _save_part(work, name, part)
    joblib.dump({"classes": prepared.classes, "fs": prepared.fs, "n_channels": prepared.n_channels,
                 "n_samples": prepared.n_samples, "preprocessor": prepared.preprocessor,
                 "extractor": prepared.extractor, "feat_scaler": prepared.feat_scaler, "raw_norm": prepared.raw_norm,
                 "parts": list(prepared.parts), "channel_names": summary.get("channel_names", [])}, work / "prepared.joblib")

    cpus = cpu_budget or __import__("os").cpu_count() or 2
    parallel = max(1, min(max_parallel or len(keys), len(keys)))
    class_windows = {k: {classes[c]: int((p.y == c).sum()) for c in range(len(classes))} for k, p in prepared.parts.items()}
    info = {"name": ", ".join(Path(p).name for p in opts.paths)[:120], "kind": "emg",
            "summary": f"{len(recs)} sessions, {len(summary['subjects'])} subjects, {len(classes)} classes, "
                       f"{summary['n_channels']} EMG channels @ {summary['sampling_rate']:.0f} Hz; split '{split_info['mode']}' "
                       f"(held out: {split_info['independent_of']}); windows {counts}",
            "rows_used": counts["train"] + counts["val"] + counts["test"], "rows_original": counts["train"] + counts["val"] + counts["test"],
            "subjects": summary["subjects"], "classes": classes, "split": {k: v for k, v in split_info.items() if k != "groups"},
            "held_out_groups": split_info.get("groups", {}).get("test", []), "window_counts": counts,
            "class_windows": class_windows, "seconds_per_class": summary["seconds_per_class"], "fs": prepared.fs,
            "channels": summary.get("channel_names", []), "window_samples": cfg.window_samples}
    return RunPlan(run_id, "emg", "classification", "gesture", opts.to_dict(), opts.seed, warn,
                   counts["train"] + counts["val"], counts["test"], dirs, keys, info, work, parallel, cpus,
                   extra={"cfg": cfg.to_dict(), "emg": True})


# --------------------------------------------------------------------------------- real-time evaluation
def causal_smooth(proba: np.ndarray, seg: np.ndarray, k: int) -> np.ndarray:
    """What the real-time loop does: the decision at time t is the argmax of the MEAN probability of the last k windows.
    The history is reset at every segment (trial) boundary; windows are chronological inside a segment."""
    k = max(1, int(k))
    if k == 1 or len(proba) == 0:
        return proba
    out = np.empty_like(proba)
    cut = np.flatnonzero(np.diff(seg) != 0) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(seg)]):
        c = np.vstack([np.zeros((1, proba.shape[1])), np.cumsum(proba[a:b], axis=0)])
        i = np.arange(1, b - a + 1)
        lo = np.maximum(0, i - k)
        out[a:b] = (c[i] - c[lo]) / (i - lo)[:, None]
    return out


# ------------------------------------------------------------------------------------------- worker
def run_emg_task(payload: dict, q) -> None:
    key = payload["key"]
    warnings.filterwarnings("ignore")

    def send(kind: str, **kw):
        q.put({"type": kind, "key": key, **kw})

    from .worker import _QueueHandler
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    LOG.addHandler(_QueueHandler(send, key))
    try:
        _train_emg(payload, send)
    except BaseException as exc:                              # noqa: BLE001
        send("failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())


def _load_prepared(work: Path):
    import joblib
    from ..pipeline import Partition, Prepared
    sk = joblib.load(work / "prepared.joblib")
    parts = {}
    for name in sk["parts"]:
        meta = {k: np.load(work / f"{name}_meta_{k}.npy") for k in ("subject", "session", "rec")}
        feat = work / f"{name}_feat.npy"
        win = work / f"{name}_windows.npy"
        parts[name] = Partition(y=np.load(work / f"{name}_y.npy"), meta=meta,
                                raw=np.load(work / f"{name}_raw.npy", mmap_mode="r"),
                                feat=np.load(feat, mmap_mode="r") if feat.exists() else None,
                                windows=np.load(win, mmap_mode="r") if win.exists() else None,
                                seg=np.load(work / f"{name}_seg.npy"))
    return Prepared(sk["classes"], sk["fs"], sk["n_channels"], sk["n_samples"], sk["preprocessor"], sk["extractor"],
                    sk["feat_scaler"], sk["raw_norm"], parts), sk


def _train_emg(p: dict, send: Callable) -> None:
    import joblib
    from ..benchmark import benchmark_classifier
    from ..crossval import build_model
    from ..evaluation import evaluate, group_spread
    from ..inference import GestureClassifier
    from ..utils import lib_version, seed_everything
    from threadpoolctl import threadpool_limits

    key, seed = p["key"], int(p["seed"])
    n_jobs = max(1, int(p["n_jobs"]))
    cfg = ExperimentConfig.from_dict(p["cfg"])
    cfg.train.n_jobs = cfg.train.torch_threads = n_jobs
    opts = EmgOptions(**p["options"])
    spec = SPECS[key]
    work = Path(p["split_path"])
    send("status", state="training")
    send("progress", pct=3, msg="loading windows", indeterminate=False)
    seed_everything(seed)
    t_total = time.perf_counter()
    prepared, sk = _load_prepared(work)
    classes = prepared.classes
    train, val, test = prepared.parts["train"], prepared.parts["val"], prepared.parts["test"]
    kind = spec.input_kind
    params = {}
    if key == "xgboost" and len(classes) == 2:
        params["eval_metric"] = "logloss"                       # registry default 'mlogloss' is multiclass-only

    with threadpool_limits(limits=n_jobs):
        send("progress", pct=10, msg=f"training ({opts.optimizer})" if kind == "raw" else "training",
             indeterminate=(kind != "raw"))
        model = build_model(spec, prepared, cfg, "cpu", params)
        t0 = time.perf_counter()
        model.fit(train.data(kind), val.data(kind))
        fit_time = time.perf_counter() - t0

        send("progress", pct=84, msg="evaluating", indeterminate=False)
        t0 = time.perf_counter()
        proba = model.predict_proba(test.data(kind)["X"])
        predict_time = time.perf_counter() - t0
        overlap = cfg.stride_samples / cfg.window_samples
        ev = evaluate(test.y, proba, classes, groups={"subject": test.meta["subject"], "session": test.meta["session"]},
                      overlap_factor=overlap)
        vproba = model.predict_proba(val.data(kind)["X"])
        vev = evaluate(val.y, vproba, classes)
        spread = group_spread(ev["by_group"].get("subject", {}))

        # ---- the real-time decision stream: soft-vote over the last k windows, history reset per trial
        sm = causal_smooth(proba, test.seg, opts.smoothing_k)
        sev = evaluate(test.y, sm, classes)
        realtime = {"smoothing_k": opts.smoothing_k, "accuracy_smoothed": sev["metrics"]["accuracy"],
                    "f1_macro_smoothed": sev["metrics"]["f1_macro"],
                    "balanced_accuracy_smoothed": sev["metrics"]["balanced_accuracy"],
                    "decision_delay_ms": float(opts.window_ms + (max(opts.smoothing_k, 1) - 1) * opts.stride_ms)}

        clf = GestureClassifier.from_training(model, prepared, cfg, key)
        if cfg.benchmark.enabled and test.windows is not None and len(test.windows):
            send("progress", pct=90, msg="latency benchmark", indeterminate=False)
            step = max(1, len(test.windows) // 256)
            bench_dir = work / f"bench_{key}"
            bench_dir.mkdir(exist_ok=True)
            try:
                realtime.update(benchmark_classifier(clf, np.asarray(test.windows[::step][:256]), cfg.benchmark,
                                                     cfg.budget_ms, bench_dir, n_jobs))
            finally:
                import shutil
                shutil.rmtree(bench_dir, ignore_errors=True)
    train_time = fit_time

    # ---- export -----------------------------------------------------------------------------------------
    send("progress", pct=95, msg="exporting", indeterminate=False)
    path = Path(p["trained_dir"]) / f"{p['file_stem']}_{p['run_id']}.joblib"
    meta = {
        "model_key": key, "model_name": p["label"], "family": spec.family, "task": "classification", "kind": "emg_gesture_classifier",
        "run_id": p["run_id"], "created": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "classes": classes, "fs": prepared.fs, "channels": sk.get("channel_names", []), "n_channels": prepared.n_channels,
        "window_samples": prepared.n_samples, "window_ms": opts.window_ms, "stride_ms": opts.stride_ms,
        "input_kind": kind, "preprocess": asdict(cfg.preprocess), "features": asdict(cfg.features) if kind == "features" else None,
        "n_features": prepared.n_features, "optimizer": opts.optimizer if kind == "raw" else None,
        "dataset": p.get("dataset", {}), "settings": p["options"],
        "split": p["dataset"].get("split"), "window_counts": p["dataset"].get("window_counts"),
        "timing_s": {"fit": round(fit_time, 3), "predict_test": round(predict_time, 3)},
        "metrics": ev["metrics"], "val_metrics": vev["metrics"], "group_spread": spread, "realtime": realtime,
        "n_params": model.n_params(),
        "versions": {"python": platform.python_version(), **{m: lib_version(m) for m in
                     ("numpy", "scipy", "sklearn", "joblib", "torch", "xgboost", "lightgbm", "catboost") if lib_version(m)}},
        "usage": "from Model_Training.app.inference import load_gesture_classifier; clf = load_gesture_classifier(path); "
                 "label, confidence = clf.predict_window(window)  # window: (samples, channels) raw ADC counts",
    }
    bundle = {"format": "emg_gesture_classifier", "format_version": 1, "classifier": clf, "classes": classes, "metadata": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    size = path.stat().st_size

    send("progress", pct=98, msg="verifying export", indeterminate=False)
    loaded = joblib.load(path)["classifier"]
    sample = np.asarray(test.windows[:16])
    a, b = clf.predict_proba(sample), loaded.predict_proba(sample)
    if not np.allclose(a, b, atol=1e-4):
        raise RuntimeError("Exported model does not reproduce its predictions after reloading.")
    meta["artifact"] = {"path": str(path), "size_bytes": int(size), "reload_check": "ok"}
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    hist = {k: [float(x) for x in v] for k, v in (model.history or {}).items() if isinstance(v, list)}
    result = {"model_key": key, "label": p["label"], "family": spec.family, "task": "classification",
              "metrics": ev["metrics"], "primary_metric": "f1_macro", "val_metrics": vev["metrics"],
              "group_spread": spread, "by_group": ev["by_group"].get("subject", {}), "realtime": realtime,
              "train_time_s": float(train_time), "fit_time_s": float(fit_time), "predict_time_s": float(predict_time),
              "total_time_s": float(time.perf_counter() - t_total), "model_size_bytes": int(size), "artifact": str(path),
              "metadata_file": str(meta_path), "n_features": int(prepared.n_features), "n_params": model.n_params(),
              "confusion_matrix": ev["confusion_matrix"].tolist(), "per_class": ev["per_class"], "classes": classes,
              "history": hist}
    send("progress", pct=100, msg="done", indeterminate=False)
    send("result", result=result)
