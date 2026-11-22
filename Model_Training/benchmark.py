"""Real-time benchmarking on CPU: latency distribution, throughput, load time, memory, model size."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import BenchmarkConfig
from .inference import GestureClassifier
from .utils import PeakMemory, dir_size_bytes, have


def _set_torch_threads(n: int) -> None:
    if have("torch"):
        import torch
        torch.set_num_threads(int(n))


def _stats(ms: np.ndarray, prefix: str) -> dict:
    return {f"{prefix}_mean_ms": float(ms.mean()), f"{prefix}_median_ms": float(np.median(ms)),
            f"{prefix}_p95_ms": float(np.percentile(ms, 95)), f"{prefix}_p99_ms": float(np.percentile(ms, 99)),
            f"{prefix}_max_ms": float(ms.max()), f"{prefix}_std_ms": float(ms.std())}


def _time_loop(fn, n: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn(0)
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn(i)
        out[i] = (time.perf_counter_ns() - t0) / 1e6
    return out


def realtime_status(p95_ms: float, budget_ms: float, comfortable_fraction: float) -> str:
    if p95_ms <= comfortable_fraction * budget_ms:
        return "comfortable"
    if p95_ms <= budget_ms:
        return "marginal"
    return "too slow"


def benchmark_classifier(clf: GestureClassifier, windows: np.ndarray, cfg: BenchmarkConfig, budget_ms: float,
                         workdir: Path, train_threads: int) -> dict:
    """Measure the *deployed* pipeline (raw window in, label out) with ``cfg.threads`` CPU threads.

    * single-window latency: filter + features/normalisation + model forward + argmax, one window per call
    * model-only latency: forward pass on an already transformed window
    * batch latency / throughput: ``batch_sizes`` windows per call
    * load time: reading the saved model directory back from disk (median of ``load_repeats``)
    """
    res: dict = {}
    clf.set_threads(cfg.threads)
    _set_torch_threads(cfg.threads)
    n = len(windows)
    if n == 0:
        return {"benchmark_error": "no test windows"}

    with PeakMemory() as mem:
        single = _time_loop(lambda i: clf.predict_proba(windows[i % n][None]), cfg.n_single, cfg.warmup)
        pre_in = [clf.transform(windows[i][None]) for i in range(min(n, 64))]
        prep = _time_loop(lambda i: clf.transform(windows[i % n][None]), max(50, cfg.n_single // 2), cfg.warmup)
        model_only = _time_loop(lambda i: clf.model.predict_proba(pre_in[i % len(pre_in)]), cfg.n_single, cfg.warmup)
        for bs in cfg.batch_sizes:
            batch = windows[np.arange(bs) % n]
            t = _time_loop(lambda i: clf.predict_proba(batch), 30, 3)
            res[f"batch{bs}_latency_ms"] = float(np.median(t))
            res[f"batch{bs}_per_window_ms"] = float(np.median(t) / bs)
            res[f"batch{bs}_throughput_wps"] = float(bs / (np.median(t) / 1000.0))
    res.update(_stats(single, "lat"))
    res["prep_median_ms"] = float(np.median(prep))
    res["model_only_median_ms"] = float(np.median(model_only))
    res["avg_time_per_window_ms"] = res["lat_mean_ms"]
    res["throughput_single_wps"] = float(1000.0 / res["lat_mean_ms"])
    res["inference_peak_mem_mb"] = None if np.isnan(mem.peak_mb) else float(mem.peak_mb)

    # initialisation / load time from disk, and on-disk size of the model (classifier only)
    tmp = Path(tempfile.mkdtemp(prefix="bench_", dir=workdir))
    try:
        model_file = clf.save(tmp)
        size = dir_size_bytes(tmp) - (tmp / "meta.json").stat().st_size - (tmp / "preproc.npz").stat().st_size
        res["model_size_kb"] = size / 1024.0
        loads = []
        for _ in range(max(1, cfg.load_repeats)):
            t0 = time.perf_counter()
            GestureClassifier.load(tmp)
            loads.append((time.perf_counter() - t0) * 1000.0)
        res["load_time_ms"] = float(np.median(loads))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res["realtime_budget_ms"] = budget_ms
    res["realtime_status"] = realtime_status(res["lat_p95_ms"], budget_ms, cfg.comfortable_fraction)
    res["realtime_headroom_x"] = float(budget_ms / max(res["lat_p95_ms"], 1e-9))
    clf.set_threads(train_threads)
    _set_torch_threads(train_threads)
    return res
