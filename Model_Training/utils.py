"""Small shared helpers: logging, seeding, timing, memory sampling, JSON, dependency checks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import platform
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger("model_training")


def setup_logging(log_file: Path | None = None, level=logging.INFO) -> logging.Logger:
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    LOG.addHandler(console)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False
    return LOG


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if have("torch"):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


_IMPORT_ERRORS: dict = {}


def have(module: str) -> bool:
    """True only if the module is installed AND actually imports (a broken torch install counts as missing).
    The result (and the first error) is cached: torch cannot be re-imported after a failed import."""
    if module not in _IMPORT_ERRORS:
        if importlib.util.find_spec(module) is None:
            _IMPORT_ERRORS[module] = "not installed"
        else:
            try:
                importlib.import_module(module)
                _IMPORT_ERRORS[module] = None
            except Exception as exc:                     # e.g. missing DLL
                _IMPORT_ERRORS[module] = f"installed but fails to import: {exc}"
    return _IMPORT_ERRORS[module] is None


def import_error(module: str) -> str | None:
    have(module)
    return _IMPORT_ERRORS[module]


def lib_version(module: str) -> str | None:
    from importlib import metadata
    dist = {"sklearn": "scikit-learn"}.get(module, module)
    try:
        return metadata.version(dist)
    except Exception:
        return None


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "libraries": {m: lib_version(m) for m in
                          ("numpy", "scipy", "pandas", "sklearn", "torch", "xgboost", "lightgbm", "catboost",
                           "matplotlib", "psutil", "joblib")}}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    info["cuda_available"] = False
    if have("torch"):
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


class PeakMemory:
    """Context manager sampling process RSS; ``peak_mb`` is the increase over the starting RSS."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak_mb = float("nan")
        self.abs_peak_mb = float("nan")
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def _sample(self):
        rss = self._proc.memory_info().rss
        if rss > self._max:
            self._max = rss

    def __enter__(self):
        if self._proc is None:
            return self
        self._base = self._max = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._proc is None:
            return False
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sample()
        self.peak_mb = (self._max - self._base) / 1e6
        self.abs_peak_mb = self._max / 1e6
        return False


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        return False


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def next_experiment_dir(root: Path, name: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if name:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        return d
    n = 1
    while (root / f"experiment_{n:03d}").exists():
        n += 1
    d = root / f"experiment_{n:03d}"
    d.mkdir()
    return d


def dir_size_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
