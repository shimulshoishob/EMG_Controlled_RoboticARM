"""Parallel training engine (Qt-free).

* Every model runs in its own **spawned process** (``worker.run_task``): a crash, an out-of-memory kill
  or an exception in one model cannot affect the others or the GUI.
* At most ``max_parallel`` processes run at once; the rest stay *Pending* and start as slots free up.
  CPU threads are divided between the running models so they do not oversubscribe the machine.
* ``poll()`` is non-blocking and meant to be called from a GUI timer.
* ``cancel_all()`` / ``cancel(key)`` terminate processes (SIGTERM, then SIGKILL) and delete half-written output.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from ..utils import LOG
from .catalog import entry_for
from .paths import Dirs
from .preprocess import PreparedData, PreprocessOptions, prepare_data

PENDING, TRAINING, COMPLETED, FAILED, CANCELLED = "Pending", "Training", "Completed", "Failed", "Cancelled"
TERMINAL = (COMPLETED, FAILED, CANCELLED)


@dataclass
class Job:
    key: str
    label: str
    family: str
    file_stem: str
    state: str = PENDING
    pct: float = 0.0
    message: str = ""
    indeterminate: bool = False
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    result: Optional[dict] = None
    error: str = ""
    process: Optional[mp.Process] = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        if self.t_start is None:
            return 0.0
        return (self.t_end or time.time()) - self.t_start


@dataclass
class RunPlan:
    """Everything the engine needs, independent of the data kind (``tabular`` CSV or ``emg`` session folders)."""
    run_id: str
    kind: str                       # tabular | emg
    task: str                       # classification | regression
    target: str
    options: dict                   # JSON-serialisable settings; stored with every exported model
    seed: int
    warnings: List[str]
    n_train: int
    n_test: int
    dirs: Dirs
    model_keys: List[str]
    dataset_info: dict
    split_path: Path                # tabular: split.joblib; emg: the work directory with the .npy arrays
    max_parallel: int
    cpu_budget: int
    extra: dict = field(default_factory=dict)      # kind-specific worker payload
    prepared: Optional[PreparedData] = None


def new_run_id(dirs: Dirs) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id, n = base, 1
    while (dirs.reports / run_id).exists() or (dirs.work / run_id).exists():
        n += 1
        run_id = f"{base}_{n}"
    return run_id


def build_run(df: pd.DataFrame, target: str, feature_types: Dict[str, str], options: PreprocessOptions,
              model_keys: List[str], dirs: Optional[Dirs] = None, max_parallel: Optional[int] = None,
              dataset_name: str = "dataset.csv", cpu_budget: Optional[int] = None) -> RunPlan:
    """Prepare the shared train/test split once and write it to the run's work folder."""
    dirs = (dirs or Dirs()).ensure()
    prepared = prepare_data(df, target, feature_types, options)
    run_id = new_run_id(dirs)
    work = dirs.work / run_id
    work.mkdir(parents=True, exist_ok=True)
    (dirs.reports / run_id).mkdir(parents=True, exist_ok=True)
    split_path = work / "split.joblib"
    joblib.dump({"X_train": prepared.X_train, "X_test": prepared.X_test, "y_train": prepared.y_train,
                 "y_test": prepared.y_test, "numeric": prepared.numeric, "categorical": prepared.categorical,
                 "classes": prepared.classes, "label_encoder": prepared.label_encoder}, split_path)
    cpus = cpu_budget or os.cpu_count() or 2
    parallel = max(1, min(max_parallel or len(model_keys), len(model_keys)))
    info = {"name": dataset_name, "rows_used": int(len(prepared.X_train) + len(prepared.X_test)),
            "rows_original": int(len(df)), "columns_original": int(df.shape[1])}
    return RunPlan(run_id, "tabular", prepared.task, prepared.target, options.to_dict(), options.seed,
                   list(prepared.warnings), len(prepared.X_train), len(prepared.X_test), dirs, list(model_keys), info,
                   split_path, parallel, cpus, prepared=prepared)


class TrainingEngine:
    def __init__(self, plan: RunPlan):
        self.plan = plan
        self.ctx = mp.get_context("spawn")
        self.queue = self.ctx.Queue()
        self.jobs: Dict[str, Job] = {}
        for k in plan.model_keys:
            e = entry_for(k, plan.task)
            self.jobs[k] = Job(k, e.label, e.family, e.file_stem)
        self._order = list(plan.model_keys)
        self._started = False
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.cancelled = False
        self.logs: List[str] = []
        self._setup_file_log()
        atexit.register(self.shutdown)        # never leave orphaned workers (they are non-daemon so joblib threading works)

    # ------------------------------------------------------------------------------------------- logging
    def _setup_file_log(self) -> None:
        self.log_path = self.plan.dirs.logs / f"training_{self.plan.run_id}.log"
        self._fh = logging.FileHandler(self.log_path, encoding="utf-8")
        self._fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        self._logger = logging.getLogger(f"training_app.{self.plan.run_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.addHandler(self._fh)

    def log(self, msg: str, level: int = logging.INFO) -> None:
        self._logger.log(level, msg)
        self.logs.append(f"{datetime.now():%H:%M:%S}  {msg}")

    # ------------------------------------------------------------------------------------- lifecycle
    def start(self) -> None:
        p = self.plan
        self.t_start = time.time()
        self._started = True
        unit = "windows" if p.kind == "emg" else "rows"
        self.log(f"Run {p.run_id}: task={p.task}, target='{p.target}', "
                 f"{p.n_train} train / {p.n_test} test {unit}, "
                 f"{len(self.jobs)} models, max {p.max_parallel} in parallel")
        for w in p.warnings:
            self.log("WARNING " + w, logging.WARNING)
        self._launch_pending()

    @property
    def running(self) -> List[Job]:
        return [j for j in self.jobs.values() if j.state == TRAINING]

    @property
    def finished(self) -> bool:
        return self._started and all(j.state in TERMINAL for j in self.jobs.values())

    def _payload(self, job: Job, n_jobs: int) -> dict:
        p = self.plan
        return {"key": job.key, "label": job.label, "file_stem": job.file_stem, "task": p.task,
                "run_id": p.run_id, "seed": p.seed, "options": p.options, "n_jobs": n_jobs,
                "split_path": str(p.split_path), "trained_dir": str(p.dirs.trained_models),
                "target": p.target, "dataset": p.dataset_info, **p.extra}

    def _launch_pending(self) -> None:
        if self.cancelled:
            return
        p = self.plan
        free = p.max_parallel - len(self.running)
        n_jobs = max(1, p.cpu_budget // max(1, min(p.max_parallel, len(self.jobs))))
        for k in self._order:
            if free <= 0:
                break
            job = self.jobs[k]
            if job.state != PENDING or job.process is not None:
                continue
            if p.kind == "emg":
                from .emg import run_emg_task as run_task
            else:
                from .worker import run_task
            proc = self.ctx.Process(target=run_task, args=(self._payload(job, n_jobs), self.queue),
                                    name=f"train-{k}", daemon=False)
            job.process, job.t_start, job.state, job.message = proc, time.time(), TRAINING, "starting"
            proc.start()
            self.log(f"[{job.label}] started (pid {proc.pid}, {n_jobs} threads)")
            free -= 1

    # --------------------------------------------------------------------------------------- polling
    def poll(self) -> None:
        """Drain worker messages, detect dead workers, start queued ones. Never blocks."""
        self._drain()
        for job in list(self.jobs.values()):
            proc = job.process
            if job.state == TRAINING and proc is not None and not proc.is_alive():
                self._drain()                                   # a final message may still be in the pipe
                if job.state == TRAINING:
                    self._finish(job, FAILED, f"worker process died unexpectedly (exit code {proc.exitcode})")
        self._launch_pending()
        if self.finished and self.t_end is None:
            self.t_end = time.time()
            n_ok = sum(j.state == COMPLETED for j in self.jobs.values())
            self.log(f"Run finished in {self.t_end - self.t_start:.1f}s: {n_ok}/{len(self.jobs)} models completed")
            shutil.rmtree(self.plan.split_path if self.plan.kind == "emg" else self.plan.split_path.parent, ignore_errors=True)
            self._logger.removeHandler(self._fh)
            self._fh.close()

    def _drain(self) -> None:
        while True:
            try:
                msg = self.queue.get_nowait()
            except queue_mod.Empty:
                return
            except Exception:                                   # noqa: BLE001 - broken pipe on shutdown
                return
            job = self.jobs.get(msg.get("key"))
            if job is None or job.state in TERMINAL:
                continue
            kind = msg["type"]
            if kind == "progress":
                job.pct, job.message = float(msg["pct"]), msg.get("msg", "")
                job.indeterminate = bool(msg.get("indeterminate", False))
            elif kind == "log":
                self.log(f"[{job.label}] {msg['msg']}")
            elif kind == "result":
                job.result = msg["result"]
                job.pct, job.indeterminate = 100.0, False
                self._finish(job, COMPLETED, "exported")
            elif kind == "failed":
                self.log(f"[{job.label}] FAILED: {msg['error']}\n{msg.get('traceback', '')}", logging.ERROR)
                self._finish(job, FAILED, msg["error"], msg.get("traceback", ""))

    def _finish(self, job: Job, state: str, message: str, tb: str = "") -> None:
        job.state, job.t_end, job.message, job.indeterminate = state, time.time(), message, False
        if state == FAILED:
            job.error = message
        proc = job.process
        if proc is not None:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
        if state == COMPLETED:
            self.log(f"[{job.label}] completed in {job.elapsed:.1f}s -> {job.result['artifact']}")
        elif state == FAILED:
            self.log(f"[{job.label}] failed: {message}", logging.ERROR)

    # ------------------------------------------------------------------------------------ cancelling
    def cancel(self, key: str) -> None:
        job = self.jobs[key]
        if job.state in TERMINAL:
            return
        proc = job.process
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        self._remove_partial(job)
        job.state, job.t_end, job.message, job.indeterminate = CANCELLED, time.time(), "cancelled by user", False
        self.log(f"[{job.label}] cancelled")

    def cancel_all(self) -> None:
        self.cancelled = True
        for k in self._order:
            self.cancel(k)
        self.poll()

    def _remove_partial(self, job: Job) -> None:
        """A model killed mid-export must not leave a truncated .joblib behind."""
        if job.result:
            return
        for f in self.plan.dirs.trained_models.glob(f"{job.file_stem}_{self.plan.run_id}.*"):
            try:
                f.unlink()
            except OSError:
                pass

    def shutdown(self) -> None:
        """Kill everything (window closed)."""
        for j in self.jobs.values():
            if j.process is not None and j.process.is_alive():
                j.process.kill()

    # ------------------------------------------------------------------------------------- results
    def results(self) -> List[dict]:
        return [j.result for j in self.jobs.values() if j.state == COMPLETED and j.result]
