"""Output directory layout (created on demand)::

    Model_Training/
    ├── datasets/                      user datasets (the file picker starts here)
    ├── outputs/
    │   ├── trained_models/            <Model>_<run_id>.joblib (+ .json metadata)
    │   ├── evaluation_reports/<run>/  per-model JSON, summary.csv/json, report.md
    │   ├── plots/<run>/               comparison + confusion-matrix PNGs
    │   └── .work/<run>/               temporary train/test split (deleted after the run)
    └── logs/                          training_<run>.log
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../Model_Training


@dataclass(frozen=True)
class Dirs:
    root: Path = ROOT

    @property
    def datasets(self) -> Path: return self.root / "datasets"
    @property
    def outputs(self) -> Path: return self.root / "outputs"
    @property
    def trained_models(self) -> Path: return self.outputs / "trained_models"
    @property
    def reports(self) -> Path: return self.outputs / "evaluation_reports"
    @property
    def plots(self) -> Path: return self.outputs / "plots"
    @property
    def work(self) -> Path: return self.outputs / ".work"
    @property
    def logs(self) -> Path: return self.root / "logs"

    def ensure(self) -> "Dirs":
        for p in (self.datasets, self.trained_models, self.reports, self.plots, self.work, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self
