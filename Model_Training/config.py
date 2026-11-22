"""Experiment configuration.

Everything that influences an experiment lives in one nested dataclass (``ExperimentConfig``)
that can be loaded from / saved to JSON, so every result directory carries the exact settings that
produced it.  Use ``python model_train.py --dump-config`` to get an editable template.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent          # .../Model_Training
PROJECT_ROOT = HERE.parent                      # .../EMG_Controlled_RoboticARM


@dataclass
class DataConfig:
    # Where recordings live. Relative paths are tried against the CWD, the project root and this folder.
    paths: List[str] = field(default_factory=lambda: ["recordings"])
    patterns: List[str] = field(default_factory=lambda: ["*.csv", "*.tsv", "*.txt", "*.parquet", "*.npy"])
    recursive: bool = True
    # Column layout. ``None`` = detect automatically.
    channel_columns: Optional[List[str]] = None
    channel_regex: str = r"^(emg|ch|chan|channel|sensor)[ _-]?\d+$"
    n_channels: Optional[int] = None             # keep only the first N channels
    time_column: Optional[str] = None
    label_column: Optional[str] = None           # per-sample gesture label inside the file
    subject_column: Optional[str] = None
    session_column: Optional[str] = None
    # Labels from file names when there is no label column, e.g. emg_20261108_185041_S02_sess1_fist_trial3.csv
    # Optional regex with named groups (?P<subject>..) (?P<session>..) (?P<gesture>..) overrides the default rules.
    filename_pattern: Optional[str] = None
    default_subject: str = "S00"
    default_session: str = "sess1"
    sampling_rate: float = 500.0                 # Hz (the BioWave armband streams 500 Hz)
    auto_detect_sampling_rate: bool = True       # estimate from the time column, resample if it deviates >5 %
    gesture_classes: Optional[List[str]] = None  # keep only these (after aliasing); None = keep all found
    gesture_aliases: Dict[str, str] = field(default_factory=dict)   # e.g. {"fist": "closed_fist"}
    min_recording_s: float = 1.0
    max_nan_fraction: float = 0.10
    trim_s: float = 0.0                          # drop this many seconds at both ends of every labelled run
    min_windows_per_class: int = 20              # classes with fewer windows are dropped (with a warning)


@dataclass
class PreprocessConfig:
    window_ms: float = 200.0                     # same default as the BioWave app (100 samples @ 500 Hz)
    stride_ms: float = 50.0                      # 75 % overlap; also the real-time decision interval
    remove_dc: bool = True
    bandpass_hz: Optional[List[float]] = field(default_factory=lambda: [20.0, 220.0])
    filter_order: int = 4
    notch_hz: Optional[float] = None             # 50 or 60 for mains hum; None = off
    # Filtering is applied per window (zero-phase) so training and real-time inference share one code path.


@dataclass
class FeatureConfig:
    time_domain: bool = True
    frequency_domain: bool = True
    cross_channel: bool = True                   # RMS ratios + channel correlations
    freq_range_hz: List[float] = field(default_factory=lambda: [20.0, 220.0])
    bands_hz: List[List[float]] = field(default_factory=lambda: [[20.0, 60.0], [60.0, 120.0], [120.0, 220.0]])
    zc_ssc_rel_threshold: float = 0.05           # thresholds relative to the window's std (scale invariant)
    wamp_rel_threshold: float = 0.25


@dataclass
class SplitConfig:
    # auto | subject | session | recording | temporal   (never a random window split: overlapping windows leak)
    mode: str = "auto"
    val_size: float = 0.15
    test_size: float = 0.20
    seed: int = 42


@dataclass
class CVConfig:
    enabled: bool = True
    folds: int = 3
    scope: str = "ml"                            # ml | all | none  (deep models are slow to cross-validate)


@dataclass
class TuningConfig:
    enabled: bool = False
    models: List[str] = field(default_factory=list)   # empty = every model that defines a search space
    n_iter: int = 6
    folds: int = 3


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"                         # auto | cpu | cuda
    n_jobs: int = 4                              # classical-ML worker threads during training
    torch_threads: int = 4                       # CPU threads for torch during training
    epochs: int = 40
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 8
    dropout: float = 0.3
    augment: bool = True                         # light gaussian noise + gain jitter on training batches
    class_weight: bool = True                    # balanced loss / class weights for imbalance
    max_train_seconds: Optional[float] = 900.0   # per-model wall-clock cap for deep models
    cv_epochs: int = 12                          # fixed epochs for deep models when scope == "all"
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # per-model overrides


@dataclass
class ModelsConfig:
    enabled: Any = "all"                         # "all" or a list of model keys
    families: List[str] = field(default_factory=lambda: ["ml", "dl", "hybrid"])
    exclude: List[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    threads: int = 1                             # CPU threads at inference (single-thread ~ embedded laptop)
    warmup: int = 20
    n_single: int = 200
    batch_sizes: List[int] = field(default_factory=lambda: [32])
    load_repeats: int = 3
    budget_ms: Optional[float] = None            # real-time budget; None = the stride (decision interval)
    comfortable_fraction: float = 0.25           # p95 <= 25 % of budget -> "comfortable"


@dataclass
class RankingConfig:
    f1_tolerance: float = 0.05                   # "lightweight"/"fastest" must stay within this F1 of the best
    perf_weights: Dict[str, float] = field(default_factory=lambda: {"f1_macro": 0.5, "balanced_accuracy": 0.25, "accuracy": 0.25})
    rt_weights: Dict[str, float] = field(default_factory=lambda: {"f1_macro": 0.60, "latency": 0.25, "size": 0.15})


@dataclass
class OutputConfig:
    root: str = "results"
    name: Optional[str] = None                   # None -> experiment_001, 002, ...
    save_models: bool = True
    top_n_radar: int = 5


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        cfg = cls()
        _merge(cfg, d or {})
        return cfg

    @classmethod
    def load(cls, path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def copy(self) -> "ExperimentConfig":
        return copy.deepcopy(self)

    # -- derived values -------------------------------------------------------------------------
    @property
    def window_samples(self) -> int:
        return max(8, int(round(self.preprocess.window_ms * self.data.sampling_rate / 1000.0)))

    @property
    def stride_samples(self) -> int:
        return max(1, int(round(self.preprocess.stride_ms * self.data.sampling_rate / 1000.0)))

    @property
    def budget_ms(self) -> float:
        return float(self.benchmark.budget_ms or self.preprocess.stride_ms)


def _merge(obj, d: dict, path: str = "") -> None:
    """Recursively copy ``d`` onto a dataclass, rejecting unknown keys (typos should be loud)."""
    names = {f.name for f in dataclasses.fields(obj)}
    for key, value in d.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{path}{key}'. Valid keys here: {sorted(names)}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, path + key + ".")
        else:
            setattr(obj, key, value)


def resolve_path(p: str) -> Path:
    """Resolve a user path: absolute as-is, else CWD, project root, then this folder."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), PROJECT_ROOT, HERE):
        cand = (base / path)
        if cand.exists():
            return cand.resolve()
    return (PROJECT_ROOT / path).resolve()
