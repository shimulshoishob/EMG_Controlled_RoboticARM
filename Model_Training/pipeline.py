"""Turns recordings + a split into model-ready arrays, fitting every transform on training data only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import ExperimentConfig
from .data_io import Recording
from .features import FeatureExtractor
from .preprocessing import (FeatureScaler, RawNormalizer, WindowPreprocessor, WindowSet, build_windows, window_starts)
from .splits import Piece
from .utils import LOG


def filter_rare_classes(recs: List[Recording], cfg: ExperimentConfig) -> List[Recording]:
    """Drop gestures that would produce fewer than ``min_windows_per_class`` windows (cannot be evaluated)."""
    win, stride = cfg.window_samples, cfg.stride_samples
    counts: Dict[str, int] = {}
    for r in recs:
        for a, b, lab in r.runs:
            counts[lab] = counts.get(lab, 0) + len(window_starts(b - a, win, stride))
    rare = {k for k, v in counts.items() if v < cfg.data.min_windows_per_class}
    for k in sorted(rare):
        LOG.warning("dropping gesture '%s': only %d windows (< %d)", k, counts[k], cfg.data.min_windows_per_class)
    out = []
    for r in recs:
        r.runs = [x for x in r.runs if x[2] not in rare]
        if r.runs:
            out.append(r)
    return out


@dataclass
class Partition:
    y: np.ndarray                                  # int labels
    meta: Dict[str, np.ndarray]                    # subject / session / rec per window
    raw: np.ndarray                                # (N, C, T) normalised windows
    feat: Optional[np.ndarray]                     # (N, F) standardised features
    windows: Optional[np.ndarray] = None           # (N, T, C) unprocessed signal (benchmark input)
    seg: Optional[np.ndarray] = None               # segment id per window (windows are chronological inside a segment)

    def data(self, input_kind: str) -> dict:
        return {"X": self.raw if input_kind == "raw" else self.feat, "y": self.y}


@dataclass
class Prepared:
    classes: List[str]
    fs: float
    n_channels: int
    n_samples: int
    preprocessor: WindowPreprocessor
    extractor: Optional[FeatureExtractor]
    feat_scaler: Optional[FeatureScaler]
    raw_norm: RawNormalizer
    parts: Dict[str, Partition] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.extractor.n_features if self.extractor else 0


def prepare(recs: Sequence[Recording], pieces: Dict[str, List[Piece]], classes: List[str],
            cfg: ExperimentConfig, need_features: bool = True, keep_windows: Sequence[str] = ()) -> Prepared:
    """Build windows per split, preprocess, extract features, fit normalisers on ``pieces['train']`` only."""
    win, stride = cfg.window_samples, cfg.stride_samples
    fs = recs[0].fs
    C = recs[0].signal.shape[1]
    pre = WindowPreprocessor(fs, cfg.preprocess)
    ext = FeatureExtractor(fs, C, win, cfg.features) if need_features else None
    cidx = {c: i for i, c in enumerate(classes)}

    wsets: Dict[str, WindowSet] = {}
    proc: Dict[str, np.ndarray] = {}
    for name, plist in pieces.items():
        ws = build_windows(recs, [p.seg for p in plist], win, stride)
        wsets[name] = ws
        proc[name] = pre(ws.X) if len(ws) else ws.X
    if len(wsets["train"]) == 0:
        raise ValueError("training split produced no windows")

    raw_norm = RawNormalizer().fit_windows(proc["train"])
    feats = {n: ext.transform(p) if (ext and len(p)) else None for n, p in proc.items()}
    scaler = FeatureScaler().fit_features(feats["train"]) if ext else None

    parts = {}
    for name, ws in wsets.items():
        if len(ws) == 0:
            continue
        parts[name] = Partition(
            y=np.array([cidx[l] for l in ws.y], dtype=np.int64),
            meta={"subject": ws.subject, "session": ws.session, "rec": ws.rec},
            raw=raw_norm.transform_windows(proc[name]),
            feat=scaler.transform_features(feats[name]) if ext else None,
            windows=ws.X if name in keep_windows else None, seg=ws.seg)
    return Prepared(classes, fs, C, win, pre, ext, scaler, raw_norm, parts)
