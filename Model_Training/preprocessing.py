"""Window-level preprocessing, sliding-window segmentation and fit-on-train normalisers.

Design rule: every operation that touches signal values is applied to *windows* (never across the whole
recording), so the exact same code runs in training and in the real-time pipeline (no train/deploy skew),
and statistics (mean/std) are only ever fitted on training windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos

from .config import PreprocessConfig
from .data_io import Recording


@dataclass(frozen=True)
class Segment:
    """A contiguous, single-label stretch of one recording: samples [start, stop)."""
    rec_idx: int
    start: int
    stop: int
    label: str


class WindowPreprocessor:
    """DC removal + band-pass (+ optional notch) applied independently to each window."""

    def __init__(self, fs: float, cfg: PreprocessConfig):
        self.fs = float(fs)
        self.cfg = cfg
        nyq = self.fs / 2.0
        self.sos = None
        bp = cfg.bandpass_hz
        if bp:
            lo, hi = bp
            lo = float(lo) if lo else None
            hi = min(float(hi), nyq * 0.95) if hi else None
            if lo and hi:
                self.sos = butter(cfg.filter_order, [lo, hi], btype="band", fs=self.fs, output="sos")
            elif lo:
                self.sos = butter(cfg.filter_order, lo, btype="high", fs=self.fs, output="sos")
            elif hi:
                self.sos = butter(cfg.filter_order, hi, btype="low", fs=self.fs, output="sos")
        self.notch = None
        if cfg.notch_hz and cfg.notch_hz < nyq:
            sections = []
            for k in range(1, max(1, int(cfg.notch_harmonics)) + 1):
                f0 = cfg.notch_hz * k
                if f0 < nyq * 0.98:
                    b, a = iirnotch(f0, Q=float(cfg.notch_q), fs=self.fs)
                    sections.append(tf2sos(b, a))
            self.notch = np.concatenate(sections) if sections else None

    def _apply(self, sos, x):
        # padlen of sosfiltfilt is ~3 * (2 * sections + 1); fall back to causal filtering for tiny windows
        need = 3 * (2 * sos.shape[0] + 1) + 1
        if x.shape[1] > need:
            return sosfiltfilt(sos, x, axis=1)
        return sosfilt(sos, x, axis=1)

    def __call__(self, windows: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """windows: (N, T, C) -> float32 (N, T, C).  Processed in chunks (identical result, bounded memory)."""
        windows = np.asarray(windows)
        if len(windows) > chunk:
            return np.concatenate([self(windows[i:i + chunk], chunk) for i in range(0, len(windows), chunk)])
        x = np.asarray(windows, dtype=np.float64)
        if self.cfg.remove_dc:
            x = x - x.mean(axis=1, keepdims=True)
        if self.sos is not None:
            x = self._apply(self.sos, x)
        if self.notch is not None:
            x = self._apply(self.notch, x)
        return x.astype(np.float32)

    def describe(self) -> dict:
        return {"fs": self.fs, **self.cfg.__dict__}


def window_starts(n: int, win: int, stride: int) -> np.ndarray:
    if n < win:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n - win + 1, stride, dtype=np.int64)


@dataclass
class WindowSet:
    X: np.ndarray            # (N, T, C) raw (unfiltered) float32
    y: np.ndarray            # (N,) str labels
    subject: np.ndarray
    session: np.ndarray
    rec: np.ndarray
    seg: np.ndarray          # index of the originating segment (all windows of a segment stay together)

    def __len__(self):
        return len(self.y)


def build_windows(recs: Sequence[Recording], segments: Sequence[Segment], win: int, stride: int) -> WindowSet:
    """Slide a window inside every segment; a window never crosses a segment (label / split) boundary."""
    xs, ys, subj, sess, rec, seg = [], [], [], [], [], []
    for si, s in enumerate(segments):
        r = recs[s.rec_idx]
        starts = window_starts(s.stop - s.start, win, stride)
        if starts.size == 0:
            continue
        chunk = r.signal[s.start:s.stop]
        idx = starts[:, None] + np.arange(win)[None, :]
        xs.append(chunk[idx])                                   # (n, win, C)
        n = len(starts)
        ys += [s.label] * n
        subj += [r.subject] * n
        sess += [f"{r.subject}|{r.session}"] * n
        rec += [r.rec_id] * n
        seg += [si] * n
    if not xs:
        c = recs[0].signal.shape[1] if recs else 1
        return WindowSet(np.empty((0, win, c), np.float32), *(np.empty(0, dtype=object) for _ in range(5)))
    return WindowSet(np.concatenate(xs).astype(np.float32), np.array(ys, dtype=object),
                     np.array(subj, dtype=object), np.array(sess, dtype=object),
                     np.array(rec, dtype=object), np.array(seg))


class Standardizer:
    """Per-feature (or per-channel) z-score; fit on training data only."""

    def __init__(self, mean=None, scale=None):
        self.mean = mean
        self.scale = scale

    def fit(self, X: np.ndarray, axes) -> "Standardizer":
        self.mean = X.mean(axis=axes, keepdims=False).astype(np.float32)
        s = X.std(axis=axes, keepdims=False).astype(np.float32)
        self.scale = np.where(s < 1e-8, 1.0, s).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale).astype(np.float32)


class RawNormalizer(Standardizer):
    """Per-channel normalisation of preprocessed windows; output layout (N, C, T) for conv/recurrent nets."""

    def fit_windows(self, Xp: np.ndarray) -> "RawNormalizer":          # Xp: (N, T, C)
        self.fit(Xp, axes=(0, 1))
        return self

    def transform_windows(self, Xp: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.transform(Xp).transpose(0, 2, 1))


class FeatureScaler(Standardizer):
    def fit_features(self, F: np.ndarray) -> "FeatureScaler":
        self.fit(F, axes=0)
        return self

    def transform_features(self, F: np.ndarray) -> np.ndarray:
        return self.transform(F)
