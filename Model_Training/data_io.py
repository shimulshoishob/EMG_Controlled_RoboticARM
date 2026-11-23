"""Dataset discovery, loading, validation and cleaning.

Nothing here assumes one fixed layout.  A recording can be

* a CSV/TSV/TXT/Parquet table with channel columns (``emg1..emg8``, ``ch1..``, or anything given in
  the config), optionally with a time column and per-sample ``label`` / ``subject`` / ``session`` columns, or
* a ``.npy`` array (samples x channels).

Gesture / subject / session are taken from columns when present, otherwise from the file name
(the BioWave app names raw recordings ``emg_<date>_<time>_<label>.csv`` where ``<label>`` is free text such as
``S02_sess1_closed_fist_trial3``).  Files without a usable label are reported and skipped - never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DataConfig, resolve_path
from .utils import LOG

_LABEL_CANDIDATES = ("label", "gesture", "class", "target", "movement", "y")
_TIME_CANDIDATES = ("host_time_s", "time_s", "time", "timestamp", "t", "seconds")
_NON_CHANNEL = re.compile(r"(time|stamp|seq|frame|id$|index|roll|pitch|yaw|label|class|gesture|target|subject|session|trial|unnamed)", re.I)
_SUBJECT = re.compile(r"^(s|subj|subject|p|part|participant)\d+$", re.I)
_SESSION = re.compile(r"^(sess|session|day|d)\d+$", re.I)
_TRIAL = re.compile(r"^(t|trial|rep|r|run)\d+$", re.I)


@dataclass
class Recording:
    rec_id: str
    path: str
    signal: np.ndarray                     # (samples, channels) float32
    fs: float
    subject: str
    session: str
    runs: List[Tuple[int, int, str]]       # (start, stop, label) contiguous constant-label stretches
    channel_names: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.signal.shape[0]


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def normalize_label(label: str, aliases: Dict[str, str] | None = None) -> str:
    lab = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
    if aliases:
        amap = {re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_"): v for k, v in aliases.items()}
        lab = amap.get(lab, lab)
    return lab


def discover_files(cfg: DataConfig) -> List[Path]:
    files: List[Path] = []
    for p in cfg.paths:
        root = resolve_path(p)
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            for pat in cfg.patterns:
                files += list(root.rglob(pat) if cfg.recursive else root.glob(pat))
        else:
            LOG.warning("Data path does not exist: %s", root)
    return sorted({f.resolve() for f in files if f.is_file()})


def parse_filename(stem: str, cfg: DataConfig) -> dict:
    """Extract subject / session / gesture from a file name (see module docstring)."""
    out = {"subject": None, "session": None, "gesture": None}
    if cfg.filename_pattern:
        m = re.search(cfg.filename_pattern, stem)
        if m:
            for k in out:
                if k in m.groupdict() and m.group(k):
                    out[k] = m.group(k)
        return out
    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]
    gesture = []
    for i, tok in enumerate(tokens):
        if i == 0 and tok.lower() == "emg":
            continue
        if re.fullmatch(r"\d{8}", tok) or re.fullmatch(r"\d{6}", tok):     # date / time stamp
            continue
        if _SUBJECT.match(tok) and out["subject"] is None:
            out["subject"] = tok
        elif _SESSION.match(tok) and out["session"] is None:
            out["session"] = tok
        elif _TRIAL.match(tok):
            continue
        elif tok.isdigit():
            continue
        else:
            gesture.append(tok)
    out["gesture"] = "_".join(gesture) if gesture else None
    return out


def _read_table(path: Path):
    import pandas as pd
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f".npy must be 2-D (samples x channels), got {arr.shape}")
        return pd.DataFrame(arr, columns=[f"ch{i + 1}" for i in range(arr.shape[1])])
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python")      # sniff the delimiter (tsv / txt / odd csv)


def _pick_channels(df, cfg: DataConfig) -> List[str]:
    if cfg.channel_columns:
        missing = [c for c in cfg.channel_columns if c not in df.columns]
        if missing:
            raise ValueError(f"channel columns missing: {missing}")
        cols = list(cfg.channel_columns)
    else:
        rx = re.compile(cfg.channel_regex, re.I)
        cols = [c for c in df.columns if rx.match(str(c).strip())]
        if not cols:                                          # fallback: numeric, non-metadata columns
            cols = [c for c in df.columns if not _NON_CHANNEL.search(str(c)) and
                    np.issubdtype(df[c].dtype, np.number)]
        cols = sorted(cols, key=lambda c: natural_key(str(c)))
    if cfg.n_channels:
        cols = cols[: cfg.n_channels]
    if not cols:
        raise ValueError("no channel columns found (set data.channel_columns)")
    return cols


def _find_col(df, explicit: Optional[str], candidates) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"column '{explicit}' not found")
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _label_runs(labels: np.ndarray) -> List[Tuple[int, int, str]]:
    runs, start = [], 0
    n = len(labels)
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            if labels[start] not in (None, ""):
                runs.append((start, i, labels[start]))
            start = i
    return runs


def load_recording(path: Path, cfg: DataConfig) -> Tuple[Optional[Recording], Optional[str]]:
    """Load one file. Returns (recording, None) or (None, reason-for-skipping)."""
    import pandas as pd
    try:
        df = _read_table(path)
    except Exception as exc:
        return None, f"unreadable ({exc})"
    try:
        chans = _pick_channels(df, cfg)
    except ValueError as exc:
        return None, str(exc)

    sig_df = df[chans].apply(pd.to_numeric, errors="coerce")
    sig_df = sig_df.replace([np.inf, -np.inf], np.nan)
    nan_frac = float(sig_df.isna().to_numpy().mean()) if len(sig_df) else 1.0
    if len(sig_df) == 0:
        return None, "empty file"
    if nan_frac > cfg.max_nan_fraction:
        return None, f"{nan_frac:.0%} missing/non-numeric values (limit {cfg.max_nan_fraction:.0%})"
    if nan_frac > 0:
        valid = ~sig_df.isna().any(axis=1)                   # remember rows where data was invented
        sig_df = sig_df.interpolate(limit_direction="both")
    signal = sig_df.to_numpy(dtype=np.float32)

    # sampling rate
    fs = float(cfg.sampling_rate)
    tcol = _find_col(df, cfg.time_column, _TIME_CANDIDATES)
    if tcol is not None and cfg.auto_detect_sampling_rate:
        t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=np.float64)
        dt = np.diff(t[np.isfinite(t)])
        dt = dt[dt > 0]
        if dt.size > 10:
            est = 1.0 / float(np.median(dt))
            if est > 20000:                                   # time given in milliseconds
                est /= 1000.0
            if abs(est - fs) / fs > 0.05:
                LOG.warning("%s: estimated sampling rate %.1f Hz differs from configured %.1f Hz - resampling",
                            path.name, est, fs)
                n_new = int(round(signal.shape[0] / est * fs))
                old_x = np.linspace(0.0, 1.0, signal.shape[0])
                new_x = np.linspace(0.0, 1.0, n_new)
                signal = np.stack([np.interp(new_x, old_x, signal[:, c]) for c in range(signal.shape[1])], 1).astype(np.float32)
                df = df.iloc[np.clip(np.round(new_x * (len(df) - 1)).astype(int), 0, len(df) - 1)].reset_index(drop=True)

    if signal.shape[0] < cfg.min_recording_s * fs:
        return None, f"too short ({signal.shape[0] / fs:.2f}s < {cfg.min_recording_s}s)"
    stds = signal.std(axis=0)
    if np.any(stds < 1e-6):
        return None, f"flat (dead) channel(s): {[chans[i] for i in np.where(stds < 1e-6)[0]]}"

    # metadata: columns first, then filename
    meta = parse_filename(path.stem, cfg)
    scol = _find_col(df, cfg.subject_column, ()) if cfg.subject_column else None
    xcol = _find_col(df, cfg.session_column, ()) if cfg.session_column else None
    subject = str(df[scol].mode().iloc[0]) if scol else (meta["subject"] or cfg.default_subject)
    session = str(df[xcol].mode().iloc[0]) if xcol else (meta["session"] or cfg.default_session)

    lcol = _find_col(df, cfg.label_column, _LABEL_CANDIDATES)
    n = signal.shape[0]
    trim = int(round(cfg.trim_s * fs))
    if lcol is not None:
        raw = df[lcol].astype(object).where(df[lcol].notna(), None).to_numpy()
        labels = np.array([normalize_label(v, cfg.gesture_aliases) if v is not None and str(v).strip() else None
                           for v in raw], dtype=object)
        runs = _label_runs(labels)
    elif meta["gesture"]:
        runs = [(0, n, normalize_label(meta["gesture"], cfg.gesture_aliases))]
    else:
        return None, "no label (no label column and no gesture in the file name)"
    runs = [(a + trim, b - trim, lab) for a, b, lab in runs if (b - trim) - (a + trim) >= 8]
    if cfg.gesture_classes:
        keep = {normalize_label(g, cfg.gesture_aliases) for g in cfg.gesture_classes}
        runs = [r for r in runs if r[2] in keep]
    if not runs:
        return None, "no labelled samples left after filtering/trimming"
    return Recording(rec_id=path.stem, path=str(path), signal=signal, fs=fs, subject=subject,
                     session=session, runs=runs, channel_names=[str(c) for c in chans]), None


def load_dataset(cfg: DataConfig) -> Tuple[List[Recording], dict]:
    """Load every supported file; returns (recordings, report).

    ``cfg.layout`` = ``sessions`` (or ``auto`` when session folders are found) uses ``session_io`` for BioWave
    session folders (signals/events/metadata/quality); otherwise every table file is one recording."""
    if cfg.layout in ("auto", "sessions"):
        from .session_io import find_sessions, load_sessions
        if cfg.layout == "sessions" or find_sessions(cfg.paths, cfg.recursive)[0]:
            return load_sessions(cfg)
    files = discover_files(cfg)
    report = {"files_found": len(files), "loaded": 0, "skipped": [], "paths": [str(resolve_path(p)) for p in cfg.paths]}
    recs: List[Recording] = []
    for f in files:
        rec, why = load_recording(f, cfg)
        if rec is None:
            report["skipped"].append({"file": f.name, "reason": why})
            LOG.warning("skipped %s: %s", f.name, why)
        else:
            recs.append(rec)
    # consistent channel layout: keep the most common channel count
    if recs:
        counts: Dict[int, int] = {}
        for r in recs:
            counts[r.signal.shape[1]] = counts.get(r.signal.shape[1], 0) + 1
        modal = max(counts, key=counts.get)
        good = []
        for r in recs:
            if r.signal.shape[1] != modal:
                report["skipped"].append({"file": Path(r.path).name, "reason": f"{r.signal.shape[1]} channels, dataset has {modal}"})
            else:
                good.append(r)
        recs = good
    # unique ids
    seen: Dict[str, int] = {}
    for r in recs:
        k = seen.get(r.rec_id, 0)
        seen[r.rec_id] = k + 1
        if k:
            r.rec_id = f"{r.rec_id}#{k}"
    report["loaded"] = len(recs)
    return recs, report


def summarize_dataset(recs: List[Recording]) -> dict:
    classes: Dict[str, float] = {}
    for r in recs:
        for a, b, lab in r.runs:
            classes[lab] = classes.get(lab, 0.0) + (b - a) / r.fs
    return {
        "n_recordings": len(recs),
        "n_channels": int(recs[0].signal.shape[1]) if recs else 0,
        "channel_names": recs[0].channel_names if recs else [],
        "sampling_rate": float(recs[0].fs) if recs else None,
        "subjects": sorted({r.subject for r in recs}),
        "sessions": sorted({f"{r.subject}|{r.session}" for r in recs}),
        "total_seconds": float(sum(r.n_samples / r.fs for r in recs)),
        "seconds_per_class": {k: round(v, 2) for k, v in sorted(classes.items())},
    }
