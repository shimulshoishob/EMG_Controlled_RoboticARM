"""Loader for BioWave *session folders*.

A session folder (one recording) looks like::

    S102_day01_right_20261122T091707Z/
        signals_<id>.csv        one row per sample: Ch1..Ch8 (raw ADC), IMU_*, Valid, Phase, Label, Label_Code, Hand, Trial_ID ...
        events_<id>.csv         cue log (prepare / perform / rest per trial)
        metadata_<id>.json|txt  participant, protocol, channel map, sampling rate, labels ...
        quality_<id>.json       packet loss, gaps, jitter
        raw_<id>.bin            raw frames (not read)

What is done here (and why):

* the EMG channels are taken from the metadata ``channel_map`` (type ``emg``) - IMU angles are never fed to the models;
* rows with ``Valid == 0`` (lost packets, NaN signal) are **masked, not interpolated**: no window may contain them;
* only the phases in ``cfg.use_phases`` (``perform`` and ``rest``) get a label; the ``prepare`` cue, ``paused``,
  ``switch_hand`` and ``end`` rows never reach the models;
* every labelled stretch is one *run* (one trial / one phase); reaction time at the start and relaxation at the end are trimmed
  (``perform_trim_s`` / ``rest_trim_s``) because the subject is not yet / no longer performing the gesture there;
* ``rest`` is by far the most frequent state (after every gesture), so rest runs are thinned to ``rest_cap_ratio`` x the mean gesture
  duration - class balance is fixed in the data instead of hoping the loss function copes;
* subject = ``participant_id`` (so left- and right-hand sessions of one person always stay on the same side of a subject split).

The result is a list of ``data_io.Recording`` objects, i.e. exactly what the rest of the pipeline (splits, windows, features, models,
benchmark) already consumes.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DataConfig, resolve_path
from .data_io import Recording, normalize_label
from .utils import LOG

STRUCTURAL = {"prepare", "paused", "settle", "aborted", "switch_hand", "end", "", "nan", "none"}


# ------------------------------------------------------------------------------------------ discovery
def find_sessions(paths: List[str], recursive: bool = True) -> Tuple[List[Path], List[dict]]:
    """Returns (session folders that contain a signals CSV, [{folder, reason}] for folders that cannot be used)."""
    sessions: List[Path] = []
    skipped: List[dict] = []
    seen = set()
    for p in paths:
        root = resolve_path(p)
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            continue
        cands = [root] + ([d for d in sorted(root.rglob("*")) if d.is_dir()] if recursive else
                          [d for d in sorted(root.iterdir()) if d.is_dir()])
        for d in cands:
            if d in seen or d.name.startswith("."):
                continue
            seen.add(d)
            if any(d.glob("signals_*.csv")):
                sessions.append(d)
            elif any(d.glob("raw_*.bin")):
                skipped.append({"file": d.name, "reason": "only a raw_*.bin frame dump (no signals CSV) - export it from the recorder first"})
            elif d != root and any(d.glob("*.csv")) and not any(d.glob("events_*.csv")):
                skipped.append({"file": d.name, "reason": "legacy recording format (Timestamp_ms/Packet_Number, 11 channels) - not a session folder"})
    return sessions, skipped


def has_sessions(paths: List[str]) -> bool:
    return bool(find_sessions(paths)[0])


# ------------------------------------------------------------------------------------------ metadata
def read_metadata(folder: Path) -> dict:
    """Merge metadata_*.txt (key=value) and metadata_*.json; missing files are fine."""
    meta: dict = {}
    for f in folder.glob("metadata_*.txt"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    js = {}
    for f in folder.glob("metadata_*.json"):
        try:
            js = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            LOG.warning("%s: unreadable metadata JSON (%s)", folder.name, exc)
    qual = {}
    for f in folder.glob("quality_*.json"):
        try:
            qual = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    emg_cols = [c["column"] for c in js.get("channel_map", []) if c.get("type") == "emg"]
    label_codes = {}
    for item in meta.get("label_codes", "").split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            try:
                label_codes[normalize_label(k)] = int(v)
            except ValueError:
                pass
    contact = (js.get("channel_contact") or {}).get("channels", [])
    return {
        "subject": js.get("participant_id") or meta.get("participant_id"),
        "day": str((js.get("session_info") or {}).get("session_day") or meta.get("session_day") or ""),
        "fs": (js.get("device") or {}).get("sampling_rate_hz") or meta.get("sample_rate_hz") or qual.get("sample_rate_hz"),
        "emg_columns": emg_cols,
        "label_codes": label_codes,
        "repeats": meta.get("repeats"),
        "randomized": meta.get("randomize_order"),
        "practice": meta.get("practice_mode"),
        "poor_channels": [c["channel"] for c in contact if str(c.get("grade", "")).upper() in ("POOR", "NO SIGNAL")],
        "packet_loss_pct": (qual.get("transport") or {}).get("loss_pct_packets"),
        "valid_sample_ratio": qual.get("valid_sample_ratio"),
    }


# ------------------------------------------------------------------------------------------ one session
def _runs_from_keys(key: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous stretches of identical integer ``key``."""
    if len(key) == 0:
        return []
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(key)]])
    return list(zip(starts.tolist(), stops.tolist()))


def load_session(folder: Path, cfg: DataConfig) -> Tuple[Optional[Recording], dict]:
    """Load one session folder -> (Recording | None, info row for the GUI / report)."""
    import pandas as pd
    folder = Path(folder)
    info: dict = {"session": folder.name, "status": "skipped", "reason": ""}
    sig_files = sorted(folder.glob("signals_*.csv"))
    if not sig_files:
        info["reason"] = "no signals_*.csv"
        return None, info
    meta = read_metadata(folder)
    head = pd.read_csv(sig_files[0], nrows=2)
    emg = meta["emg_columns"] or [c for c in head.columns if re.fullmatch(r"Ch\d+", str(c))][:8]
    emg = [c for c in emg if c in head.columns]
    needed = {"Valid", "Phase", "Label"}
    missing = needed - set(head.columns)
    if not emg or missing:
        info["reason"] = f"missing columns: {sorted(missing) or 'EMG channels'}"
        return None, info
    use = emg + ["Valid", "Phase", "Label"] + [c for c in ("Trial_ID", "Hand") if c in head.columns]
    try:
        df = pd.read_csv(sig_files[0], usecols=use, dtype={c: np.float32 for c in emg})
    except Exception as exc:
        info["reason"] = f"unreadable ({exc})"
        return None, info

    fs = float(meta["fs"] or cfg.sampling_rate)
    n = len(df)
    signal = df[emg].to_numpy(dtype=np.float32)
    valid = (df["Valid"].to_numpy() == 1) & np.isfinite(signal).all(axis=1)
    valid_ratio = float(valid.mean()) if n else 0.0
    subject = str(meta["subject"] or (re.match(r"(S\d+)", folder.name) or [None, cfg.default_subject])[1])
    hand = str(df["Hand"].mode().iloc[0]).lower() if "Hand" in df and df["Hand"].notna().any() else "na"
    day = meta["day"] or (re.search(r"day0*(\d+)", folder.name, re.I) or [None, "1"])[1]
    info.update(subject=subject, hand=hand, day=day, rows=n, duration_s=round(n / fs, 1), fs=fs,
                valid_pct=round(100 * valid_ratio, 2), packet_loss_pct=meta["packet_loss_pct"],
                poor_channels=",".join(map(str, meta["poor_channels"])), n_channels=len(emg),
                repeats=meta["repeats"], randomized=meta["randomized"])
    if valid_ratio < cfg.min_valid_ratio:
        info["reason"] = f"only {valid_ratio:.1%} valid samples (limit {cfg.min_valid_ratio:.0%})"
        return None, info

    phase = df["Phase"].astype(str).str.lower().to_numpy()
    raw_label = df["Label"].astype(object).where(df["Label"].notna(), "").to_numpy()
    norm = {v: normalize_label(v, cfg.gesture_aliases) for v in set(raw_label.tolist())}
    label = np.array([norm[v] for v in raw_label], dtype=object)
    trial = df["Trial_ID"].fillna(-1).to_numpy(dtype=np.int64) if "Trial_ID" in df else np.zeros(n, np.int64)

    codes = meta["label_codes"]
    gesture_set = {k for k, v in codes.items() if v < 90} or {l for l in set(label.tolist()) if l not in STRUCTURAL}
    use_phases = {p.lower() for p in cfg.use_phases}
    keep_cls = {normalize_label(g, cfg.gesture_aliases) for g in cfg.gesture_classes} if cfg.gesture_classes else None

    ok = valid & np.isin(phase, list(use_phases)) & np.isin(label, list(gesture_set))
    if keep_cls is not None:
        ok &= np.isin(label, list(keep_cls))
    # one key per (usable?, phase, trial, label): a run never spans a phase change, a trial change, or a lost packet
    phase_codes = {p: i for i, p in enumerate(sorted(set(phase.tolist())))}
    label_codes_i = {l: i for i, l in enumerate(sorted(set(label.tolist())))}
    key = (ok.astype(np.int64) * 1_000_000_000
           + np.array([phase_codes[p] for p in phase], dtype=np.int64) * 10_000_000
           + trial.clip(-1, 99_999) * 100 + np.array([label_codes_i[l] for l in label], dtype=np.int64))
    runs: List[Tuple[int, int, str]] = []
    rest_runs: List[Tuple[int, int, str]] = []
    for a, b in _runs_from_keys(key):
        if not ok[a]:
            continue
        lab, ph = label[a], phase[a]
        t0, t1 = cfg.rest_trim_s if ph == "rest" else cfg.perform_trim_s
        a2, b2 = a + int(round(t0 * fs)), b - int(round(t1 * fs))
        if b2 - a2 < 8:
            continue
        (rest_runs if (ph == "rest" and lab == "rest") else runs).append((a2, b2, lab))

    # thin the (over-represented) relaxation periods that follow every gesture
    if rest_runs:
        per_class: Dict[str, float] = {}
        for a, b, lab in runs:
            if lab != "rest":
                per_class[lab] = per_class.get(lab, 0.0) + (b - a) / fs
        if cfg.rest_cap_ratio and per_class:
            have = sum((b - a) / fs for a, b, lab in runs if lab == "rest")       # explicit 'rest' gesture trials
            allowed = max(0.0, cfg.rest_cap_ratio * float(np.mean(list(per_class.values()))) - have)
            mean_len = float(np.mean([(b - a) / fs for a, b, _ in rest_runs]))
            k = int(max(1, min(len(rest_runs), round(allowed / max(mean_len, 1e-6)))))
            pick = np.unique(np.round(np.linspace(0, len(rest_runs) - 1, k)).astype(int))
            rest_runs = [rest_runs[i] for i in pick]
        runs += rest_runs
    runs.sort()

    classes = sorted({lab for _, _, lab in runs})
    secs: Dict[str, float] = {}
    for a, b, lab in runs:
        secs[lab] = secs.get(lab, 0.0) + (b - a) / fs
    info.update(classes=len(classes), class_list=",".join(classes), usable_s=round(sum(secs.values()), 1),
                seconds_per_class={k: round(v, 1) for k, v in sorted(secs.items())}, runs=len(runs))
    if len(classes) < cfg.min_classes_per_session:
        info["reason"] = f"only {len(classes)} usable class(es) after filtering"
        return None, info
    stds = np.array([np.nanstd(signal[a:b], axis=0) for a, b, _ in runs[:: max(1, len(runs) // 40)]]).mean(axis=0)
    dead = [emg[i] for i in np.flatnonzero(stds < 1e-3)]
    if dead:
        info["reason"] = f"flat (dead) channel(s): {dead}"
        return None, info

    sess = f"day{day}_{hand}"
    rec = Recording(rec_id=folder.name, path=str(sig_files[0]), signal=np.nan_to_num(signal, nan=0.0), fs=fs,
                    subject=subject, session=sess, runs=runs, channel_names=list(emg))
    info["status"] = "loaded"
    return rec, info


def load_sessions(cfg: DataConfig) -> Tuple[List[Recording], dict]:
    """Load every usable session folder under ``cfg.paths``; returns (recordings, report)."""
    folders, skipped_dirs = find_sessions(cfg.paths, cfg.recursive)
    report = {"layout": "sessions", "files_found": len(folders) + len(skipped_dirs), "loaded": 0,
              "skipped": list(skipped_dirs), "sessions": [], "paths": [str(resolve_path(p)) for p in cfg.paths]}
    recs: List[Recording] = []
    for f in folders:
        try:
            rec, info = load_session(f, cfg)
        except Exception as exc:                             # one broken session must not stop the others
            rec, info = None, {"session": f.name, "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
        report["sessions"].append(info)
        if rec is None:
            report["skipped"].append({"file": f.name, "reason": info.get("reason", "")})
            LOG.warning("skipped session %s: %s", f.name, info.get("reason"))
        else:
            recs.append(rec)
    for s in skipped_dirs:
        report["sessions"].append({"session": s["file"], "status": "skipped", "reason": s["reason"]})
    if recs:                                                  # consistent channel layout
        counts: Dict[int, int] = {}
        for r in recs:
            counts[r.signal.shape[1]] = counts.get(r.signal.shape[1], 0) + 1
        modal = max(counts, key=counts.get)
        good = []
        for r in recs:
            if r.signal.shape[1] != modal:
                report["skipped"].append({"file": r.rec_id, "reason": f"{r.signal.shape[1]} EMG channels, dataset has {modal}"})
            else:
                good.append(r)
        recs = good
    report["loaded"] = len(recs)
    return recs, report
