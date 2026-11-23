"""Leakage-safe train / validation / test splitting and cross-validation folds.

Splitting happens on *recording segments before windows are created*, because neighbouring sliding
windows share most of their samples; a random window split would put near-copies of the same signal into
train and test and inflate every score.

Modes (``auto`` picks the strictest one the data allows):

* ``subject``   - whole subjects held out (subject-independent evaluation)
* ``session``   - whole sessions held out (session-independent evaluation)
* ``recording`` - whole recordings held out, stratified by gesture
* ``trial``     - within every recording and gesture the LATER repetitions are test (the ones before them validation, the earliest
                  train): "calibrate on early repetitions, predict later ones" - the realistic protocol for a personal model
* ``temporal``  - each labelled run is cut into disjoint, time-separated blocks that are assigned to
                  train / val / test (within-recording evaluation; the optimistic fallback for tiny datasets)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import SplitConfig
from .data_io import Recording
from .preprocessing import Segment
from .utils import LOG

MODES = ("subject", "session", "recording", "trial", "temporal")
TEMPORAL_BLOCKS = 10


@dataclass(frozen=True)
class Piece:
    seg: Segment
    unit: int           # index of the labelled run this piece was cut from
    group: str          # what must never be shared between splits in group modes


def make_units(recs: Sequence[Recording]) -> List[Segment]:
    return [Segment(i, a, b, lab) for i, r in enumerate(recs) for a, b, lab in r.runs]


def _group_key(recs, seg: Segment, level: str) -> str:
    r = recs[seg.rec_idx]
    return {"subject": r.subject, "session": f"{r.subject}|{r.session}", "recording": r.rec_id}[level]


def choose_mode(recs, units, requested: str) -> str:
    if requested != "auto":
        if requested not in MODES:
            raise ValueError(f"split.mode must be 'auto' or one of {MODES}")
        return requested
    n_sub = len({r.subject for r in recs})
    n_sess = len({(r.subject, r.session) for r in recs})
    per_class: Dict[str, set] = {}
    for u in units:
        per_class.setdefault(u.label, set()).add(recs[u.rec_idx].rec_id)
    if n_sub >= 3:
        return "subject"
    if n_sess >= 3:
        return "session"
    if per_class and min(len(v) for v in per_class.values()) >= 3:
        return "recording"
    return "temporal"


def _n_assign(n: int, val: float, test: float) -> Tuple[int, int]:
    n_test = max(1, int(round(test * n)))
    n_val = max(1, int(round(val * n)))
    while n - n_test - n_val < 1 and (n_test > 1 or n_val > 1):
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        else:
            n_val -= 1
    return n_val, n_test


def _group_split(recs, units, level, cfg: SplitConfig):
    groups = sorted({_group_key(recs, u, level) for u in units})
    if len(groups) < 3:
        raise ValueError(f"{level}-independent split needs >= 3 {level}s, found {len(groups)}")
    n_val, n_test = _n_assign(len(groups), cfg.val_size, cfg.test_size)
    classes = {u.label for u in units}
    rng = np.random.default_rng(cfg.seed)
    best, best_missing = None, 10 ** 9
    for _ in range(200):
        order = list(rng.permutation(len(groups)))
        assign = {}
        for rank, gi in enumerate(order):
            assign[groups[gi]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
        missing = 0
        for split, weight in (("train", 100), ("val", 1), ("test", 1)):
            have = {u.label for u in units if assign[_group_key(recs, u, level)] == split}
            missing += weight * len(classes - have)
        if missing < best_missing:
            best, best_missing = assign, missing
        if missing == 0:
            break
    if best_missing >= 100:
        raise ValueError("could not find a split where every gesture appears in the training set")
    if best_missing:
        LOG.warning("some gestures are absent from the validation/test split (%d missing class-slots)", best_missing)
    out = {"train": [], "val": [], "test": []}
    for ui, u in enumerate(units):
        g = _group_key(recs, u, level)
        out[best[g]].append(Piece(u, ui, g))
    return out


def _recording_split(recs, units, cfg: SplitConfig):
    by_rec: Dict[int, List[int]] = {}
    for ui, u in enumerate(units):
        by_rec.setdefault(u.rec_idx, []).append(ui)
    rec_class = {ri: Counter(units[ui].label for ui in uis).most_common(1)[0][0] for ri, uis in by_rec.items()}
    by_class: Dict[str, List[int]] = {}
    for ri, lab in rec_class.items():
        by_class.setdefault(lab, []).append(ri)
    rng = np.random.default_rng(cfg.seed)
    assign = {}
    for lab, ris in sorted(by_class.items()):
        if len(ris) < 3:
            raise ValueError(f"gesture '{lab}' has only {len(ris)} recording(s); recording-level split needs >= 3")
        ris = [ris[i] for i in rng.permutation(len(ris))]
        n_val, n_test = _n_assign(len(ris), cfg.val_size, cfg.test_size)
        for k, ri in enumerate(ris):
            assign[ri] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    out = {"train": [], "val": [], "test": []}
    for ui, u in enumerate(units):
        out[assign[u.rec_idx]].append(Piece(u, ui, recs[u.rec_idx].rec_id))
    return out


def _trial_split(recs, units, cfg: SplitConfig):
    """Chronological repetition split inside every (recording, gesture)."""
    by: Dict[Tuple[int, str], List[int]] = {}
    for ui, u in enumerate(units):
        by.setdefault((u.rec_idx, u.label), []).append(ui)
    out = {"train": [], "val": [], "test": []}
    for (ri, lab), uis in sorted(by.items()):
        uis.sort(key=lambda i: units[i].start)
        n = len(uis)
        if n < 3:
            LOG.warning("%s: gesture '%s' has only %d repetition(s) - used for training only", recs[ri].rec_id, lab, n)
            n_val = n_test = 0
        else:
            n_val, n_test = _n_assign(n, cfg.val_size, cfg.test_size)
        for rank, ui in enumerate(uis):
            split = "test" if rank >= n - n_test else "val" if rank >= n - n_test - n_val else "train"
            out[split].append(Piece(units[ui], ui, f"{ri}:{ui}"))
    return out


def _blocks(u: Segment, win: int, n_blocks: int, gap: int) -> List[Segment]:
    L = u.stop - u.start
    edges = np.linspace(u.start, u.stop, n_blocks + 1).astype(int)
    segs = []
    for i in range(n_blocks):
        a = edges[i] + (gap if i > 0 else 0)
        b = edges[i + 1] - (gap if i < n_blocks - 1 else 0)
        if b - a >= win:
            segs.append(Segment(u.rec_idx, int(a), int(b), u.label))
    return segs


def _n_blocks(u: Segment, win: int) -> int:
    return int(min(TEMPORAL_BLOCKS, (u.stop - u.start) // (4 * win)))


def _temporal_split(recs, units, cfg: SplitConfig, win: int):
    rng = np.random.default_rng(cfg.seed)
    gap = win // 2
    out = {"train": [], "val": [], "test": []}
    for ui, u in enumerate(units):
        nb = _n_blocks(u, win)
        if nb < 3:
            LOG.warning("run %s[%d:%d] too short for a temporal split - used for training only",
                        recs[u.rec_idx].rec_id, u.start, u.stop)
            out["train"].append(Piece(u, ui, f"{ui}:all"))
            continue
        blocks = _blocks(u, win, nb, gap)
        ids = list(rng.permutation(len(blocks)))
        n_val, n_test = _n_assign(len(blocks), cfg.val_size, cfg.test_size)
        for rank, bi in enumerate(ids):
            split = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
            out[split].append(Piece(blocks[bi], ui, f"{ui}:{bi}"))
    return out


def split_pieces(recs, cfg: SplitConfig, win: int):
    """Returns (splits, info). splits: {'train'|'val'|'test': [Piece]}."""
    units = make_units(recs)
    mode = choose_mode(recs, units, cfg.mode)
    if cfg.mode == "auto":
        LOG.info("split mode (auto): %s", mode)
    if mode in ("subject", "session"):
        splits = _group_split(recs, units, mode, cfg)
    elif mode == "recording":
        splits = _recording_split(recs, units, cfg)
    elif mode == "trial":
        splits = _trial_split(recs, units, cfg)
    else:
        splits = _temporal_split(recs, units, cfg, win)
    verify_no_leakage(recs, splits, mode)
    info = {"mode": mode,
            "independent_of": {"subject": "subject", "session": "session (and recording)",
                               "recording": "recording", "trial": "repetition (same people, later trials)", "temporal": "nothing (same recordings, disjoint time blocks)"}[mode],
            "groups": {k: sorted({recs[p.seg.rec_idx].subject if mode == "subject" else
                                  f"{recs[p.seg.rec_idx].subject}|{recs[p.seg.rec_idx].session}" if mode == "session" else
                                  recs[p.seg.rec_idx].rec_id for p in v}) for k, v in splits.items()},
            "n_segments": {k: len(v) for k, v in splits.items()}}
    return splits, info


def verify_no_leakage(recs, splits: Dict[str, List[Piece]], mode: str) -> None:
    """Hard checks: no sample of a recording is used by two splits, and group modes keep groups disjoint."""
    spans: Dict[int, List[Tuple[int, int, str]]] = {}
    for name, pieces in splits.items():
        for p in pieces:
            spans.setdefault(p.seg.rec_idx, []).append((p.seg.start, p.seg.stop, name))
    for ri, items in spans.items():
        items.sort()
        for (a0, b0, n0), (a1, b1, n1) in zip(items, items[1:]):
            if a1 < b0 and n0 != n1:
                raise AssertionError(f"data leakage: recording {recs[ri].rec_id} samples shared by {n0} and {n1}")
    if mode != "temporal":
        owner: Dict[str, str] = {}
        for name, pieces in splits.items():
            for p in pieces:
                if owner.setdefault(p.group, name) != name:
                    raise AssertionError(f"data leakage: group {p.group} in both {owner[p.group]} and {name}")


def make_folds(pieces: Sequence[Piece], mode: str, k: int, seed: int) -> List[Tuple[List[Piece], List[Piece]]]:
    """Cross-validation folds over ``pieces`` that respect the split mode (never random windows)."""
    pieces = list(pieces)
    if len(pieces) < 2 or k < 2:
        return []
    rng = np.random.default_rng(seed)
    if mode == "temporal":
        by_unit: Dict[int, List[int]] = {}
        for i, p in enumerate(pieces):
            by_unit.setdefault(p.unit, []).append(i)
        fold_of = np.full(len(pieces), -1)
        for idxs in by_unit.values():
            idxs = [idxs[i] for i in rng.permutation(len(idxs))]
            off = int(rng.integers(0, k))
            for j, i in enumerate(idxs):
                fold_of[i] = (j + off) % k
        folds = []
        for f in range(k):
            va = [p for p, g in zip(pieces, fold_of) if g == f]
            tr = [p for p, g in zip(pieces, fold_of) if g != f]
            if va and tr:
                folds.append((tr, va))
        return folds
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    groups = np.array([p.group for p in pieces])
    y = np.array([p.seg.label for p in pieces])
    n_groups = len(set(groups))
    k = min(k, n_groups)
    if k < 2:
        return []
    try:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        idx = list(splitter.split(np.zeros(len(pieces)), y, groups))
    except ValueError:
        idx = list(GroupKFold(n_splits=k).split(np.zeros(len(pieces)), y, groups))
    return [([pieces[i] for i in tr], [pieces[i] for i in va]) for tr, va in idx]
