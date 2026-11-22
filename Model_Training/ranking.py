"""Multi-criteria ranking and recommendations.  No single metric decides: accuracy alone never picks a winner."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import RankingConfig


def pareto_front(df: pd.DataFrame, quality="f1_macro", cost="lat_p95_ms") -> pd.Series:
    """True for models not dominated on (higher quality, lower latency)."""
    flags = []
    for i, r in df.iterrows():
        dominated = ((df[quality] >= r[quality]) & (df[cost] <= r[cost]) &
                     ((df[quality] > r[quality]) | (df[cost] < r[cost]))).any()
        flags.append(not dominated)
    return pd.Series(flags, index=df.index)


def add_scores(df: pd.DataFrame, rcfg: RankingConfig, budget_ms: float) -> pd.DataFrame:
    df = df.copy()
    w = rcfg.perf_weights
    df["perf_score"] = sum(w[k] * df[k].fillna(0.0) for k in w)                       # overall predictive quality
    if "lat_p95_ms" in df and df["lat_p95_ms"].notna().any():
        df["latency_score"] = (1.0 - df["lat_p95_ms"] / budget_ms).clip(0, 1)
        mb = (df["model_size_kb"] / 1024.0).clip(lower=0.01)
        df["size_score"] = (1.0 - np.log10(mb / 0.01) / 3.0).clip(0, 1)               # 0.01 MB -> 1, 10 MB -> 0
        r = rcfg.rt_weights
        df["rt_score"] = r["f1_macro"] * df["f1_macro"] + r["latency"] * df["latency_score"] + r["size"] * df["size_score"]
        df["rt_eligible"] = df["lat_p95_ms"] <= budget_ms
        df["pareto"] = pareto_front(df)
    return df


def _pick(df: pd.DataFrame, col: str, ascending: bool = False, tie: Optional[str] = "perf_score"):
    d = df.dropna(subset=[col])
    if d.empty:
        return None
    cols, asc = [col] + ([tie] if tie and tie != col else []), [ascending, False]
    return d.sort_values(cols, ascending=asc[: len(cols)]).iloc[0]


def recommendations(df: pd.DataFrame, rcfg: RankingConfig, split_mode: str, budget_ms: float) -> List[dict]:
    """The eight requested 'best model' categories, each with the evidence behind it."""
    out: List[dict] = []
    tol = rcfg.f1_tolerance
    best_f1 = float(df["f1_macro"].max())
    has_rt = "rt_score" in df

    def add(criterion, row, value, why):
        out.append({"criterion": criterion, "model": None if row is None else row["model"],
                    "label": None if row is None else row["label"], "value": value, "why": why})

    r = _pick(df, "perf_score")
    add("Best overall classification performance", r,
        None if r is None else f"score {r['perf_score']:.3f} (acc {r['accuracy']:.3f}, bal.acc {r['balanced_accuracy']:.3f}, macro-F1 {r['f1_macro']:.3f})",
        f"Weighted blend of macro-F1/balanced-accuracy/accuracy ({rcfg.perf_weights}); not accuracy alone.")
    r = _pick(df, "f1_macro")
    add("Best Macro F1-score", r, None if r is None else f"{r['f1_macro']:.3f}",
        "Treats every gesture equally, so it exposes weak classes that accuracy hides.")

    if split_mode in ("subject", "session") and "group_f1_mean" in df and df["group_f1_mean"].notna().any():
        d = df.copy()
        d["gen_score"] = d["group_f1_mean"] - 0.5 * d["group_f1_std"].fillna(0)
        r = _pick(d, "gen_score")
        add(f"Best {split_mode}-independent generalisation", r,
            None if r is None else f"mean macro-F1 over held-out {split_mode}s {r['group_f1_mean']:.3f} +/- {r['group_f1_std']:.3f}",
            f"Trained without the test {split_mode}s; ranked by mean minus half the std across held-out {split_mode}s (rewards consistency).")
    else:
        col = "cv_f1_mean" if "cv_f1_mean" in df and df["cv_f1_mean"].notna().any() else "f1_macro"
        r = _pick(df, col)
        add("Best generalisation (subject-independence NOT measurable)", r, None if r is None else f"{col} = {r[col]:.3f}",
            f"Split mode is '{split_mode}': the data has fewer than 3 subjects/sessions, so unseen-subject performance "
            f"cannot be measured. This is the best held-out-{'recording' if split_mode == 'recording' else 'time-block'} result instead.")

    if has_rt:
        viable = df[df["f1_macro"] >= best_f1 - 3 * tol]
        r = _pick(viable, "lat_p95_ms", ascending=True)
        raw = _pick(df, "lat_p95_ms", ascending=True)
        extra = "" if raw is None or r is None or raw["model"] == r["model"] else \
            f" (absolute fastest: {raw['label']} at {raw['lat_p95_ms']:.2f} ms, but macro-F1 only {raw['f1_macro']:.3f})"
        add("Fastest model for inference", r, None if r is None else f"P95 {r['lat_p95_ms']:.3f} ms, median {r['lat_median_ms']:.3f} ms",
            f"Lowest P95 end-to-end latency among models within {3 * tol:.2f} macro-F1 of the best{extra}.")
        r = _pick(df[df["rt_eligible"]], "rt_score") if df["rt_eligible"].any() else None
        add("Best model for real-time deployment", r,
            None if r is None else f"rt-score {r['rt_score']:.3f}: macro-F1 {r['f1_macro']:.3f}, P95 {r['lat_p95_ms']:.2f} ms ({r['realtime_status']}), {r['model_size_kb']:.0f} KB",
            f"Max of {rcfg.rt_weights} over accuracy, latency headroom vs the {budget_ms:.0f} ms budget and size; models over budget are excluded.")
        near = df[df["f1_macro"] >= best_f1 - tol]
        r = _pick(near, "model_size_kb", ascending=True, tie="lat_p95_ms")
        add("Best lightweight model", r, None if r is None else f"{r['model_size_kb']:.1f} KB, macro-F1 {r['f1_macro']:.3f}",
            f"Smallest on-disk model among those within {tol:.2f} macro-F1 of the best.")
    for fam, name in (("dl", "Best deep learning architecture"), ("hybrid", "Best hybrid architecture")):
        sub = df[df["family"] == fam]
        r = _pick(sub, "perf_score")
        add(name, r, None if r is None else f"score {r['perf_score']:.3f}, macro-F1 {r['f1_macro']:.3f}" +
            (f", P95 {r['lat_p95_ms']:.2f} ms" if has_rt and pd.notna(r.get("lat_p95_ms")) else ""),
            f"Highest overall performance score among {fam} models." if r is not None else "No model of this family completed.")
    return out
