"""Markdown summary report generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd

TABLE_COLS = [("label", "Model"), ("family", "Family"), ("accuracy", "Acc"), ("balanced_accuracy", "Bal.Acc"),
              ("f1_macro", "Macro-F1"), ("roc_auc_macro", "AUC"), ("lat_p95_ms", "P95 ms"),
              ("model_size_kb", "Size KB"), ("train_time_s", "Train s"), ("realtime_status", "Real-time")]


def md_table(df: pd.DataFrame, cols=TABLE_COLS) -> str:
    cols = [c for c in cols if c[0] in df.columns]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for key, _ in cols:
            v = r[key]
            if isinstance(v, float):
                cells.append("-" if pd.isna(v) else (f"{v:.3f}" if key not in ("model_size_kb", "train_time_s", "lat_p95_ms") else f"{v:.1f}" if v >= 10 else f"{v:.2f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tradeoff_text(df: pd.DataFrame, recs: List[dict]) -> str:
    by = {r["criterion"]: r for r in recs}
    row = lambda m: df[df["model"] == m].iloc[0]
    out = []
    best = by.get("Best overall classification performance", {}).get("model")
    if best is None:
        return "No models completed."
    b = row(best)
    out.append(f"**Reference - best predictive model: {b['label']}** (macro-F1 {b['f1_macro']:.3f}"
               + (f", P95 {b['lat_p95_ms']:.2f} ms, {b['model_size_kb']:.0f} KB)." if "lat_p95_ms" in df and pd.notna(b.get("lat_p95_ms")) else ")."))
    if "lat_p95_ms" not in df.columns:
        return "\n\n".join(out)
    seen = {best}
    for crit in ("Best model for real-time deployment", "Best lightweight model", "Fastest model for inference"):
        m = by.get(crit, {}).get("model")
        if not m or m in seen:
            if m == best:
                out.append(f"- *{crit}*: the same model - no accuracy/efficiency trade-off needed.")
            continue
        seen.add(m)
        r = row(m)
        d_f1 = r["f1_macro"] - b["f1_macro"]
        speed = b["lat_p95_ms"] / max(r["lat_p95_ms"], 1e-9)
        size = b["model_size_kb"] / max(r["model_size_kb"], 1e-9)
        out.append(f"- *{crit}* - **{r['label']}**: macro-F1 {r['f1_macro']:.3f} ({d_f1:+.3f} vs reference), "
                   f"P95 latency {r['lat_p95_ms']:.2f} ms ({speed:.1f}x {'faster' if speed >= 1 else 'slower'}), "
                   f"{r['model_size_kb']:.0f} KB ({size:.1f}x {'smaller' if size >= 1 else 'larger'}).")
    front = df[df["pareto"]].sort_values("lat_p95_ms")
    out.append("**Pareto-optimal models** (no other model is both more accurate and faster): " +
               ", ".join(f"{r['label']} ({r['f1_macro']:.3f} F1 @ {r['lat_p95_ms']:.2f} ms)" for _, r in front.iterrows()) + ".")
    half = df["accuracy_ci95_halfwidth"].median() if "accuracy_ci95_halfwidth" in df else None
    if half is not None and pd.notna(half):
        out.append(f"**Statistical caution:** test accuracy has a ~95% half-width of about +/-{half:.3f} "
                   f"(effective sample size shrinks because windows overlap). Differences smaller than that are not "
                   f"evidence that one model is better; prefer the cheaper model when scores are tied.")
    return "\n\n".join(out)


def write_summary(path: Path, *, exp_name: str, df: pd.DataFrame, recs: List[dict], data_summary: dict,
                  split_info: dict, window_counts: dict, skipped: list, errors: dict, cfg, env: dict,
                  budget_ms: float, demo: bool) -> None:
    L: List[str] = []
    L.append(f"# EMG gesture-model benchmark - {exp_name}\n")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} - {len(df)} model(s) evaluated, "
             f"{len(errors)} failed, {len(skipped)} skipped._\n")
    if demo:
        L.append("> **WARNING - synthetic demo data.** These numbers only prove that the pipeline runs end to end. "
                 "They say nothing about real gesture recognition; re-run on labelled BioWave recordings.\n")
    L.append("## 1. Evaluation protocol\n")
    mode = split_info["mode"]
    L.append(f"- **Split mode:** `{mode}` - held-out data is independent of: *{split_info['independent_of']}*.")
    L.append(f"- Windows: {cfg.preprocess.window_ms:.0f} ms ({cfg.window_samples} samples @ {data_summary['sampling_rate']:.0f} Hz), "
             f"stride {cfg.preprocess.stride_ms:.0f} ms. Windows are created **after** splitting, inside each split, so no window straddles two splits.")
    L.append(f"- train / val / test windows: {window_counts.get('train', 0)} / {window_counts.get('val', 0)} / {window_counts.get('test', 0)}.")
    for name in ("train", "val", "test"):
        g = split_info["groups"].get(name, [])
        if mode != "temporal" and g:
            L.append(f"- {name} {mode}s: {', '.join(g[:12])}{' ...' if len(g) > 12 else ''}")
    L.append("- Normalisation, feature scaling and class weights are fitted on the training split only. All models share identical windows, "
             f"seed ({cfg.train.seed}) and test data. Latency: CPU, {cfg.benchmark.threads} thread(s), full pipeline "
             f"(filter + features/normalisation + model) per window.")
    if mode == "temporal":
        L.append("- **Caveat:** with `temporal` splitting the test blocks come from the *same recordings* as training, so scores "
                 "are optimistic and say nothing about new subjects or sessions. Record more subjects/sessions (>= 3 subjects enables a subject-independent test).")
    elif mode == "recording":
        L.append("- **Caveat:** whole recordings are held out but the same subjects/sessions appear in training; subject-independence is not measured.")
    L.append(f"\n## 2. Dataset\n")
    L.append(f"{data_summary['n_recordings']} recordings, {data_summary['n_channels']} channels, {len(data_summary['subjects'])} subject(s), "
             f"{len(data_summary['sessions'])} session(s), {data_summary['total_seconds']:.0f} s total.")
    L.append("Labelled seconds per gesture: " + ", ".join(f"{k}: {v:.0f}" for k, v in data_summary["seconds_per_class"].items()) + ".\n")

    L.append("## 3. Recommendations\n")
    L.append("| Criterion | Model | Evidence |\n|---|---|---|")
    for r in recs:
        L.append(f"| {r['criterion']} | **{r['label'] or 'n/a'}** | {r['value'] or '-'} |")
    L.append("")
    for r in recs:
        L.append(f"- *{r['criterion']}* - {r['why']}")
    L.append("\n## 4. Accuracy vs efficiency trade-offs\n")
    L.append(tradeoff_text(df, recs))
    L.append("\n## 5. Leaderboard (sorted by overall performance score)\n")
    L.append(md_table(df.sort_values("perf_score", ascending=False)))
    if "realtime_status" in df:
        L.append(f"\n## 6. Real-time feasibility (budget {budget_ms:.0f} ms = decision interval)\n")
        for status, desc in (("comfortable", f"P95 <= {cfg.benchmark.comfortable_fraction:.0%} of budget"),
                             ("marginal", "P95 within budget but little headroom"), ("too slow", "P95 exceeds the budget")):
            m = df[df["realtime_status"] == status]
            L.append(f"- **{status}** ({desc}): " + (", ".join(m.sort_values("lat_p95_ms")["label"]) if len(m) else "none"))
    L.append("\n## 7. Skipped / failed\n")
    if not skipped and not errors:
        L.append("None.")
    for k, why in skipped:
        L.append(f"- `{k}` skipped: {why}")
    for k, why in errors.items():
        L.append(f"- `{k}` FAILED: {why} (full traceback in `training.log`)")
    L.append("\n## 8. Files\n")
    L.append("`model_comparison.csv` / `metrics.json` (all metrics), `leaderboard.csv`, `recommendations.json`, "
             "`plots/` (rankings, latency, size, trade-off, radar, heatmaps), `confusion_matrices/`, `per_model/` (raw per-model records), "
             "`models/<key>/` (deployable bundles; `models/recommended_realtime/` is the pick above), `config.json`, `environment.json`, `training.log`.")
    L.append("\nLoad a model: `from Model_Training.inference import GestureClassifier; clf = GestureClassifier.load('<models dir>/<key>')`.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
