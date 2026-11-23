"""Comparison table, best-model selection, charts (matplotlib ``Figure`` objects, usable in Qt or saved as PNG)
and the on-disk evaluation reports."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

LIGHT_CHART = dict(bg="#FFFFFF", fg="#222222", muted="#6B7489", grid="#E6E9F0", accent="#4C6EF5",
                   best="#2B8A3E", cmap="Blues")
CHART = dict(LIGHT_CHART)          # colours used by every figure; the GUI swaps these when the theme changes


def set_chart_theme(**kw) -> None:
    CHART.update(kw)


@contextmanager
def chart_theme(**kw):
    old = dict(CHART)
    CHART.update(kw)
    try:
        yield
    finally:
        CHART.clear()
        CHART.update(old)


def _finish(fig: Figure) -> None:
    """Apply the theme to the figure background, every axis and its text, then lay out."""
    fig.set_facecolor(CHART["bg"])
    for ax in fig.axes:
        ax.set_facecolor(CHART["bg"])
        for sp in ax.spines.values():
            sp.set_color(CHART["muted"])
        ax.tick_params(colors=CHART["fg"], labelcolor=CHART["fg"])
        ax.xaxis.label.set_color(CHART["fg"])
        ax.yaxis.label.set_color(CHART["fg"])
        ax.title.set_color(CHART["fg"])
    try:
        fig.tight_layout()
    except Exception:
        pass

COLUMNS = {
    "classification": [("accuracy", "Accuracy"), ("precision_macro", "Precision"), ("recall_macro", "Recall"),
                       ("f1_macro", "F1"), ("roc_auc_macro", "ROC-AUC")],
    "regression": [("mae", "MAE"), ("mse", "MSE"), ("rmse", "RMSE"), ("r2", "R²")],
}
PRIMARY = {"classification": ("f1_macro", "F1 (macro)", True), "regression": ("r2", "R²", True)}


def comparison_table(results: List[dict], task: str) -> pd.DataFrame:
    """One row per completed model, ranked by the primary metric (F1 / R²), ties broken by training time."""
    rows = []
    for r in results:
        row = {"Model": r["label"], "Family": r["family"]}
        for k, name in COLUMNS[task]:
            row[name] = r["metrics"].get(k)
        rt = r.get("realtime") or {}
        if r.get("realtime") is not None:                          # EMG gesture runs: deployment-oriented columns
            row["Bal. acc"] = r["metrics"].get("balanced_accuracy")
            row["Smoothed F1"] = rt.get("f1_macro_smoothed")
            row["Worst subject F1"] = (r.get("group_spread") or {}).get("group_f1_min")
            row["Latency P95 (ms)"] = rt.get("lat_p95_ms")
        row["Train time (s)"] = r["train_time_s"]
        row["Size (KB)"] = r["model_size_bytes"] / 1024.0
        row["Artifact"] = r["artifact"]
        row["_key"] = r["model_key"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    col = PRIMARY[task][1].replace(" (macro)", "")
    col = {"F1": "F1", "R²": "R²"}[col]
    df = df.sort_values([col, "Train time (s)"], ascending=[False, True], na_position="last").reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def best_model(results: List[dict], task: str) -> Optional[dict]:
    if not results:
        return None
    t = comparison_table(results, task)
    key = t.iloc[0]["_key"]
    return next(r for r in results if r["model_key"] == key)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=CHART["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)


def _barh(ax, labels, values, best_idx, title, fmt="{:.3f}"):
    colors = [CHART["best"] if i == best_idx else CHART["accent"] for i in range(len(labels))]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    vmax = max([v for v in values if v is not None and np.isfinite(v)] + [1e-12])
    for yi, v in zip(y, values):
        if v is not None and np.isfinite(v):
            ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8, color=CHART["fg"])
    ax.set_xlim(0, vmax * 1.15)
    _style(ax)


def fig_scores(results: List[dict], task: str, fig: Optional[Figure] = None) -> Figure:
    """Primary-metric bars + the remaining metrics side by side."""
    fig = fig or Figure(figsize=(9, 4.5))
    fig.clear()
    t = comparison_table(results, task)
    if t.empty:
        return fig
    metrics = [c for _, c in COLUMNS[task]]
    show = ["F1", "Accuracy", "ROC-AUC"] if task == "classification" else ["R²", "RMSE", "MAE"]
    axes = fig.subplots(1, len(show), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), show):
        vals = [None if pd.isna(v) else float(v) for v in t[m]]
        lower_better = m in ("RMSE", "MAE", "MSE")
        finite = [v for v in vals if v is not None]
        bi = (int(np.argmin(vals)) if lower_better else int(np.nanargmax([np.nan if v is None else v for v in vals]))) \
            if finite else -1
        _barh(ax, list(t["Model"]), [np.nan if v is None else v for v in vals], bi,
              m + (" (lower is better)" if lower_better else ""))
    _finish(fig)
    return fig


def fig_cost(results: List[dict], task: str, fig: Optional[Figure] = None) -> Figure:
    """Training time and exported-model size."""
    fig = fig or Figure(figsize=(9, 4.5))
    fig.clear()
    t = comparison_table(results, task)
    if t.empty:
        return fig
    a1, a2 = fig.subplots(1, 2, sharey=True)
    _barh(a1, list(t["Model"]), list(t["Train time (s)"]), int(np.argmin(t["Train time (s)"])), "Training time (s)", "{:.2f}")
    _barh(a2, list(t["Model"]), list(t["Size (KB)"]), int(np.argmin(t["Size (KB)"])), "Model file size (KB)", "{:.0f}")
    _finish(fig)
    return fig


def fig_confusion(result: dict, fig: Optional[Figure] = None, normalize: bool = True) -> Figure:
    fig = fig or Figure(figsize=(5.5, 5))
    fig.clear()
    ax = fig.add_subplot(111)
    cm = np.asarray(result["confusion_matrix"], dtype=float)
    classes = result["classes"]
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    im = ax.imshow(shown, cmap=CHART["cmap"], vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{result['label']} - confusion matrix (row-normalised, counts in cells)" if normalize
                 else f"{result['label']} - confusion matrix", fontsize=9, loc="left")
    if len(classes) <= 20:
        for i in range(len(classes)):
            for j in range(len(classes)):
                rgb = im.cmap(im.norm(shown[i, j]))[:3]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="#111111" if lum > 0.55 else "#FFFFFF")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finish(fig)
    return fig


def fig_pred_vs_actual(result: dict, fig: Optional[Figure] = None) -> Figure:
    fig = fig or Figure(figsize=(5.5, 5))
    fig.clear()
    ax = fig.add_subplot(111)
    yt, yp = np.asarray(result["y_true_sample"]), np.asarray(result["y_pred_sample"])
    ax.scatter(yt, yp, s=14, color=CHART["accent"], alpha=0.6, edgecolors="none")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], color=CHART["muted"], linewidth=1, linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{result['label']} - predicted vs actual (test sample)", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _finish(fig)
    return fig


# --------------------------------------------------------------------------------------------- on disk
def save_figure(fig: Figure, path: Path) -> None:
    FigureCanvasAgg(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")


def write_reports(engine) -> Dict[str, str]:
    with chart_theme(**LIGHT_CHART):              # files on disk are always light, whatever the GUI theme
        return _write_reports(engine)


def _write_reports(engine) -> Dict[str, str]:
    """Write summary.csv/json, report.md, one JSON per model and the PNG plots; returns the paths."""
    plan = engine.plan
    task = plan.task
    results = engine.results()
    rdir, pdir = plan.dirs.reports / plan.run_id, plan.dirs.plots / plan.run_id
    rdir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {"reports": str(rdir), "plots": str(pdir)}

    for r in results:
        (rdir / f"{r['label'].replace(' ', '_')}_{plan.run_id}.json").write_text(
            json.dumps(r, indent=2, default=str), encoding="utf-8")

    table = comparison_table(results, task)
    status = [{"Model": j.label, "Status": j.state, "Train time (s)": round(j.elapsed, 2),
               "Error": j.error, "Artifact": (j.result or {}).get("artifact", "")} for j in engine.jobs.values()]
    pd.DataFrame(status).to_csv(rdir / "training_status.csv", index=False)
    if not table.empty:
        table.drop(columns=["_key"]).to_csv(rdir / "model_comparison.csv", index=False)
        best = best_model(results, task)
        for name, fn in (("scores", fig_scores), ("time_and_size", fig_cost)):
            save_figure(fn(results, task), pdir / f"{name}.png")
        for r in results:
            if task == "classification":
                save_figure(fig_confusion(r), pdir / f"confusion_{r['label'].replace(' ', '_')}.png")
            else:
                save_figure(fig_pred_vs_actual(r), pdir / f"pred_vs_actual_{r['label'].replace(' ', '_')}.png")
    else:
        best = None

    prim = PRIMARY[task][1]
    lines = [f"# Training report - run {plan.run_id}", "",
             f"- Task: **{task}**, target `{plan.target}`",
             f"- Dataset: {plan.dataset_info.get('name')} - " + (plan.dataset_info.get("summary") or
             f"{plan.dataset_info.get('rows_used')} rows used of {plan.dataset_info.get('rows_original')}"),
             f"- Split: {plan.n_train} train / {plan.n_test} test {'windows' if plan.kind == 'emg' else 'rows'} (seed {plan.seed})",
             f"- Settings: `{json.dumps(plan.options, default=str)}`", ""]
    for w in plan.warnings:
        lines.append(f"> Warning: {w}")
    lines += ["", f"## Best model: {best['label']} ({prim} = {best['metrics'][PRIMARY[task][0]]:.4f})" if best
              else "## No model completed", "", "## Status", ""]
    lines += ["| Model | Status | Time (s) | Note |", "|---|---|---|---|"]
    for s in status:
        lines.append(f"| {s['Model']} | {s['Status']} | {s['Train time (s)']} | {str(s['Error'])[:120]} |")
    if not table.empty:
        show = table.drop(columns=["_key", "Artifact"])
        lines += ["", "## Comparison", "", show.to_markdown(index=False, floatfmt=".4f") if _has_tabulate() else show.to_string(index=False)]
        lines += ["", "## Exported models", ""] + [f"- `{r['artifact']}`" for r in results]
    (rdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    out["report"] = str(rdir / "report.md")
    return out


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
