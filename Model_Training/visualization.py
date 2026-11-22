"""Matplotlib figures for the comparison report (headless, PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_COLORS = {"ml": "#0072B2", "dl": "#D55E00", "hybrid": "#009E73"}      # colour-blind safe (Okabe-Ito)
FAMILY_NAMES = {"ml": "Classical ML", "dl": "Deep learning", "hybrid": "Hybrid"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def _legend_handles():
    return [plt.Line2D([], [], marker="s", ls="", color=c, label=FAMILY_NAMES[f]) for f, c in FAMILY_COLORS.items()]


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_ranking(df: pd.DataFrame, col: str, title: str, path: Path, ascending=False, log=False, xlabel=None, fmt="{:.3f}"):
    d = df.dropna(subset=[col]).sort_values(col, ascending=not ascending)       # best at top
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.4))
    bars = ax.barh(d["label"], d[col], color=[FAMILY_COLORS[f] for f in d["family"]])
    if log:
        ax.set_xscale("log")
    for b, v in zip(bars, d[col]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel or col)
    ax.legend(handles=_legend_handles(), loc="lower right", frameon=False)
    _save(fig, path)


def confusion_matrix_plot(cm: np.ndarray, classes: Sequence[str], title: str, path: Path):
    K = len(classes)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(5, 0.65 * K + 2), max(4.4, 0.6 * K + 1.6)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K), classes, rotation=45, ha="right")
    ax.set_yticks(range(K), classes)
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    ax.set_title(title, loc="left", fontweight="bold")
    for i in range(K):
        for j in range(K):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}" if K > 8 else f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                        fontsize=7 if K > 8 else 8, color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    ax.spines[:].set_visible(False)
    _save(fig, path)


def per_gesture_heatmap(per_class: Dict[str, List[dict]], df: pd.DataFrame, path: Path):
    models = [m for m in df.sort_values("f1_macro", ascending=False)["model"] if m in per_class]
    if not models:
        return
    classes = [r["gesture"] for r in per_class[models[0]]]
    M = np.array([[r["f1"] for r in per_class[m]] for m in models])
    labels = [df.set_index("model").loc[m, "label"] for m in models]
    fig, ax = plt.subplots(figsize=(0.8 * len(classes) + 3, 0.34 * len(models) + 1.8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(models)), labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.6 else "black")
    ax.set_title("Per-gesture F1 (test set), models sorted by macro-F1", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, label="F1")
    ax.spines[:].set_visible(False)
    _save(fig, path)


def per_gesture_top(per_class: Dict[str, List[dict]], df: pd.DataFrame, path: Path, top: int = 4):
    models = [m for m in df.sort_values("perf_score", ascending=False)["model"][:top] if m in per_class]
    if not models:
        return
    classes = [r["gesture"] for r in per_class[models[0]]]
    x = np.arange(len(classes))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(classes)), 4))
    lab = df.set_index("model")["label"]
    for i, m in enumerate(models):
        ax.bar(x + i * w - 0.4 + w / 2, [r["f1"] for r in per_class[m]], w, label=lab[m])
    ax.set_xticks(x, classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(f"Per-gesture F1 - top {len(models)} models", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    _save(fig, path)


def radar(df: pd.DataFrame, path: Path, top_n: int = 5):
    d = df.sort_values("perf_score", ascending=False).head(top_n)
    axes = {"Accuracy": "accuracy", "Balanced acc.": "balanced_accuracy", "Macro F1": "f1_macro"}
    if "latency_score" in d:
        axes.update({"Speed\n(1 - P95/budget)": "latency_score", "Compactness": "size_score"})
    if "group_f1_mean" in d and d["group_f1_mean"].notna().all():
        axes["Held-out group\nF1"] = "group_f1_mean"
    names = list(axes)
    ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 6), subplot_kw={"projection": "polar"})
    for _, r in d.iterrows():
        vals = [float(r[axes[n]]) for n in names]
        ax.plot(ang + ang[:1], vals + vals[:1], lw=1.8, label=r["label"])
        ax.fill(ang + ang[:1], vals + vals[:1], alpha=0.07)
    ax.set_xticks(ang, names)
    ax.set_ylim(0, 1)
    ax.set_title(f"Top {len(d)} models (all axes: higher is better)", fontweight="bold", y=1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False)
    _save(fig, path)


def tradeoff(df: pd.DataFrame, budget_ms: float, path: Path):
    d = df.dropna(subset=["lat_p95_ms", "f1_macro"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.4))
    sizes = 40 + 260 * np.sqrt(d["model_size_kb"] / max(d["model_size_kb"].max(), 1))
    ax.scatter(d["lat_p95_ms"], d["f1_macro"], s=sizes, c=[FAMILY_COLORS[f] for f in d["family"]], alpha=0.75,
               edgecolor="white", linewidth=0.8)
    for _, r in d.iterrows():
        ax.annotate(r["label"], (r["lat_p95_ms"], r["f1_macro"]), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    front = d[d["pareto"]].sort_values("lat_p95_ms")
    ax.step(front["lat_p95_ms"], front["f1_macro"], where="post", color="0.35", lw=1, ls="--", label="Pareto front")
    ax.axvline(budget_ms, color="#CC0000", lw=1, ls=":")
    ax.text(budget_ms, ax.get_ylim()[0], f" real-time budget {budget_ms:.0f} ms", color="#CC0000", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("P95 single-window latency, CPU (ms, log scale)  -  lower is better")
    ax.set_ylabel("Macro F1 (test)  -  higher is better")
    ax.set_title("Accuracy vs latency trade-off (bubble size = model size)", loc="left", fontweight="bold")
    ax.legend(handles=_legend_handles() + [plt.Line2D([], [], color="0.35", ls="--", label="Pareto front")], frameon=False, loc="upper left")
    _save(fig, path)


def metrics_heatmap(df: pd.DataFrame, path: Path):
    cols = [("accuracy", "Accuracy"), ("balanced_accuracy", "Bal. acc"), ("precision_macro", "Prec (m)"),
            ("recall_macro", "Recall (m)"), ("f1_macro", "F1 (m)"), ("f1_weighted", "F1 (w)"), ("roc_auc_macro", "ROC-AUC")]
    cols = [c for c in cols if c[0] in df and df[c[0]].notna().any()]
    d = df.sort_values("f1_macro", ascending=False)
    M = d[[c[0] for c in cols]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.95 * len(cols) + 2.5, 0.34 * len(d) + 1.6))
    im = ax.imshow(M, cmap="viridis", vmin=max(0, np.nanmin(M) - 0.05), vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c[1] for c in cols])
    ax.set_yticks(range(len(d)), d["label"])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "" if np.isnan(M[i, j]) else f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] < 0.75 else "black")
    ax.set_title("Test-set metrics by model", loc="left", fontweight="bold")
    ax.xaxis.tick_top()
    ax.spines[:].set_visible(False)
    _save(fig, path)


def training_curves(histories: Dict[str, dict], labels: Dict[str, str], path: Path):
    items = [(k, h) for k, h in histories.items() if h and h.get("epoch")]
    if not items:
        return
    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (k, h) in zip(axes.ravel(), items):
        ax.axis("on")
        ax.plot(h["epoch"], h["train_loss"], color="#D55E00", label="train loss")
        if h.get("val_loss"):
            ax.plot(h["epoch"], h["val_loss"], color="#0072B2", label="val loss")
            ax2 = ax.twinx()
            ax2.plot(h["epoch"], h["val_macro_f1"], color="#009E73", ls="--", label="val macro-F1")
            ax2.set_ylim(0, 1.02)
            ax2.spines["right"].set_visible(True)
            ax2.tick_params(labelsize=7)
        ax.set_title(labels.get(k, k), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
    fig.suptitle("Deep-model training curves (loss left, validation macro-F1 right)", x=0.01, ha="left", fontweight="bold")
    _save(fig, path)


def class_distribution(seconds: Dict[str, float], path: Path):
    if not seconds:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(list(seconds), list(seconds.values()), color="#0072B2")
    ax.set_ylabel("labelled seconds")
    ax.set_title("Dataset: labelled signal per gesture", loc="left", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, path)
