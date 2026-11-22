#!/usr/bin/env python
"""Train, evaluate, compare and benchmark EMG gesture classifiers - one command, one result folder.

    python Model_Training/model_train.py --demo --quick          # smoke test on synthetic data
    python Model_Training/model_train.py                         # all models on ./recordings
    python Model_Training/model_train.py --data-path D:/emg --models rf,svm_rbf,cnn1d
    python Model_Training/model_train.py --list-models | --dump-config > my.json

See README.md in this folder for the data conventions and the meaning of every output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):                                   # `python model_train.py` from any directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Model_Training"

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

from .benchmark import benchmark_classifier
from .config import ExperimentConfig, HERE, resolve_path
from .crossval import FoldCache, build_model, cross_validate, random_search
from .data_io import load_dataset, summarize_dataset
from .evaluation import evaluate, group_spread
from .inference import GestureClassifier
from .models.registry import SPECS, select_models
from .models.torch_wrapper import resolve_device
from .pipeline import filter_rare_classes, prepare
from .ranking import add_scores, recommendations
from .report import write_summary
from .splits import split_pieces
from .utils import (LOG, PeakMemory, Timer, dir_size_bytes, environment_info, have, next_experiment_dir,
                    save_json, seed_everything, setup_logging, to_jsonable)
from . import visualization as viz

QUICK_PARAMS = {"rf": {"n_estimators": 60}, "extra_trees": {"n_estimators": 60}, "xgboost": {"n_estimators": 80},
                "lightgbm": {"n_estimators": 80}, "catboost": {"iterations": 80}, "gradient_boosting": {"n_estimators": 40},
                "mlp": {"max_iter": 80}}
DEMO_DIR = HERE / "data_demo"


# ---------------------------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config (see --dump-config); CLI flags override it")
    p.add_argument("--dump-config", action="store_true", help="print the default config as JSON and exit")
    p.add_argument("--list-models", action="store_true", help="list models and whether their libraries are installed")
    p.add_argument("--data-path", action="append", help="recordings folder or file (repeatable)")
    p.add_argument("--demo", action="store_true", help="generate + use SYNTHETIC data (pipeline smoke test only)")
    p.add_argument("--quick", action="store_true", help="few epochs/trees, no CV/tuning, short benchmark (minutes, not hours)")
    p.add_argument("--models", help="comma-separated model keys (default: all)")
    p.add_argument("--families", help="comma-separated subset of ml,dl,hybrid")
    p.add_argument("--exclude", help="comma-separated model keys to skip")
    p.add_argument("--split-mode", choices=["auto", "subject", "session", "recording", "temporal"])
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--window-ms", type=float)
    p.add_argument("--stride-ms", type=float)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--tune", action="store_true", help="random-search hyper-parameters of classical models")
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--no-benchmark", action="store_true")
    p.add_argument("--name", help="experiment folder name (default experiment_NNN)")
    p.add_argument("--resume", action="store_true", help="reuse --name folder and skip models already finished there")
    p.add_argument("--output", help="results root (default Model_Training/results)")
    return p.parse_args(argv)


def build_config(a) -> ExperimentConfig:
    cfg = ExperimentConfig.load(a.config) if a.config else ExperimentConfig()
    csv = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    if a.data_path:
        cfg.data.paths = a.data_path
    if a.models:
        cfg.models.enabled = csv(a.models)
    if a.families:
        cfg.models.families = csv(a.families)
    if a.exclude:
        cfg.models.exclude = csv(a.exclude)
    if a.split_mode:
        cfg.split.mode = a.split_mode
    if a.seed is not None:
        cfg.train.seed = cfg.split.seed = a.seed
    if a.epochs:
        cfg.train.epochs = a.epochs
    if a.window_ms:
        cfg.preprocess.window_ms = a.window_ms
    if a.stride_ms:
        cfg.preprocess.stride_ms = a.stride_ms
    if a.device:
        cfg.train.device = a.device
    if a.tune:
        cfg.tuning.enabled = True
    if a.no_cv:
        cfg.cv.enabled = False
    if a.no_benchmark:
        cfg.benchmark.enabled = False
    if a.name:
        cfg.output.name = a.name
    if a.output:
        cfg.output.root = a.output
    if a.quick:
        cfg.train.epochs = min(cfg.train.epochs, 8)
        cfg.train.patience = 4
        cfg.train.max_train_seconds = 120.0
        cfg.cv.enabled = False
        cfg.tuning.enabled = False
        cfg.benchmark.n_single, cfg.benchmark.warmup = 60, 10
        for k, v in QUICK_PARAMS.items():
            cfg.train.model_params.setdefault(k, {}).update(v)
    if a.demo:
        cfg.data.paths = [str(DEMO_DIR)]
    return cfg


# ----------------------------------------------------------------------------------- one model
def run_model(spec, prepared, pieces, cfg, device, folds, exp_dir, mode, group_key):
    key = spec.key
    row = {"model": key, "label": spec.label, "family": spec.family, "input_kind": spec.input_kind}
    record = {}
    seed_everything(cfg.train.seed)
    params = {}
    if cfg.tuning.enabled and spec.space and (not cfg.tuning.models or key in cfg.tuning.models):
        LOG.info("  tuning %s (%d candidates x %d folds)", key, cfg.tuning.n_iter + 1, cfg.tuning.folds)
        tf = folds.get("tune", pieces["train"], cfg.tuning.folds, spec.input_kind == "features")
        params, trials = random_search(spec, tf, cfg, device)
        row["tuned_params"] = json.dumps(to_jsonable(params))
        record["tuning_trials"] = trials

    seed_everything(cfg.train.seed)
    model = build_model(spec, prepared, cfg, device, params)
    train = prepared.parts["train"].data(spec.input_kind)
    val = prepared.parts["val"].data(spec.input_kind) if "val" in prepared.parts else None
    with PeakMemory() as mem, Timer() as t:
        model.fit(train, val)
    row["train_time_s"] = t.seconds
    row["train_peak_mem_mb"] = mem.peak_mb
    row["n_params"] = model.n_params()
    LOG.info("  trained in %.1fs (peak +%.0f MB)", t.seconds, mem.peak_mb)

    overlap = cfg.stride_samples / cfg.window_samples
    if val is not None:
        v = evaluate(val["y"], model.predict_proba(val["X"]), prepared.classes)
        row["val_accuracy"], row["val_f1_macro"] = v["metrics"]["accuracy"], v["metrics"]["f1_macro"]
    test_part = prepared.parts["test"]
    ev = evaluate(test_part.y, model.predict_proba(test_part.data(spec.input_kind)["X"]), prepared.classes,
                  groups=test_part.meta, overlap_factor=overlap)
    row.update(ev["metrics"])
    row.update(group_spread(ev["by_group"].get(group_key, {})))
    if row.get("val_f1_macro") is not None:
        row["val_test_f1_gap"] = row["val_f1_macro"] - row["f1_macro"]
    LOG.info("  test: acc %.3f  bal.acc %.3f  macro-F1 %.3f", row["accuracy"], row["balanced_accuracy"], row["f1_macro"])

    clf = GestureClassifier.from_training(model, prepared, cfg, key)
    if cfg.output.save_models:
        clf.save(exp_dir / "models" / key)
    if cfg.benchmark.enabled:
        windows = test_part.windows
        step = max(1, len(windows) // 256)                         # a spread of test windows
        bench = benchmark_classifier(clf, windows[::step][:256], cfg.benchmark, cfg.budget_ms, exp_dir,
                                     cfg.train.torch_threads)
        row.update(bench)
        LOG.info("  CPU latency: median %.2f ms, P95 %.2f ms (%s), size %.0f KB",
                 bench["lat_median_ms"], bench["lat_p95_ms"], bench["realtime_status"], bench["model_size_kb"])

    if cfg.cv.enabled and cfg.cv.scope != "none" and (cfg.cv.scope == "all" or spec.family == "ml"):
        cvf = folds.get("cv", pieces["train"] + pieces["val"], cfg.cv.folds, spec.input_kind == "features")
        row.update(cross_validate(spec, cvf, cfg, device, params))
        if "cv_f1_mean" in row:
            LOG.info("  %d-fold group CV macro-F1: %.3f +/- %.3f", row["cv_folds"], row["cv_f1_mean"], row["cv_f1_std"])

    record.update(row=row, per_class=ev["per_class"], confusion_matrix=ev["confusion_matrix"],
                  by_group=ev["by_group"], history=model.history)
    return record


# ------------------------------------------------------------------------------------- aggregate
def load_records(exp_dir: Path):
    recs = {}
    for f in sorted((exp_dir / "per_model").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        recs[d["row"]["model"]] = d
    return recs


def finalize(exp_dir, cfg, classes, data_summary, split_info, window_counts, skipped, errors, demo):
    recs = load_records(exp_dir)
    if not recs:
        LOG.error("no model finished - nothing to report")
        return None
    df = pd.DataFrame([r["row"] for r in recs.values()])
    budget = cfg.budget_ms
    df = add_scores(df, cfg.ranking, budget)
    df.to_csv(exp_dir / "model_comparison.csv", index=False)
    df.sort_values("perf_score", ascending=False).to_csv(exp_dir / "leaderboard.csv", index=False)
    save_json(df.to_dict("records"), exp_dir / "metrics.json")
    rec_list = recommendations(df, cfg.ranking, split_info["mode"], budget)
    save_json(rec_list, exp_dir / "recommendations.json")

    plots = exp_dir / "plots"
    per_class = {k: v["per_class"] for k, v in recs.items()}
    pc_rows = [dict(model=k, **r) for k, v in per_class.items() for r in v]
    pd.DataFrame(pc_rows).to_csv(exp_dir / "per_gesture_metrics.csv", index=False)
    jobs = [
        ("accuracy_ranking", lambda: viz.bar_ranking(df, "accuracy", "Accuracy (test)", plots / "accuracy_ranking.png")),
        ("f1_ranking", lambda: viz.bar_ranking(df, "f1_macro", "Macro F1-score (test)", plots / "f1_macro_ranking.png")),
        ("perf_ranking", lambda: viz.bar_ranking(df, "perf_score", "Overall performance score", plots / "performance_score_ranking.png")),
        ("train_time", lambda: viz.bar_ranking(df, "train_time_s", "Training time (s, log)", plots / "training_time.png", ascending=True, log=True, fmt="{:.1f}")),
        ("latency", lambda: viz.bar_ranking(df, "lat_p95_ms", "P95 single-window latency (ms, CPU, log)", plots / "latency_p95.png", ascending=True, log=True, fmt="{:.2f}")),
        ("latency_med", lambda: viz.bar_ranking(df, "lat_median_ms", "Median single-window latency (ms, CPU, log)", plots / "latency_median.png", ascending=True, log=True, fmt="{:.2f}")),
        ("size", lambda: viz.bar_ranking(df, "model_size_kb", "Model size on disk (KB, log)", plots / "model_size.png", ascending=True, log=True, fmt="{:.0f}")),
        ("throughput", lambda: viz.bar_ranking(df, "batch32_throughput_wps", "Throughput, batch of 32 (windows/s, log)", plots / "throughput.png", log=True, fmt="{:.0f}")),
        ("metrics_heatmap", lambda: viz.metrics_heatmap(df, plots / "metrics_heatmap.png")),
        ("per_gesture_heatmap", lambda: viz.per_gesture_heatmap(per_class, df, plots / "per_gesture_f1_heatmap.png")),
        ("per_gesture_top", lambda: viz.per_gesture_top(per_class, df, plots / "per_gesture_top_models.png")),
        ("radar", lambda: viz.radar(df, plots / "radar_top_models.png", cfg.output.top_n_radar)),
        ("tradeoff", lambda: viz.tradeoff(df, budget, plots / "tradeoff_f1_vs_latency.png") if "pareto" in df else None),
        ("curves", lambda: viz.training_curves({k: v.get("history") for k, v in recs.items()},
                                               dict(zip(df["model"], df["label"])), plots / "training_curves.png")),
        ("classes", lambda: viz.class_distribution(data_summary["seconds_per_class"], plots / "dataset_class_distribution.png")),
    ]
    plots.mkdir(exist_ok=True)
    for name, fn in jobs:
        try:
            fn()
        except Exception as exc:                                    # a plotting problem must not lose results
            LOG.warning("plot '%s' failed: %s", name, exc)

    best_rt = next((r for r in rec_list if r["criterion"] == "Best model for real-time deployment" and r["model"]), None)
    if best_rt and (exp_dir / "models" / best_rt["model"]).exists():
        dest = exp_dir / "models" / "recommended_realtime"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(exp_dir / "models" / best_rt["model"], dest)
    env = json.loads((exp_dir / "environment.json").read_text(encoding="utf-8"))
    write_summary(exp_dir / "summary_report.md", exp_name=exp_dir.name, df=df, recs=rec_list, data_summary=data_summary,
                  split_info=split_info, window_counts=window_counts, skipped=skipped, errors=errors, cfg=cfg,
                  env=env, budget_ms=budget, demo=demo)
    return df, rec_list


# ------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dump_config:
        print(json.dumps(ExperimentConfig().to_dict(), indent=2))
        return 0
    cfg = build_config(a)
    if a.list_models:
        run, skipped = select_models("all", ["ml", "dl", "hybrid"], [])
        ok = {s.key for s in run}
        for k, s in SPECS.items():
            print(f"{k:18s} {s.family:7s} {s.label:22s} {'ready' if k in ok else 'MISSING: ' + ', '.join(m for m in s.requires if not have(m))}")
        return 0

    root = Path(cfg.output.root)
    root = root if root.is_absolute() else HERE / root
    if a.resume and not cfg.output.name:
        sys.exit("--resume needs --name <existing experiment folder>")
    exp_dir = next_experiment_dir(root, cfg.output.name)
    for sub in ("per_model", "confusion_matrices", "models", "plots"):
        (exp_dir / sub).mkdir(exist_ok=True)
    setup_logging(exp_dir / "training.log")
    cfg.save(exp_dir / "config.json")
    save_json(environment_info(), exp_dir / "environment.json")
    LOG.info("experiment folder: %s", exp_dir)
    seed_everything(cfg.train.seed)

    if a.demo and not any(DEMO_DIR.glob("*.csv")):
        from .synthetic import generate_dataset
        LOG.info("generating SYNTHETIC demo data in %s ...", DEMO_DIR)
        generate_dataset(DEMO_DIR, n_subjects=5, n_sessions=2, seconds=5.0)

    # ---- data
    recs, report = load_dataset(cfg.data)
    save_json(report, exp_dir / "data_report.json")
    if not recs:
        LOG.error("No usable labelled recordings found (%d files seen, %d skipped).", report["files_found"], len(report["skipped"]))
        for s in report["skipped"][:10]:
            LOG.error("  %s: %s", s["file"], s["reason"])
        LOG.error("Gesture labels come from a label column or from the file name, e.g. "
                  "emg_20261108_185041_S02_sess1_closed_fist_trial3.csv (the 'label' box of the app's raw recorder). "
                  "Use --data-path to point at labelled recordings, or --demo for synthetic data.")
        return 2
    recs = filter_rare_classes(recs, cfg)
    classes = sorted({lab for r in recs for _, _, lab in r.runs})
    if len(classes) < 2:
        LOG.error("Need at least 2 gesture classes, found %s", classes)
        return 2
    data_summary = summarize_dataset(recs)
    data_summary["classes"] = classes
    save_json(data_summary, exp_dir / "data_summary.json")
    LOG.info("dataset: %d recordings, %d subject(s), %d class(es): %s", len(recs), len(data_summary["subjects"]), len(classes), classes)

    # ---- split (before windowing) + preprocessing fitted on train only
    pieces, split_info = split_pieces(recs, cfg.split, cfg.window_samples)
    models, skipped = select_models(cfg.models.enabled, cfg.models.families, cfg.models.exclude)
    for k, why in skipped:
        LOG.warning("skipping %s: %s", k, why)
    if not models:
        LOG.error("no runnable models selected")
        return 2
    need_feat = any(m.input_kind == "features" for m in models)
    prepared = prepare(recs, pieces, classes, cfg, need_feat, keep_windows=("test",))
    if "test" not in prepared.parts or "val" not in prepared.parts:
        LOG.error("the split left the validation or test set empty - record more data")
        return 2
    window_counts = {k: int(len(p.y)) for k, p in prepared.parts.items()}
    split_info["window_counts"] = window_counts
    split_info["class_windows"] = {k: {classes[c]: int((p.y == c).sum()) for c in range(len(classes))} for k, p in prepared.parts.items()}
    save_json(split_info, exp_dir / "split_manifest.json")
    LOG.info("split '%s': windows %s (feature dim %d)", split_info["mode"], window_counts, prepared.n_features)
    missing = [classes[c] for c in range(len(classes)) if (prepared.parts["test"].y == c).sum() == 0]
    if missing:
        LOG.warning("gestures absent from the test set (metrics cover the rest): %s", missing)

    device = resolve_device(cfg.train.device)
    LOG.info("device: %s | models: %s", device, [m.key for m in models])
    mode = split_info["mode"]
    group_key = {"subject": "subject", "session": "session", "recording": "rec"}.get(mode, "subject")
    folds = FoldCache(recs, classes, cfg, mode)

    # ---- model loop: failures are isolated, results written after every model
    errors = {}
    for i, spec in enumerate(models, 1):
        out = exp_dir / "per_model" / f"{spec.key}.json"
        if a.resume and out.exists():
            LOG.info("[%d/%d] %s - already done, skipping (--resume)", i, len(models), spec.label)
            continue
        LOG.info("[%d/%d] %s (%s, input: %s)", i, len(models), spec.label, spec.family, spec.input_kind)
        try:
            rec = run_model(spec, prepared, pieces, cfg, device, folds, exp_dir, mode, group_key)
            save_json(rec, out)
            cm = rec["confusion_matrix"]
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(exp_dir / "confusion_matrices" / f"{spec.key}.csv")
            try:
                viz.confusion_matrix_plot(np.array(cm), classes, f"{spec.label} - confusion matrix (test)",
                                          exp_dir / "confusion_matrices" / f"{spec.key}.png")
            except Exception as exc:
                LOG.warning("  confusion plot failed: %s", exc)
            with open(exp_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(to_jsonable(rec["row"])) + "\n")
        except Exception as exc:                                   # noqa: BLE001 - isolate every model
            errors[spec.key] = f"{type(exc).__name__}: {exc}"
            LOG.error("  %s FAILED: %s", spec.key, errors[spec.key])
            LOG.debug(traceback.format_exc())
            save_json(errors, exp_dir / "errors.json")

    res = finalize(exp_dir, cfg, classes, data_summary, split_info, window_counts, skipped, errors, a.demo)
    if res is None:
        return 1
    df, rec_list = res
    LOG.info("=" * 78)
    LOG.info("DONE: %d/%d models -> %s", len(df), len(models), exp_dir)
    for r in rec_list:
        LOG.info("  %-52s %s", r["criterion"], r["label"] or "n/a")
    LOG.info("report: %s", exp_dir / "summary_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
