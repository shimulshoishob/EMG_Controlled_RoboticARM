<div align="center">

# 🏁 `results/`

**Experiment folders written by the CLI benchmark** (`python Model_Training/model_train.py`)

[↑ Model_Training](../README.md)

</div>

Each run creates `experiment_NNN/` (or the name you pass with `--name`) containing:

| Item | Contents |
|---|---|
| `summary_report.md` | **start here**: ranking, recommendations, caveats |
| `model_comparison.csv`, `leaderboard.csv`, `metrics.json`, `per_gesture_metrics.csv`, `recommendations.json` | numbers |
| `plots/`, `confusion_matrices/` | figures |
| `models/<key>/`, `models/recommended_realtime/` | deployable bundles (`meta.json`, `preproc.npz`, model file) |
| `config.json`, `environment.json`, `split_manifest.json`, `data_report.json`, `training.log` | exactly what produced the result |

| Folder | What it is |
|---|---|
| `experiment_001`, `experiment_002` | benchmark runs |
| `smoke_all`, `smoke_ml` | quick smoke-test runs on synthetic data (scores mean nothing) |

Load a bundle with `GestureClassifier.load("Model_Training/results/experiment_001/models/recommended_realtime")`.
