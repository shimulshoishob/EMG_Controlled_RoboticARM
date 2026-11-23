# Model_Training - EMG gesture model benchmark

Trains, evaluates, compares and benchmarks **25 models** for real-time EMG gesture classification with one command.
It lives next to the app and **does not modify or import anything from `src/emg_arm`**; it has its own dependency list and virtual environment.

| Family | Models (key) |
|---|---|
| Classical ML (engineered time + frequency features) | Random Forest `rf`, Extra Trees `extra_trees`, XGBoost `xgboost`, LightGBM `lightgbm`, CatBoost `catboost`, SVM-RBF `svm_rbf`, SVM-linear `svm_linear`, k-NN `knn`, Logistic Regression `logreg`, LDA `lda`, Gradient Boosting `gradient_boosting`, MLP `mlp` |
| Deep learning (raw windows) | 1D CNN `cnn1d`, 2D CNN `cnn2d`, LSTM `lstm`, BiLSTM `bilstm`, GRU `gru`, Transformer `transformer` |
| Hybrid | CNN-LSTM `cnn_lstm`, CNN-GRU `cnn_gru`, CNN-BiLSTM `cnn_bilstm`, CNN-TCN `cnn_tcn`, CNN-Attention `cnn_attention`, CNN-Transformer `cnn_transformer`, CNN+Random Forest `cnn_rf` |

Each architecture exists once. `cnn_rf` = a 1D CNN trained end-to-end, then frozen; its 64-d embeddings are the input of a Random Forest that makes the decision (feature extraction + classifier, not an unjustified vote of two models).

## Run it

```powershell
cd D:\RoboArm\EMG_Controlled_RoboticARM
python -m venv Model_Training\.venv
Model_Training\.venv\Scripts\python -m pip install -r Model_Training\requirements.txt
# CPU torch (smaller): ... pip install torch --index-url https://download.pytorch.org/whl/cpu

Model_Training\.venv\Scripts\python Model_Training\model_train.py --demo --quick   # smoke test, SYNTHETIC data, minutes
Model_Training\.venv\Scripts\python Model_Training\model_train.py                  # all models on .\recordings
Model_Training\.venv\Scripts\python Model_Training\model_train.py --data-path D:\emg --models rf,svm_rbf,cnn1d,cnn_gru
Model_Training\.venv\Scripts\python Model_Training\model_train.py --list-models    # what is installed
Model_Training\.venv\Scripts\python Model_Training\model_train.py --dump-config > my.json   # edit, then --config my.json
```
Useful flags: `--families ml,dl,hybrid`, `--exclude catboost`, `--split-mode subject|session|recording|temporal`, `--tune`, `--no-cv`,
`--epochs N`, `--window-ms`, `--stride-ms`, `--device cpu|cuda`, `--name my_run --resume` (continue an interrupted run, finished models are skipped).
A missing library only skips its models (with the `pip install` hint); a crashing model is logged and the run continues; results are written after every model.

## Data you need (important)

The existing `recordings/emg_20261108_185041_s011.csv` has **no gesture label**, so it cannot be used for supervised training
(the loader reports it as skipped). Gesture, subject and session are taken from, in order:

1. columns in the CSV: `label`/`gesture`/`class` (per-sample, windows never straddle a label change), `subject`, `session` (names configurable), or
2. the file name - exactly what the app's *Record raw signals -> label* box produces:
   `emg_<date>_<time>_<label>.csv` with label like `S02_sess1_closed_fist_trial3`
   (`S..` subject, `sess..`/`day..` session, `trial..` ignored, the remaining words are the gesture). A regex can override this (`data.filename_pattern`).

Channels are auto-detected (`emg1..emg8`, `ch1..`, ...) or set via `data.channel_columns`; the sampling rate is taken from `data.sampling_rate` (500 Hz) and checked against the time column.
Invalid files (unreadable, too short, >10 % missing, dead channel, wrong channel count, no label) are skipped and listed in `data_report.json`; small gaps are interpolated.

**How much data:** for an honest subject-independent result you need **>= 3 subjects** (5+ is far better); with fewer, `auto` falls back to session -> recording -> temporal splitting and the report says exactly what that does and does not prove.
Record every gesture the same number of times per subject.

## Fair & leak-free evaluation

* Splitting happens on recordings/segments **before** windowing; windows never cross a split boundary (`splits.py` hard-asserts no shared samples / groups).
* `auto` picks the strictest possible split: subject -> session -> recording -> temporal (disjoint time blocks). There is deliberately no random-window split.
* Filtering is per window (same code at train and deploy time). Channel normalisation, feature scaling and class weights are fitted on the training split only.
* All models see the same windows, split, seed and test set. Test data is never used for early stopping, tuning or model choice.
* Optional group-aware cross-validation (`cv`, default on for classical models) and random-search tuning (`--tune`, group-aware CV on the training split only; the defaults are always a candidate so tuning cannot make a model worse on CV).
* Class imbalance: balanced class weights in every model that supports them; metrics include balanced accuracy and macro scores.

## Output: `results/experiment_NNN/`

`summary_report.md` (start here) | `model_comparison.csv`, `leaderboard.csv`, `metrics.json`, `per_gesture_metrics.csv`, `recommendations.json` |
`plots/` (accuracy/F1/score rankings, training time, latency, size, throughput, metric + per-gesture heatmaps, radar, F1-vs-latency trade-off with Pareto front, training curves) |
`confusion_matrices/` (PNG + CSV per model) | `models/<key>/` (deployable bundles) and `models/recommended_realtime/` | `per_model/` | `config.json`, `environment.json`, `split_manifest.json`, `data_report.json`, `training.log`.

Metrics per model: accuracy, balanced accuracy, macro/weighted precision/recall/F1, one-vs-rest macro ROC-AUC (softmax scores for SVMs), per-gesture P/R/F1/AUC, per-held-out-group scores, training time and peak memory, parameters, size on disk,
CPU latency of the **whole deployed pipeline** (filter + features + model) per window - mean, median, P95, P99, max -, model-only and preprocessing-only latency, batch latency, throughput, load time, inference peak memory.
Latency is measured single-threaded by default (`benchmark.threads`). Real-time budget defaults to the stride (50 ms): P95 <= 25 % of budget = *comfortable*, <= 100 % = *marginal*, else *too slow*.

Recommendations (never accuracy alone): best overall (blend of macro-F1 / balanced-acc / accuracy), best macro-F1, best subject-independent generalisation (only when subjects were held out - otherwise labelled as not measurable),
fastest viable, best real-time (macro-F1 + latency headroom + size, over-budget models excluded), best lightweight, best deep, best hybrid. Weights are in `ranking` of the config.

## Using a trained model in the real-time app

```python
from Model_Training.inference import GestureClassifier
clf = GestureClassifier.load(r"Model_Training/results/experiment_001/models/recommended_realtime")
label, confidence = clf.predict_window(window)     # window: (samples, channels) raw ADC counts, clf.window_samples long
```
A bundle holds the classifier, filter settings, feature definition, normalisation statistics and class names, so nothing else is needed.
`label, confidence` has the same shape as the app's `prediction_ready(str, float)` signal; the app's `ml/inference.py` currently expects its own
joblib format with its own features, so wiring this in means replacing the single `extract_window_features(...)` + `predict_proba(...)` call in `InferenceWorker.run` with `clf.predict_window(win)`
(not done here on purpose: the existing app was left untouched). Gesture names are passed through as trained, so use labels the app's `resolve_gesture` understands (`left`, `closed_fist`, `thumbs_up`, ...) or `data.gesture_aliases`.

## Layout

```
model_train.py   entry point (CLI + orchestration)      config.py       all settings (JSON round-trip)
data_io.py       discovery / loading / validation        preprocessing.py window filter, sliding windows, train-fitted scalers
features.py      time + frequency features               splits.py       leakage-safe splits and CV folds
pipeline.py      recordings -> model-ready arrays        crossval.py     CV + random-search tuning
models/          registry, sklearn wrappers, torch nets  evaluation.py   metrics
benchmark.py     latency / memory / size / load time     ranking.py      criteria + recommendations
inference.py     GestureClassifier (deployment)          visualization.py / report.py   figures + markdown
synthetic.py     fake data for smoke tests               tests/          python -m unittest discover Model_Training/tests
```

## Honest limitations

* Without labelled multi-subject data, scores are indicative only; `--demo` data is synthetic and proves nothing about real gestures.
* Deep models are not tuned and not cross-validated by default (cost); they use early stopping on the validation split with a wall-clock cap (`train.max_train_seconds`).
* Hardware here (6-core Ryzen, no CUDA torch): deep models train on CPU; latencies are CPU/single-thread numbers for *this* machine.
* In `cnn_rf` the forest is trained on embeddings of windows the CNN already fitted, so it cannot repair CNN overfitting.
* GPU is used for deep models (and XGBoost) only when CUDA is available; benchmarks are always CPU.

---

# Model Training Studio (GUI application) - `app/`

A desktop app that trains **several of the models above at the same time** on any tabular CSV, compares them,
and exports every successful model as a reusable `.joblib` file. It reuses `models/registry.py` (all 25 models,
their factories and defaults), `evaluation.evaluate`, `config.TrainConfig` and `utils` - nothing was duplicated.
`model_train.py` (the EMG benchmark CLI) is untouched and still works.

## Install and start

```bash
cd EMG_Controlled_RoboticARM
python3.12 -m venv Model_Training/.venv && source Model_Training/.venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r Model_Training/requirements.txt
python Model_Training/app/main.py                       # optionally: ... main.py path/to/data.csv
```
Developed and tested on Python 3.12 (macOS). XGBoost / LightGBM / CatBoost / PyTorch are optional: a model whose library is missing is greyed out
with the reason (hover the row) and everything else still works. Two ready-made datasets are in `datasets/`
(`sample_breast_cancer_classification.csv`, `sample_diabetes_regression.csv`).

## Using it

1. **Data** - *Open CSV*. You get rows / columns / missing cells / duplicate rows, every column's dtype, missing %, unique count and example,
   and a statistical summary. Pick the **target column** (the last column is pre-selected). Numeric vs categorical features are detected
   automatically (numbers stored as text are recognised, ID-like and free-text columns are set to `ignore`); change any column's *Role* if you disagree.
   The task (classification / regression) is auto-detected and can be overridden on the next tab.
2. **Configure** - split size and seed, duplicate removal, constant-feature removal, imputation, scaling, one-hot/ordinal encoding,
   optional `SelectKBest`, class balancing, how many models run at once, and the model checklist (*Select all*, *Classical ML only*, ...).
3. **Train** - *Start training*. Each model is a separate **process** with its own row: **Pending -> Training -> Completed / Failed / Cancelled**,
   progress, duration and, when finished, its headline metric and exported file (double-click a row's file to reveal it).
   *Cancel all* or right-click a row -> *Cancel this model*. A failing model never stops the others; its error is shown and logged.
4. **Results** - best-model banner, ranked comparison table, metric bar charts, time and size charts, confusion matrix (classification) or
   predicted-vs-actual (regression) per model, list of exported files, and buttons for the folders and log.

**Themes:** the *Theme* box (top right) switches between **Light**, **Dark** and **Pink**. The choice is remembered; on first start it follows the macOS appearance.
Every colour (palette, stylesheet, charts) comes from one token set in `gui.py::THEMES`, so text stays readable in macOS Light and Dark mode. Saved report PNGs are always light.

## Which models run on a generic CSV

| Task | Models |
|---|---|
| Classification | all 25 of the registry: 12 classical (consume the preprocessed feature matrix) + 6 deep + 7 hybrid. The deep/hybrid nets were written for `(N, channels, time)` EMG windows; a tabular row is fed as a **1-channel sequence of length n_features** (the 2-D CNN gets the row padded into a 4 x T grid). Needs PyTorch. |
| Regression | the 11 classical models as their regressor equivalents with the registry's default hyper-parameters (the registry itself is classification-only): RF, Extra Trees, XGBoost, LightGBM, CatBoost, SVR (RBF / linear), k-NN, Ridge (in place of logistic regression), Gradient Boosting, MLP. LDA and the cross-entropy networks are classification-only. |

Deep models on small tabular data are usually worse than tree/SVM models (see the sample results) - they are included because they are in the registry, not because they are recommended for tables.

## Metrics

Classification: accuracy, precision, recall, F1 (macro; weighted ones are in the JSON), confusion matrix, one-vs-rest macro ROC-AUC
(SVMs use softmax of the decision scores, noted in the report). Regression: MAE, MSE, RMSE, R². Always: training time (preprocessing + fit),
test prediction time and exported file size. **Best model = highest macro-F1 (classification) or R² (regression)**, ties broken by training time.

## Leakage and reproducibility

* One train/test split (stratified for classification, `seed`) is made **before** anything is learned; duplicates are removed before it.
* Imputation, scaling, encoding, constant-feature removal and feature selection are **fitted on the training rows only** inside each worker and then applied to the test rows.
* Early stopping (boosters, deep nets) uses a 15 % validation slice carved out of the *training* rows; the test set is only used for the final scores.
* Every model gets the same split and the same preprocessing code; seeds are set in every process.

## Output layout

```
Model_Training/
  datasets/                          sample / your CSVs (the file picker opens here)
  outputs/
    trained_models/<Model>_<run>.joblib + .json       e.g. RandomForest_20261123_101500.joblib  (run id = timestamp, never overwritten)
    evaluation_reports/<run>/        report.md, model_comparison.csv, training_status.csv, <model>.json (full metrics, confusion matrix, per-class)
    plots/<run>/                     scores.png, time_and_size.png, confusion_<model>.png / pred_vs_actual_<model>.png
  logs/training_<run>.log            everything the GUI log pane shows, including full tracebacks of failures
```
Each `.joblib` is a dict: `pipeline` (**preprocessing + model in one scikit-learn `Pipeline`**), `label_encoder`, `classes`, `target`, feature lists and `metadata`
(model + parameters, preprocessing options, seed, split sizes, timings, metrics, library versions). A sibling `.json` holds the same metadata. After saving, each worker
reloads the file and checks that it reproduces the pipeline's predictions (`reload_check: ok`); a model that fails this check is reported as Failed.

## Use an exported model

```python
from Model_Training.app.inference import load_bundle, predict
bundle = load_bundle("Model_Training/outputs/trained_models/RandomForest_20261123_101500.joblib")
labels = predict(bundle, new_dataframe)          # raw columns, same names as the training CSV; labels in the original form
proba  = predict(bundle, new_dataframe, return_proba=True)   # not available for SVMs
```
Without the package: `joblib.load(path)["pipeline"].predict(df)` returns encoded class indices (use `bundle["label_encoder"]` to decode) - pass categorical
columns as strings, numbers as numbers. Deep/hybrid bundles additionally need `torch` and `Model_Training` importable (`load_bundle` handles the path).

## Parallelism, in short

`app/engine.py` starts one spawned process per model (at most *Models trained at once*; the rest wait as *Pending*) and polls them from a GUI timer, so the window never blocks.
CPU threads are split across the running models (`cores / parallel`). Cancelling terminates the processes and deletes half-written output; closing the window while training asks first.

## Code map

`app/main.py` launcher · `gui.py` PySide6 UI · `engine.py` scheduler / cancel / crash isolation · `worker.py` one model's train -> evaluate -> export ·
`preprocess.py` analysis, type inference, split, preprocessing pipeline · `catalog.py` model list from the registry · `regression.py` regressors + metrics ·
`estimators.py` torch-in-sklearn adapter · `reports.py` tables, charts, report files · `inference.py` loading helper · tests: `python -m unittest discover Model_Training/tests`.

## Limitations

* No hyper-parameter tuning or cross-validation in the GUI (defaults from the registry; the EMG CLI has `--tune` and group-aware CV).
* The whole CSV is held in memory; very large files (millions of rows) will be slow for the O(n^2) SVMs.
* A model's *Duration* includes process start-up (a few seconds, mostly importing PyTorch/XGBoost), which dominates on small datasets; the *Train time* column in Results is the real preprocessing + fit time.
