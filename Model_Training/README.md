<div align="center">

# 🧠 Model_Training

**Train, compare and benchmark 25 EMG gesture models: one command, or a desktop Training Studio**

[↑ Project](../README.md) · [Machine-learning guide](../docs/MACHINE_LEARNING.md)

</div>

## Where things are

| Folder / file | What it is |
|---|---|
| [`model_train.py`](model_train.py) + the modules next to it | the EMG benchmark **CLI** (data → splits → windows → features → models → report) |
| [`app/`](app/README.md) | **Training Studio**, a desktop GUI that trains several models at once and exports `.joblib` bundles |
| [`models/`](models/README.md) | the model registry, classical-ML wrappers and the PyTorch networks |
| [`outputs/`](outputs/README.md) | everything the Training Studio produces (trained models, reports, plots) |
| [`results/`](results/README.md) | experiment folders written by the CLI benchmark |
| [`datasets/`](datasets/README.md) | sample tabular datasets for the Studio |
| [`data_demo/`](data_demo/README.md) | synthetic EMG recordings for smoke tests |
| [`logs/`](logs/README.md) | training logs |
| [`tests/`](tests/README.md) | training-side tests |

**The whole story of the data, preprocessing, features and models is in [docs/MACHINE_LEARNING.md](../docs/MACHINE_LEARNING.md).**

---

# Model_Training - EMG gesture model benchmark

Trains, evaluates, compares and benchmarks **25 models** for real-time EMG gesture classification with one command.
It lives next to the app and **does not import anything from `src/emg_arm`** (the app, in turn, loads the bundles written here); it has its own dependency list and virtual environment.

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

Exported bundles work in the desktop app directly: **Gesture model → Upload .joblib files… → Use selected model**.

```python
from Model_Training.app.inference import load_gesture_classifier      # bundles from the Training Studio
# or: from Model_Training.inference import GestureClassifier           # bundles from the CLI benchmark
clf = load_gesture_classifier("Model_Training/outputs/trained_models/LDA_20261124_143936.joblib")
label, confidence = clf.predict_window(window)     # window: (samples, channels) raw ADC counts, clf.window_samples long
```
A bundle holds the classifier, filter settings, feature definition, normalisation statistics and class names, so nothing else is needed.
`label, confidence` has the same shape as the app's `prediction_ready(str, float)` signal, and the app's `ml/inference.py` calls `predict_window()` for these bundles
(older joblibs with `model` + `class_names` still use the app's own features). Gesture names are passed through as trained, so use labels the app's `resolve_gesture` understands
(`left`, `closed_fist`, `thumbs_up`, ...) or `data.gesture_aliases`. A model that needs XGBoost / LightGBM / CatBoost / PyTorch needs that library in the app's `.venv` too.

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

---

# EMG session folders (BioWave recordings) in the GUI

In the app choose **Data source -> EMG session folders (BioWave)** and open the folder that contains your session folders
(e.g. `BioWaveEMG_ArmBand/dataset`, or a single `S102_day01_right_...Z` folder). The same flow applies: *Data -> Configure -> Train -> Results*,
with the same parallel engine, `.joblib` export and reports. CLI equivalent: `python Model_Training/model_train.py --data-path <folder> --split-mode trial`.

## What is read from a session folder (`session_io.py`)

`signals_*.csv` (samples) + `metadata_*.json/.txt` (participant, sampling rate, which columns are EMG, labels) + `quality_*.json` (packet loss).
Legacy recordings (`Timestamp_ms/Packet_Number`, 11 channels) and folders that only hold `raw_*.bin` are listed on the *Skipped folders* tab with the reason, never silently dropped.
On the sample dataset: 34 session folders loaded (18 subjects, 19 left / 15 right hand, 8 EMG channels, 500 Hz, 10 gestures incl. `rest`, ~106 usable minutes).

## Preprocessing - what is done and why

| Step | Reason |
|---|---|
| Only `Ch1-Ch8` (type `emg` in the channel map) are used | the IMU angles identify the pose/trial, not the muscle activity |
| Samples with `Valid == 0` (lost packets, NaN) are masked; no window may contain one | interpolating invented data would be learned as signal |
| Only the `perform` and `rest` phases are labelled; the `prepare` cue, `paused`, `switch_hand`, `end` never are | they are not gestures |
| 0.5 s / 0.25 s trimmed from the start / end of every gesture, 1.0 s / 0.25 s from every rest | reaction time and relaxation after the gesture are not the gesture |
| Rest is thinned to <= 2 x the mean gesture duration per session | rest follows every gesture; otherwise it is ~half of all data |
| Sessions with < 90 % valid samples or a dead channel are skipped | listed with the reason |
| Per window: DC removal, 20-220 Hz Butterworth band-pass (optional 50/60 Hz notch) | identical code in training and in the real-time classifier (no train/deploy skew) |
| Time + frequency features (classical models) or filtered raw windows (deep models); z-score normalisers fitted on **train windows only** | no leakage |
| Split by trials / people **before** windowing, with hard checks that no sample or group is shared | overlapping windows are near-copies of each other |

**Evidence-based choices** (macro-F1 averaged over 8 sessions, personal model, held-out later repetitions): band-pass only 0.757-0.792, band-pass + 50 Hz notch 0.748-0.773 - the notch did **not** help (mains hum is
strong on many channels but short 200 ms windows ring on a narrow notch), so it is off by default but selectable. A 300 ms window gave +0.01 to +0.03 over 200 ms, at +100 ms latency; 200 ms is the default.

## Evaluation protocols - what the score means (choose on the Configure tab)

| Test set = | Meaning | Measured on this dataset (macro-F1, 10 classes, chance 0.10) |
|---|---|---|
| **New people** (subject-independent) | a stranger puts the band on, nothing calibrated | **~0.30-0.33** (RF 0.32, LDA 0.33, 1D CNN 0.30, quick runs) - 3x chance but not usable; per-session feature normalisation lifts it to ~0.37 |
| **Same people, later repetitions** (`trial`) | a user's own early trials train the model, later trials are tested | **0.72** SVM-RBF (0.76 with real-time smoothing), RF 0.71, k-NN 0.67, MLP 0.67, LDA 0.46 - one pooled model for all 34 sessions |
| Personal model per session (script experiment) | one model per person/hand/session | **0.75-0.80** (LDA / SVM / Extra Trees) - best, and what a calibration step would give |
| Interleaved time blocks (`temporal`) | blocks of the same trial in train and test | optimistic - shown with a warning |

Take-aways for real use: **calibrate per user** (record the user's own repetitions and train/fine-tune on them); a single pooled model across people is the weakest option on dry-electrode data.
A very high score (e.g. 1.00) on this kind of data means leakage - typically a label-like column used as a feature or a random row split.
Deep models are trained with Adam (or AdamW, selectable) + one-cycle learning-rate schedule + early stopping; with few epochs they stay below the classical models here - raise *Deep-learning epochs*.

## Real-time numbers in the results table

* **Smoothed F1** - simulates the live loop: the decision is the argmax of the mean probability of the last *k* windows (default 5), history reset at each trial. Decision delay = window + (k-1) x stride.
  It does not include the transition lag between two gestures.
* **Latency P95 (ms)** - median/P95 of the whole deployed pipeline per window (filter + features + model), 1 CPU thread, from the existing `benchmark.py`; it runs while other models train, so treat it as an upper bound.
  The budget it is judged against is the *decision interval* (default 50 ms).
* **Worst subject F1** - lowest macro-F1 among held-out subjects; **Size** - the `.joblib` on disk (a 200-tree forest is ~90 MB, an SVM ~20 MB, an MLP/CNN well under 1 MB).

## Use an exported EMG model

```python
from Model_Training.app.inference import load_gesture_classifier
clf = load_gesture_classifier("Model_Training/outputs/trained_models/SVMRBF_20261123_101500.joblib")
label, confidence = clf.predict_window(window)   # window: (clf.window_samples, 8) raw ADC counts; filtering/features/normalisation happen inside
clf.classes, clf.fs, clf.window_samples, clf.stride_samples
```
The bundle (`format: emg_gesture_classifier`) also carries the settings, split, metrics and library versions (`metadata`, mirrored in the sidecar `.json`). Each export is reloaded and checked to reproduce its probabilities.

## Limitations

* Cross-person generalisation is weak on this data (see table); per-user calibration is the practical fix and is not automated in the GUI yet.
* Pooled training treats left- and right-hand sessions as one distribution; training one model per hand/user may do better.
* The 3 `raw_*.bin`-only folders and the legacy recordings are not read.
