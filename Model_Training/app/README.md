<div align="center">

# 🎛️ Training Studio (`app/`)

**A desktop app that trains several models at once on EMG recordings or any tabular CSV, compares them and exports reusable `.joblib` bundles**

[↑ Model_Training](../README.md) · [Machine-learning guide](../../docs/MACHINE_LEARNING.md)

</div>

```bash
python Model_Training/app/main.py              # optionally: ... main.py path/to/data.csv   (also: python -m Model_Training.app)
```

Four tabs: **Data → Configure → Train → Results** (full walkthrough in the [Model_Training README](../README.md#model-training-studio-gui-application---app)).

## Two kinds of task

| Task | Module | What it does |
|---|---|---|
| **EMG gesture classification** | [`emg.py`](emg.py) | loads BioWave session folders, windows and filters the signals, builds features, splits by trial/session/subject, trains, evaluates with real-time smoothing and latency, exports `emg_gesture_classifier` bundles |
| **Tabular classification / regression** | [`preprocess.py`](preprocess.py), [`regression.py`](regression.py), [`estimators.py`](estimators.py) | generic CSV training with the preprocessing options below (deep and hybrid networks are offered for classification only) |

## Modules

| Module | Role |
|---|---|
| [`gui.py`](gui.py) | the window: tabs, themes (Light / Dark / Pink), charts, tables |
| [`engine.py`](engine.py) | orchestrates a run: one **process per model** (a crashing model never stops the others), progress, cancel, collecting results |
| [`worker.py`](worker.py) | one training task = one OS process: loads the shared split, fits the shared preprocessing, builds the model through the existing factories, trains, evaluates, exports |
| [`catalog.py`](catalog.py) | the model list shown in the GUI, taken straight from `models/registry.py` plus availability (missing libraries are greyed out with the reason) |
| [`emg.py`](emg.py) | the EMG task end to end (see above) |
| [`preprocess.py`](preprocess.py) | dataset analysis, feature-type inference, leakage-safe split **before** any statistic is learned, and the preprocessing pipeline (fitted on training rows only) |
| [`regression.py`](regression.py) | regressor counterparts of the classical models (same families and default settings) |
| [`estimators.py`](estimators.py) | adapter that lets the PyTorch models sit in a scikit-learn `Pipeline` for tabular data |
| [`reports.py`](reports.py) | JSON / CSV / Markdown reports and plots |
| [`inference.py`](inference.py) | `load_bundle`, `predict` (tabular) and `load_gesture_classifier` (EMG) |
| [`paths.py`](paths.py) | the output folder layout |

## Tabular preprocessing options

| Option | Default |
|---|---|
| test split / seed | 0.2 / 42 |
| drop duplicate rows, drop constant features | on |
| numeric imputation / categorical imputation | median / most frequent |
| scaling | standard |
| categorical encoding | one-hot (max 30 categories) |
| feature selection | optional `SelectKBest` |
| class balancing | balanced class weights |

## EMG options (the *Configure* tab)

Window 200 ms, stride 100 ms, band-pass 20-220 Hz, optional notch, trims and rest cap for session folders, split mode (`auto`, `trial`, `session`, `subject`…), validation/test size, optimiser (Adam / AdamW), epochs, batch size, learning rate, weight decay, patience, per-deep-model time cap (600 s), smoothing *k* = 5 and a 50 ms real-time budget. Every option is stored in the exported model's JSON.

## Exported files

`outputs/trained_models/<Model>_<run>.joblib` + `.json` (see [outputs/](../outputs/README.md)). Use them in the desktop app from *Gesture model → Upload .joblib files*.
