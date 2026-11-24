<div align="center">

# 📦 `outputs/`

**Everything the Training Studio produces** (not committed; this README is)

[↑ Model_Training](../README.md)

</div>

```text
outputs/
├── trained_models/              <Model>_<run_id>.joblib  +  <Model>_<run_id>.json
├── evaluation_reports/<run>/    per-model JSON, model_comparison.csv, training_status.csv, report.md
├── plots/<run>/                 confusion matrices and comparison charts (PNG)
└── .work/<run>/                 temporary split files, removed after the run
```

`<run_id>` is a timestamp such as `20261124_143936`; all models trained in one run share it.

## The model files

| File | Contents |
|---|---|
| `<Model>_<run>.joblib` | the bundle: `format = "emg_gesture_classifier"`, the `classifier` (model + preprocessing + normalisation + class names), `classes`, `metadata` |
| `<Model>_<run>.json` | the same metadata as readable text: dataset summary, split, windows per class, settings, test/validation metrics, real-time latency and size, library versions |

Use a model in the desktop app: **Gesture model → Upload .joblib files…**. Pick a model that is accurate **and** comfortable for real time (see `realtime_status` in its JSON).

> Which run is trustworthy? Check the JSON: `dataset.split.mode` should be `trial`, `session` or `subject`, and the features must be the 8 EMG channels only (`n_features` 204). Runs whose JSON lists `Label_Code`, `Trial_ID`, timestamps or `Phase` as features are leaky and must not be used.
