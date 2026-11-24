<div align="center">

# 🧪 `Model_Training/tests/`

**Training-side tests**

[↑ Model_Training](../README.md)

</div>

```bash
python -m unittest discover Model_Training/tests
```

| File | Covers |
|---|---|
| [`test_pipeline.py`](test_pipeline.py) | fast checks of the properties that make the benchmark trustworthy (leakage-safe splits, windowing, preprocessing, features, evaluation) |
| [`test_session_emg.py`](test_session_emg.py) | BioWave session-folder loading, the `trial` split, real-time smoothing and the EMG training engine (on synthetic session folders in the real format) |
| [`test_app_engine.py`](test_app_engine.py) | headless tests of the Studio's engine (no Qt needed) |

Application tests are in [`/tests`](../../tests/README.md).
