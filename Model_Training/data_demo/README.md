<div align="center">

# 🎭 `data_demo/`

**Synthetic EMG recordings for smoke tests**

[↑ Model_Training](../README.md)

</div>

About 100 small CSV files that mimic the raw recording format (same header, ADC-count scale, 500 Hz) and the file-naming convention:

```text
emg_<date>_<time>_S<subject>_sess<n>_<gesture>_trial<n>.csv      e.g. emg_20261101_100100_S01_sess1_left_trial1.csv
```

Subjects `S01…`, two sessions each, the ten gestures. They are produced by [`synthetic.py`](../synthetic.py) so the whole pipeline (discovery → splits → models → report) can be exercised without a labelled dataset.

> ⚠️ **Scores on this data say nothing about real gesture recognition.** Never train a model for the arm on it.
