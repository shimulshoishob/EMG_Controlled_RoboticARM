<div align="center">

# 🎙️ `recordings/`

**Raw EMG/IMU recordings made in the app**

[↑ Project](../README.md) · [How recordings become training data](../docs/MACHINE_LEARNING.md#2-data)

</div>

Created by **Settings & data → Record raw signals** while the armband is connected. One CSV per recording:

```text
emg_<YYYYMMDD>_<HHMMSS>[_<label>].csv
host_time_s, packet_seq, frame_id, emg1 … emg8, roll, pitch, yaw
```

- `emg1…emg8`: raw ADC counts (0-4095) at 500 Hz; `roll, pitch, yaw`: degrees.
- Give the recording a **label** such as `S02_sess1_closed_fist_trial3` (subject, session, gesture, trial): the training pipeline reads the gesture, subject and session from the file name.
- A recording **without a label in its name cannot be used for supervised training**.

For multi-gesture studies, record with the BioWave recording app instead; it produces *session folders* (`signals_*.csv` with `Phase`/`Label`/`Valid`, plus events, metadata and quality files), which the training pipeline prefers.

Train from here: `python Model_Training/model_train.py --data-path recordings` (see the [Model_Training README](../Model_Training/README.md)).
