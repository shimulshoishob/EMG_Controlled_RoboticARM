"""Tests for BioWave session-folder loading, the 'trial' split, real-time smoothing and the EMG training engine.

Run (from the project root):  python -m unittest Model_Training.tests.test_session_emg -v
The fixtures are synthetic session folders in the real format (signals / metadata json+txt / events / quality).
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Model_Training.app import engine as eng
from Model_Training.app.emg import EmgOptions, build_emg_run, causal_smooth, to_config
from Model_Training.app.inference import load_gesture_classifier
from Model_Training.app.paths import Dirs
from Model_Training.config import DataConfig, PreprocessConfig, SplitConfig
from Model_Training.data_io import load_dataset
from Model_Training.preprocessing import WindowPreprocessor
from Model_Training.session_io import find_sessions, load_session
from Model_Training.splits import split_pieces

FS = 500
GESTURES = ["left", "right", "fist_close"]
CODES = "rest:0,left:1,right:2,fist_close:3,aborted:96,paused:97,settle:98,prepare:99"


def make_session(root: Path, subject: str, hand: str, day: int = 1, reps: int = 4, seed: int = 0, bad_packets: bool = True):
    """prepare(1 s) -> perform(2 s) -> rest(2 s) for every repetition of every gesture, like the BioWave recorder."""
    rng = np.random.default_rng(seed + hash(subject + hand) % 1000)
    name = f"{subject}_day{day:02d}_{hand}_20261122T091707Z"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    gain = rng.uniform(0.8, 1.4, 8)                         # per-person electrode gain
    rows, trial = [], 0
    pattern = {g: rng.uniform(20, 120, 8) for g in GESTURES}
    for rep in range(reps):
        for g in GESTURES:
            trial += 1
            for phase, label, dur in (("prepare", "prepare", 1.0), ("perform", g, 2.0), ("rest", "rest", 2.0)):
                n = int(dur * FS)
                amp = pattern[g] * gain if phase == "perform" else np.full(8, 8.0) * gain
                sig = 1750 + rng.normal(0, 1, (n, 8)) * amp
                for i in range(n):
                    rows.append((phase, label, trial, sig[i]))
    n = len(rows)
    sig = np.stack([r[3] for r in rows]).astype(np.float32)
    valid = np.ones(n, dtype=int)
    if bad_packets:
        for s in rng.integers(1000, n - 1000, 6):
            valid[s:s + 10] = 0
            sig[s:s + 10] = np.nan
    df = pd.DataFrame({"Device_Frame_ID": np.arange(n), "Device_Timestamp_us": np.arange(n) * 2000, "Packet_Sequence": np.arange(n) // 5,
                       "Valid": valid, "Late_Flag": 0})
    for c in range(8):
        df[f"Ch{c + 1}"] = sig[:, c]
    df["IMU_roll"], df["IMU_pitch"], df["IMU_yaw"] = 1.0, 2.0, 3.0
    df["IMU_Sample_ID"], df["IMU_Fresh"], df["Host_Received_s"] = 0, 0, 0.0
    df["Label"] = [r[1] for r in rows]
    df["Label_Code"] = [{"rest": 0, "prepare": 99, "left": 1, "right": 2, "fist_close": 3}[r[1]] for r in rows]
    df["Phase"] = [r[0] for r in rows]
    df["Hand"] = hand.capitalize()
    df["Trial_ID"] = [r[2] for r in rows]
    df.to_csv(d / f"signals_{name}.csv", index=False)
    (d / f"events_{name}.csv").write_text("event_index,phase,label\n0,prepare,prepare\n")
    (d / f"quality_{name}.json").write_text(json.dumps({"sample_rate_hz": FS, "valid_sample_ratio": float(valid.mean()),
                                                        "transport": {"loss_pct_packets": 0.03}}))
    meta = {"participant_id": subject, "session_info": {"session_day": day}, "device": {"sampling_rate_hz": FS},
            "channel_map": [{"channel": i + 1, "column": f"Ch{i + 1}", "type": "emg"} for i in range(8)] +
                           [{"channel": 9 + i, "column": c, "type": "imu_orientation"} for i, c in enumerate(["IMU_roll", "IMU_pitch", "IMU_yaw"])],
            "channel_contact": {"channels": [{"channel": 6, "grade": "POOR"}]}}
    (d / f"metadata_{name}.json").write_text(json.dumps(meta))
    (d / f"metadata_{name}.txt").write_text(f"participant_id={subject}\nsession_day={day}\nlabel_codes={CODES}\nrepeats={reps}\n")
    return d


def make_dataset(root: Path, subjects=("S01", "S02", "S03", "S04"), hands=("left", "right")):
    for i, s in enumerate(subjects):
        for h in hands:
            make_session(root, s, h, seed=i)
    (root / "raw_only_S99_day01_left_x").mkdir()
    (root / "raw_only_S99_day01_left_x" / "raw_S99.bin").write_bytes(b"\x00" * 16)
    (root / "legacy_recording").mkdir()
    (root / "legacy_recording" / "emg_data.csv").write_text("Timestamp_ms,Packet_Number,Trial_ID,Ch1,Label\n0,0,1,5,left\n")


class SessionLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_discovery_reports_unusable_folders(self):
        sessions, skipped = find_sessions([str(self.root)])
        self.assertEqual(len(sessions), 8)
        reasons = " ".join(s["reason"] for s in skipped)
        self.assertIn("raw_*.bin", reasons)
        self.assertIn("legacy", reasons)

    def test_labels_phases_trims_and_invalid_samples(self):
        d = next(self.root.glob("S01_day01_left*"))
        rec, info = load_session(d, DataConfig(paths=[str(d)]))
        self.assertEqual(info["status"], "loaded")
        self.assertEqual(rec.subject, "S01")
        self.assertEqual(len(rec.channel_names), 8)                       # IMU columns are not signal channels
        self.assertEqual({l for _, _, l in rec.runs}, {"left", "right", "fist_close", "rest"})   # no 'prepare'
        df = pd.read_csv(next(d.glob("signals_*.csv")))
        sig_valid = (df["Valid"] == 1).to_numpy()
        for a, b, lab in rec.runs:
            self.assertTrue(sig_valid[a:b].all(), "a labelled run must never contain a lost (NaN) sample")
            self.assertNotIn(set(df["Phase"].iloc[a:b]) - {"perform", "rest"}, [{"prepare"}])
            if lab != "rest":                                              # perform runs: 2 s - 0.5 s - 0.25 s of trimming
                self.assertLessEqual(b - a, int(1.3 * FS))
        self.assertEqual(info["poor_channels"], "6")

    def test_rest_is_capped_relative_to_gestures(self):
        d = next(self.root.glob("S01_day01_left*"))
        rec, info = load_session(d, DataConfig(paths=[str(d)], rest_cap_ratio=1.0))
        s = info["seconds_per_class"]
        self.assertLessEqual(s["rest"], 1.6 * np.mean([s[g] for g in GESTURES]))

    def test_load_dataset_dispatches_to_sessions(self):
        recs, rep = load_dataset(DataConfig(paths=[str(self.root)]))
        self.assertEqual(len(recs), 8)
        self.assertEqual(rep["layout"], "sessions")
        self.assertEqual(len(rep["skipped"]), 2)

    def test_session_with_too_many_invalid_samples_is_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            d = make_session(Path(t), "S09", "left", bad_packets=False)
            f = next(d.glob("signals_*.csv"))
            df = pd.read_csv(f)
            df.loc[: int(0.3 * len(df)), "Valid"] = 0
            df.loc[: int(0.3 * len(df)), "Ch1"] = np.nan
            df.to_csv(f, index=False)
            rec, info = load_session(d, DataConfig(paths=[str(d)]))
            self.assertIsNone(rec)
            self.assertIn("valid", info["reason"])


class SplitTests(unittest.TestCase):
    def test_trial_split_holds_out_later_repetitions_without_leakage(self):
        with tempfile.TemporaryDirectory() as t:
            make_dataset(Path(t), subjects=("S01", "S02"))
            recs, _ = load_dataset(DataConfig(paths=[t]))
            pieces, info = split_pieces(recs, SplitConfig(mode="trial", test_size=0.25, val_size=0.25), 100)
            self.assertEqual(info["mode"], "trial")
            for ri in range(len(recs)):
                for lab in {"left", "right", "fist_close"}:
                    start = {k: [p.seg.start for p in v if p.seg.rec_idx == ri and p.seg.label == lab] for k, v in pieces.items()}
                    if start["test"] and start["train"]:
                        self.assertGreater(min(start["test"]), max(start["train"]))      # test is chronologically later
            self.assertTrue(all(len(v) for v in pieces.values()))

    def test_subject_split_never_shares_a_person(self):
        with tempfile.TemporaryDirectory() as t:
            make_dataset(Path(t))
            recs, _ = load_dataset(DataConfig(paths=[t]))
            pieces, info = split_pieces(recs, SplitConfig(mode="subject"), 100)
            who = {k: {recs[p.seg.rec_idx].subject for p in v} for k, v in pieces.items()}
            self.assertFalse(who["train"] & who["test"] or who["train"] & who["val"] or who["val"] & who["test"])


class SignalTests(unittest.TestCase):
    def test_chunked_preprocessing_is_identical(self):
        pre = WindowPreprocessor(500.0, PreprocessConfig(notch_hz=50.0, notch_harmonics=2, notch_q=8.0))
        x = np.random.default_rng(0).normal(1750, 40, (50, 100, 8)).astype(np.float32)
        np.testing.assert_allclose(pre(x, chunk=7), pre(x, chunk=100000), atol=1e-6)
        self.assertEqual(pre.notch.shape[0], 2)                                         # 50 Hz and 100 Hz sections

    def test_causal_smoothing_resets_per_segment_and_averages_history(self):
        p = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)
        seg = np.array([0, 0, 0, 1, 1])
        out = causal_smooth(p, seg, 3)
        np.testing.assert_allclose(out[0], [1, 0])
        np.testing.assert_allclose(out[2], [1 / 3, 2 / 3])
        np.testing.assert_allclose(out[3], [1, 0])                                      # history was reset at segment 1


class EmgEngineTests(unittest.TestCase):
    def test_parallel_training_export_and_realtime_classifier(self):
        with tempfile.TemporaryDirectory() as t:
            data, out = Path(t) / "data", Path(t) / "out"
            make_dataset(data)
            dirs = Dirs(out)
            o = EmgOptions(paths=[str(data)], split_mode="subject", benchmark=True, smoothing_k=3)
            recs, rep = load_dataset(to_config(o).data)
            plan = build_emg_run(recs, rep, o, ["lda", "rf", "svm_rbf"], dirs, max_parallel=3, cpu_budget=4)
            self.assertEqual(plan.kind, "emg")
            e = eng.TrainingEngine(plan)
            e.start()
            t0 = time.time()
            while not e.finished and time.time() - t0 < 240:
                e.poll()
                time.sleep(0.2)
            self.assertTrue(all(j.state == eng.COMPLETED for j in e.jobs.values()), {k: j.error for k, j in e.jobs.items()})
            for r in e.results():
                self.assertIn("f1_macro_smoothed", r["realtime"])
                self.assertIn("lat_p95_ms", r["realtime"])
                self.assertEqual(r["classes"], ["fist_close", "left", "rest", "right"])
            files = sorted(dirs.trained_models.glob("*.joblib"))
            self.assertEqual(len(files), 3)
            clf = load_gesture_classifier(files[0])
            window = recs[0].signal[3000:3000 + clf.window_samples]
            label, conf = clf.predict_window(window)
            self.assertIn(label, clf.classes)
            self.assertTrue(0.0 <= conf <= 1.0)
            meta = json.loads(files[0].with_suffix(".json").read_text())
            self.assertEqual(meta["artifact"]["reload_check"], "ok")
            self.assertEqual(meta["split"]["mode"], "subject")


if __name__ == "__main__":
    unittest.main()
