"""Fast checks of the properties that make the benchmark trustworthy.

Run:  python -m unittest discover Model_Training/tests     (from the project root)
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Model_Training.config import DataConfig, ExperimentConfig, SplitConfig
from Model_Training.data_io import Recording, load_dataset, parse_filename
from Model_Training.features import FeatureExtractor
from Model_Training.pipeline import prepare
from Model_Training.preprocessing import Segment, WindowPreprocessor, build_windows
from Model_Training.splits import make_folds, split_pieces, verify_no_leakage


def fake_recs(n_sub=4, n_sess=2, gestures=("a", "b", "c"), n=1500, ch=4, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for s in range(n_sub):
        for k in range(n_sess):
            for gi, g in enumerate(gestures):
                sig = rng.normal(0, 1 + gi, (n, ch)).astype(np.float32) * 50 + 2000
                recs.append(Recording(f"S{s}_{k}_{g}", "x", sig, 500.0, f"S{s}", f"sess{k}", [(0, n, g)], [f"emg{i}" for i in range(ch)]))
    return recs


class SplitTests(unittest.TestCase):
    def test_subject_split_has_disjoint_subjects(self):
        recs = fake_recs()
        pieces, info = split_pieces(recs, SplitConfig(mode="subject"), 100)
        subs = {k: {recs[p.seg.rec_idx].subject for p in v} for k, v in pieces.items()}
        self.assertEqual(info["mode"], "subject")
        self.assertFalse(subs["train"] & subs["test"] or subs["train"] & subs["val"] or subs["val"] & subs["test"])

    def test_temporal_split_never_shares_samples(self):
        recs = fake_recs(n_sub=1, n_sess=1)
        pieces, info = split_pieces(recs, SplitConfig(mode="temporal"), 100)
        verify_no_leakage(recs, pieces, "temporal")                 # raises on any overlap
        self.assertTrue(all(len(v) for v in pieces.values()))

    def test_auto_mode_choice(self):
        self.assertEqual(split_pieces(fake_recs(n_sub=4), SplitConfig(), 100)[1]["mode"], "subject")
        self.assertEqual(split_pieces(fake_recs(n_sub=1, n_sess=1), SplitConfig(), 100)[1]["mode"], "temporal")
        self.assertEqual(split_pieces(fake_recs(n_sub=1, n_sess=4), SplitConfig(), 100)[1]["mode"], "session")

    def test_windows_never_cross_segments(self):
        recs = fake_recs(n_sub=1, n_sess=1)
        pieces, _ = split_pieces(recs, SplitConfig(mode="temporal"), 100)
        for plist in pieces.values():
            segs = [p.seg for p in plist]
            ws = build_windows(recs, segs, 100, 25)
            for si, w in zip(ws.seg, ws.X):
                s = segs[si]
                sig = recs[s.rec_idx].signal[s.start:s.stop]
                # window must be an exact contiguous slice of its own segment
                found = any(np.array_equal(sig[i:i + 100], w) for i in range(0, len(sig) - 99, 25))
                self.assertTrue(found)

    def test_group_cv_folds_keep_groups_apart(self):
        recs = fake_recs()
        pieces, _ = split_pieces(recs, SplitConfig(mode="subject"), 100)
        for tr, va in make_folds(pieces["train"] + pieces["val"], "subject", 3, 0):
            self.assertFalse({p.group for p in tr} & {p.group for p in va})


class PreprocessTests(unittest.TestCase):
    def test_normalisers_are_fit_on_train_only(self):
        recs = fake_recs()
        cfg = ExperimentConfig()
        cfg.preprocess.window_ms = 200
        pieces, _ = split_pieces(recs, cfg.split, cfg.window_samples)
        prep = prepare(recs, pieces, ["a", "b", "c"], cfg)
        tr = prep.parts["train"]
        # training features are standardised exactly; held-out ones are not forced to mean 0
        self.assertTrue(np.allclose(tr.feat.mean(0), 0, atol=1e-3))
        self.assertTrue(np.allclose(tr.raw.mean((0, 2)), 0, atol=1e-3))

    def test_feature_dimension_matches_names(self):
        cfg = ExperimentConfig()
        ex = FeatureExtractor(500, 8, 100, cfg.features)
        f = ex.transform(np.random.randn(5, 100, 8).astype(np.float32))
        self.assertEqual(f.shape, (5, len(ex.names)))
        self.assertTrue(np.isfinite(f).all())

    def test_filter_removes_dc(self):
        pre = WindowPreprocessor(500, ExperimentConfig().preprocess)
        out = pre(np.full((2, 100, 3), 2048.0, np.float32) + np.random.randn(2, 100, 3).astype(np.float32))
        self.assertLess(abs(float(out.mean())), 1.0)


class DataTests(unittest.TestCase):
    def test_filename_parsing(self):
        m = parse_filename("emg_20261108_185041_S02_sess1_closed_fist_trial3", DataConfig())
        self.assertEqual((m["subject"], m["session"], m["gesture"]), ("S02", "sess1", "closed_fist"))
        self.assertIsNone(parse_filename("emg_20261108_185041_s011", DataConfig())["gesture"])

    def test_bad_files_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            hdr = "time_s,emg1,emg2\n"
            good = hdr + "\n".join(f"{i / 500:.3f},{2000 + (i % 7) * 5},{2100 + (i % 5) * 9}" for i in range(1500))
            (d / "S01_fist_trial1.csv").write_text(good)
            (d / "S01_open_trial1.csv").write_text(hdr + "\n".join(f"{i / 500:.3f},nan,nan" for i in range(1500)))   # all missing
            (d / "S01_rest_trial1.csv").write_text(hdr + "0,1,1\n")                                               # too short
            (d / "S01_trial2.csv").write_text(good)                                                              # no gesture
            cfg = DataConfig(paths=[str(d)])
            recs, rep = load_dataset(cfg)
            self.assertEqual(len(recs), 1)
            self.assertEqual(len(rep["skipped"]), 3)


if __name__ == "__main__":
    unittest.main()
