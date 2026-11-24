"""Loading a Model_Training gesture bundle: decision rate, sidecar info, friendly errors. No hardware.

    python -m unittest tests.test_model_loading -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib  # noqa: E402

from emg_arm.ml.inference import MAX_STRIDE_SAMPLES, ModelError, load_model  # noqa: E402


class DummyClassifier:
    def __init__(self, stride):
        self.classes = ["rest", "left"]
        self.n_channels, self.window_samples, self.stride_samples = 8, 100, stride


def save(folder, stride, sidecar=None):
    path = Path(folder) / "m.joblib"
    joblib.dump({"format": "emg_gesture_classifier", "classifier": DummyClassifier(stride), "metadata": {}}, path)
    if sidecar is not None:
        path.with_suffix(".json").write_text(json.dumps(sidecar))
    return path


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_slow_training_stride_is_decided_faster(self):
        loaded = load_model(save(self.dir, stride=50))
        self.assertEqual(loaded.stride_samples, MAX_STRIDE_SAMPLES)
        self.assertEqual((loaded.input_channels, loaded.window_samples), (8, 100))

    def test_faster_stride_is_kept(self):
        self.assertEqual(load_model(save(self.dir, stride=10)).stride_samples, 10)

    def test_sidecar_gives_training_windows_and_accuracy(self):
        side = {"dataset": {"class_windows": {"train": {"rest": 289, "left": 40}}}, "metrics": {"accuracy": 0.87}}
        loaded = load_model(save(self.dir, 50, side))
        self.assertEqual(loaded.train_windows, {"rest": 289, "left": 40})
        self.assertAlmostEqual(loaded.test_accuracy, 0.87)

    def test_missing_or_broken_sidecar_is_ignored(self):
        self.assertIsNone(load_model(save(self.dir, 50)).train_windows)
        path = save(self.dir, 50)
        path.with_suffix(".json").write_text("not json")
        self.assertIsNone(load_model(path).test_accuracy)

    def test_table_model_is_refused_with_a_clear_message(self):
        path = Path(self.dir) / "t.joblib"
        joblib.dump({"pipeline": object()}, path)
        with self.assertRaises(ModelError) as ctx:
            load_model(path)
        self.assertIn("cannot drive the arm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
