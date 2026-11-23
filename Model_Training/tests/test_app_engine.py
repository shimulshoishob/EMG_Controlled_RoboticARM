"""Headless tests of the GUI application's engine (no Qt needed).

Run (from the project root):  python -m unittest Model_Training.tests.test_app_engine -v
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Model_Training.app import engine as eng
from Model_Training.app.inference import load_bundle, predict
from Model_Training.app.paths import Dirs
from Model_Training.app.preprocess import PreprocessOptions, analyze, detect_task, infer_types, prepare_data
from Model_Training.app.reports import best_model, comparison_table, write_reports


def make_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["x", "y", "z"], n)
    label = np.where(a + 0.5 * b + (cat == "x") > 0.3, "pos", "neg")
    df = pd.DataFrame({"id": np.arange(n), "a": a, "b": b, "cat": cat, "const": 1.0,
                       "reg_target": 3 * a - 2 * b + rng.normal(0, 0.1, n), "label": label})
    df.loc[rng.choice(n, 20, replace=False), "a"] = np.nan        # missing values
    df = pd.concat([df, df.iloc[:10]])                              # duplicates
    return df


def run_to_end(plan, timeout=180):
    e = eng.TrainingEngine(plan)
    e.start()
    t0 = time.time()
    while not e.finished and time.time() - t0 < timeout:
        e.poll()
        time.sleep(0.1)
    e.poll()
    return e


class PreprocessTests(unittest.TestCase):
    def test_analysis_and_types(self):
        df = make_frame()
        info = analyze(df)
        self.assertEqual(info["rows"], len(df))
        self.assertGreaterEqual(info["duplicates"], 10)
        self.assertGreater(info["missing_cells"], 0)
        t = infer_types(df, "label")
        self.assertEqual(t["a"], "numeric")
        self.assertEqual(t["cat"], "categorical")
        self.assertEqual(t["id"], "ignore")
        self.assertEqual(detect_task(df["label"]), "classification")
        self.assertEqual(detect_task(df["reg_target"]), "regression")

    def test_split_is_disjoint_and_duplicates_removed(self):
        df = make_frame()
        t = {k: v for k, v in infer_types(df, "label").items() if k not in ("id", "reg_target")}
        p = prepare_data(df, "label", t, PreprocessOptions())
        self.assertEqual(p.task, "classification")
        self.assertEqual(p.n_dropped_duplicates, 10)
        self.assertEqual(len(p.X_train) + len(p.X_test), len(df) - 10)
        self.assertEqual(set(p.y_train), set(p.y_test))                # stratified


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dirs = Dirs(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _plan(self, target, keys, **kw):
        df = make_frame()
        drop = {"id", "label", "reg_target"} - {target}
        types = {k: v for k, v in infer_types(df, target).items() if k not in drop}
        return eng.build_run(df, target, types, PreprocessOptions(**kw), keys, self.dirs, max_parallel=3,
                             dataset_name="synthetic.csv", cpu_budget=4)

    def test_classification_parallel_and_export(self):
        e = run_to_end(self._plan("label", ["rf", "extra_trees", "svm_rbf", "logreg", "lda"]))
        self.assertTrue(all(j.state == eng.COMPLETED for j in e.jobs.values()), {k: j.error for k, j in e.jobs.items()})
        files = sorted((self.dirs.trained_models).glob("*.joblib"))
        self.assertEqual(len(files), 5)
        res = e.results()
        self.assertTrue(all(0.6 < r["metrics"]["accuracy"] <= 1 for r in res))
        self.assertTrue(all("confusion_matrix" in r for r in res))
        # reload and predict on raw rows
        b = load_bundle(files[0])
        out = predict(b, make_frame().head(5))
        self.assertTrue(set(out) <= {"pos", "neg"})
        paths = write_reports(e)
        self.assertTrue((Path(paths["reports"]) / "model_comparison.csv").exists())
        self.assertTrue(list(Path(paths["plots"]).glob("confusion_*.png")))
        self.assertIsNotNone(best_model(res, "classification"))

    def test_regression(self):
        e = run_to_end(self._plan("reg_target", ["rf", "xgboost", "svm_rbf", "logreg", "mlp"]))
        self.assertTrue(all(j.state == eng.COMPLETED for j in e.jobs.values()), {k: j.error for k, j in e.jobs.items()})
        self.assertTrue(all(r["metrics"]["r2"] > 0.8 for r in e.results()))
        b = load_bundle(e.results()[0]["artifact"])
        self.assertEqual(len(predict(b, make_frame().head(7))), 7)

    def test_failure_is_isolated(self):
        plan = self._plan("label", ["rf", "svm_rbf"])
        e = eng.TrainingEngine(plan)
        # sabotage one model by giving it an invalid parameter through the payload
        orig = e._payload
        e._payload = lambda job, n: {**orig(job, n), "params": {"C": -1}} if job.key == "svm_rbf" else orig(job, n)
        e.start()
        t0 = time.time()
        while not e.finished and time.time() - t0 < 120:
            e.poll()
            time.sleep(0.1)
        self.assertEqual(e.jobs["svm_rbf"].state, eng.FAILED)
        self.assertEqual(e.jobs["rf"].state, eng.COMPLETED)

    def test_cancel(self):
        e = eng.TrainingEngine(self._plan("label", ["rf", "extra_trees", "gradient_boosting", "knn"]))
        e.start()
        e.cancel_all()
        self.assertTrue(all(j.state in (eng.CANCELLED, eng.COMPLETED) for j in e.jobs.values()))
        self.assertTrue(any(j.state == eng.CANCELLED for j in e.jobs.values()))
        self.assertTrue(all(j.process is None or not j.process.is_alive() for j in e.jobs.values()))


if __name__ == "__main__":
    unittest.main()
