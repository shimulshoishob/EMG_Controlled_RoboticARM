"""Classical machine-learning models (scikit-learn / XGBoost / LightGBM / CatBoost) behind ``BaseModel``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .base import BaseModel, ModelContext


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SklearnModel(BaseModel):
    """Wraps any estimator with the scikit-learn API.

    ``booster`` ("xgb" | "lgbm" | "cat" | None) enables early stopping on the validation set.
    Models without predict_proba (SVMs) expose softmax(decision_function) as pseudo-probabilities.
    """

    kind_name = "sklearn"
    input_kind = "features"

    def __init__(self, key, family, estimator, n_classes, booster=None, sample_weight=False):
        super().__init__()
        self.key, self.family = key, family
        self.est = estimator
        self.n_classes = int(n_classes)
        self.booster = booster
        self.sample_weight = sample_weight
        self._pp_kw = {}

    def fit(self, train, val=None):
        X, y = train["X"], train["y"]
        kw = {}
        if self.sample_weight:
            from sklearn.utils.class_weight import compute_sample_weight
            kw["sample_weight"] = compute_sample_weight("balanced", y)
        est = self.est
        if self.booster == "xgb":
            if val is not None:
                est.set_params(early_stopping_rounds=20)
                est.fit(X, y, eval_set=[(val["X"], val["y"])], verbose=False, **kw)
            else:
                est.set_params(early_stopping_rounds=None)
                est.fit(X, y, verbose=False, **kw)
        elif self.booster == "lgbm":
            import lightgbm as lgb
            if val is not None:
                est.fit(X, y, eval_set=[(val["X"], val["y"])],
                        callbacks=[lgb.early_stopping(20, verbose=False)], **kw)
            else:
                est.fit(X, y, **kw)
        elif self.booster == "cat":
            if val is not None:
                est.fit(X, y, eval_set=(val["X"], val["y"]), early_stopping_rounds=20, verbose=False)
            else:
                est.fit(X, y, verbose=False)
        else:
            est.fit(X, y, **kw)
        return self

    def predict_proba(self, X):
        est = self.est
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X, **self._pp_kw)
        else:
            d = est.decision_function(X)
            if d.ndim == 1:
                d = np.stack([-d, d], axis=1)
            p = _softmax(d)
        classes = np.asarray(getattr(est, "classes_", np.arange(p.shape[1]))).astype(int)
        if p.shape[1] == self.n_classes and np.array_equal(classes, np.arange(self.n_classes)):
            return p
        full = np.zeros((len(p), self.n_classes), dtype=p.dtype)
        full[:, classes] = p
        return full

    def set_inference_threads(self, n: int) -> None:
        if self.booster == "cat":
            self._pp_kw = {"thread_count": int(n)}
            return
        params = self.est.get_params() if hasattr(self.est, "get_params") else {}
        if "n_jobs" in params:
            self.est.set_params(n_jobs=int(n))
        if self.booster == "xgb" and "device" in params:
            self.est.set_params(device="cpu")

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.joblib"
        joblib.dump({"est": self.est, "n_classes": self.n_classes, "booster": self.booster,
                     "key": self.key, "family": self.family}, path, compress=3)
        return path

    @classmethod
    def load(cls, directory: Path):
        d = joblib.load(Path(directory) / "model.joblib")
        return cls(d["key"], d["family"], d["est"], d["n_classes"], booster=d["booster"])


# ---------------------------------------------------------------------------------------------
# factories: (ctx, params) -> SklearnModel.  ``params`` override the defaults (tuning / config).
# ---------------------------------------------------------------------------------------------
def _cw(ctx: ModelContext):
    return "balanced" if ctx.train_cfg.class_weight else None


def build_rf(ctx, params):
    from sklearn.ensemble import RandomForestClassifier
    kw = dict(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features="sqrt",
              class_weight="balanced_subsample" if ctx.train_cfg.class_weight else None,
              n_jobs=ctx.train_cfg.n_jobs, random_state=ctx.seed)
    kw.update(params)
    return SklearnModel("rf", "ml", RandomForestClassifier(**kw), ctx.n_classes)


def build_extra_trees(ctx, params):
    from sklearn.ensemble import ExtraTreesClassifier
    kw = dict(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features="sqrt",
              class_weight="balanced_subsample" if ctx.train_cfg.class_weight else None,
              n_jobs=ctx.train_cfg.n_jobs, random_state=ctx.seed)
    kw.update(params)
    return SklearnModel("extra_trees", "ml", ExtraTreesClassifier(**kw), ctx.n_classes)


def build_xgboost(ctx, params):
    from xgboost import XGBClassifier
    kw = dict(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", n_jobs=ctx.train_cfg.n_jobs, random_state=ctx.seed,
              eval_metric="mlogloss", verbosity=0,
              device="cuda" if ctx.device == "cuda" else "cpu")
    kw.update(params)
    return SklearnModel("xgboost", "ml", XGBClassifier(**kw), ctx.n_classes, booster="xgb",
                        sample_weight=ctx.train_cfg.class_weight)


def build_lightgbm(ctx, params):
    from lightgbm import LGBMClassifier
    kw = dict(n_estimators=300, learning_rate=0.06, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, class_weight=_cw(ctx), n_jobs=ctx.train_cfg.n_jobs,
              random_state=ctx.seed, verbose=-1)
    kw.update(params)
    return SklearnModel("lightgbm", "ml", LGBMClassifier(**kw), ctx.n_classes, booster="lgbm")


def build_catboost(ctx, params):
    from catboost import CatBoostClassifier
    kw = dict(iterations=300, depth=6, learning_rate=0.1, loss_function="MultiClass",
              auto_class_weights="Balanced" if ctx.train_cfg.class_weight else None,
              thread_count=ctx.train_cfg.n_jobs, random_seed=ctx.seed, allow_writing_files=False, verbose=0)
    kw.update(params)
    return SklearnModel("catboost", "ml", CatBoostClassifier(**kw), ctx.n_classes, booster="cat")


def build_svm_rbf(ctx, params):
    from sklearn.svm import SVC
    kw = dict(kernel="rbf", C=10.0, gamma="scale", class_weight=_cw(ctx), cache_size=500, random_state=ctx.seed)
    kw.update(params)
    return SklearnModel("svm_rbf", "ml", SVC(**kw), ctx.n_classes)


def build_svm_linear(ctx, params):
    # Exact kernel SVM is O(n^2); beyond ~10k windows use the equivalent linear solver (liblinear).
    from sklearn.svm import SVC, LinearSVC
    if ctx.n_train <= 10000:
        kw = dict(kernel="linear", C=1.0, class_weight=_cw(ctx), cache_size=500, random_state=ctx.seed)
        kw.update(params)
        est = SVC(**kw)
    else:
        kw = dict(C=1.0, class_weight=_cw(ctx), random_state=ctx.seed, max_iter=5000, dual="auto")
        kw.update(params)
        est = LinearSVC(**kw)
    return SklearnModel("svm_linear", "ml", est, ctx.n_classes)


def build_knn(ctx, params):
    from sklearn.neighbors import KNeighborsClassifier
    kw = dict(n_neighbors=7, weights="distance", n_jobs=ctx.train_cfg.n_jobs)
    kw.update(params)
    return SklearnModel("knn", "ml", KNeighborsClassifier(**kw), ctx.n_classes)


def build_logreg(ctx, params):
    from sklearn.linear_model import LogisticRegression
    kw = dict(C=1.0, max_iter=1000, class_weight=_cw(ctx), random_state=ctx.seed)
    kw.update(params)
    return SklearnModel("logreg", "ml", LogisticRegression(**kw), ctx.n_classes)


def build_lda(ctx, params):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    kw = dict(solver="lsqr", shrinkage="auto")
    kw.update(params)
    return SklearnModel("lda", "ml", LinearDiscriminantAnalysis(**kw), ctx.n_classes)


def build_gradient_boosting(ctx, params):
    from sklearn.ensemble import GradientBoostingClassifier
    kw = dict(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, max_features="sqrt",
              random_state=ctx.seed)
    kw.update(params)
    return SklearnModel("gradient_boosting", "ml", GradientBoostingClassifier(**kw), ctx.n_classes,
                        sample_weight=ctx.train_cfg.class_weight)


def build_mlp(ctx, params):
    from sklearn.neural_network import MLPClassifier
    kw = dict(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=300,
              early_stopping=True, n_iter_no_change=15, random_state=ctx.seed)
    kw.update(params)
    return SklearnModel("mlp", "ml", MLPClassifier(**kw), ctx.n_classes)
