"""Regression counterparts of the classical models in ``models/registry.py``.

The existing registry is classification-only (it was built for gesture recognition), so for regression
tasks the same *families and default hyper-parameters* are instantiated as their regressor classes.
Deep / hybrid networks are cross-entropy classifiers and are not offered for regression.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from ..utils import have

# key -> (label, required module, factory(ctx, params) -> (estimator, booster))
REGRESSION_LABELS = {
    "rf": "Random Forest", "extra_trees": "Extra Trees", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "catboost": "CatBoost", "svm_rbf": "SVR (RBF)", "svm_linear": "SVR (linear)", "knn": "k-NN",
    "logreg": "Ridge Regression", "gradient_boosting": "Gradient Boosting", "mlp": "MLP",
}
REGRESSION_REQUIRES = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}
NOT_FOR_REGRESSION = {"lda": "LDA is a classifier"}


def _scaled_target(est):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def build_regressor(key: str, ctx, params: Optional[dict] = None) -> Tuple[object, Optional[str]]:
    """Returns (unfitted estimator, booster tag for early stopping)."""
    p = dict(params or {})
    nj, seed = ctx.train_cfg.n_jobs, ctx.seed
    if key == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**{**dict(n_estimators=200, n_jobs=nj, random_state=seed), **p}), None
    if key == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**{**dict(n_estimators=200, n_jobs=nj, random_state=seed), **p}), None
    if key == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**{**dict(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, tree_method="hist", n_jobs=nj, random_state=seed,
                                      verbosity=0, device="cuda" if ctx.device == "cuda" else "cpu"), **p}), "xgb"
    if key == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**{**dict(n_estimators=300, learning_rate=0.06, num_leaves=31, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, n_jobs=nj, random_state=seed,
                                       verbose=-1), **p}), "lgbm"
    if key == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**{**dict(iterations=300, depth=6, learning_rate=0.1, loss_function="RMSE",
                                           thread_count=nj, random_seed=seed, allow_writing_files=False,
                                           verbose=0), **p}), "cat"
    if key == "svm_rbf":
        from sklearn.svm import SVR
        return _scaled_target(SVR(**{**dict(kernel="rbf", C=10.0, gamma="scale", cache_size=500), **p})), None
    if key == "svm_linear":
        from sklearn.svm import SVR, LinearSVR
        if ctx.n_train <= 10000:
            return _scaled_target(SVR(**{**dict(kernel="linear", C=1.0, cache_size=500), **p})), None
        return _scaled_target(LinearSVR(**{**dict(C=1.0, random_state=seed, max_iter=5000, dual="auto"), **p})), None
    if key == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**{**dict(n_neighbors=7, weights="distance", n_jobs=nj), **p}), None
    if key == "logreg":
        from sklearn.linear_model import Ridge
        return Ridge(**{**dict(alpha=1.0, random_state=seed), **p}), None
    if key == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**{**dict(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8,
                                                   max_features="sqrt", random_state=seed), **p}), None
    if key == "mlp":
        from sklearn.neural_network import MLPRegressor
        return _scaled_target(MLPRegressor(**{**dict(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300,
                                                     early_stopping=True, n_iter_no_change=15,
                                                     random_state=seed), **p})), None
    raise KeyError(f"No regression variant for model '{key}'")


def fit_regressor(est, booster: Optional[str], X, y, val: Optional[Tuple]) -> None:
    """Fit with early stopping on ``val`` for the boosters (same convention as ``SklearnModel.fit``)."""
    if booster == "xgb":
        if val is not None:
            est.set_params(early_stopping_rounds=20)
            est.fit(X, y, eval_set=[val], verbose=False)
        else:
            est.set_params(early_stopping_rounds=None)
            est.fit(X, y, verbose=False)
    elif booster == "lgbm":
        import lightgbm as lgb
        if val is not None:
            est.fit(X, y, eval_set=[val], callbacks=[lgb.early_stopping(20, verbose=False)])
        else:
            est.fit(X, y)
    elif booster == "cat":
        if val is not None:
            est.fit(X, y, eval_set=val, early_stopping_rounds=20, verbose=False)
        else:
            est.fit(X, y, verbose=False)
    else:
        est.fit(X, y)


def regression_metrics(y_true, y_pred) -> dict:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse = float(mean_squared_error(y_true, y_pred))
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)), "n_samples": int(len(y_true))}
