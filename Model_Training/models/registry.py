"""Model registry: one entry per architecture (no duplicates under different names)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..utils import have, import_error
from . import ml_models as ml
from .base import BaseModel, ModelContext


@dataclass
class ModelSpec:
    key: str
    label: str
    family: str                      # ml | dl | hybrid
    input_kind: str                  # features | raw
    requires: Tuple[str, ...]        # importable modules
    factory: Callable                # (ctx, params) -> BaseModel
    description: str = ""
    space: Optional[Dict[str, list]] = None      # random-search space (classical models)
    pip: str = ""


def _torch_factory(arch: str, family: str):
    def make(ctx: ModelContext, params: dict) -> BaseModel:
        from .torch_wrapper import TorchModel
        return TorchModel(arch, family, arch, ctx.n_channels, ctx.n_samples, ctx.n_classes,
                          ctx.train_cfg, params, ctx.device, ctx.seed)
    return make


def _cnn_rf(ctx: ModelContext, params: dict) -> BaseModel:
    from sklearn.ensemble import RandomForestClassifier
    from .torch_wrapper import HybridCNNRF, TorchModel
    params = dict(params)
    rf_kw = dict(n_estimators=200, min_samples_leaf=1, max_features="sqrt",
                 class_weight="balanced_subsample" if ctx.train_cfg.class_weight else None,
                 n_jobs=ctx.train_cfg.n_jobs, random_state=ctx.seed)
    rf_kw.update({k[3:]: params.pop(k) for k in list(params) if k.startswith("rf_")})
    cnn = TorchModel("cnn_rf", "hybrid", "cnn1d", ctx.n_channels, ctx.n_samples, ctx.n_classes,
                     ctx.train_cfg, params, ctx.device, ctx.seed)
    return HybridCNNRF("cnn_rf", "hybrid", cnn, RandomForestClassifier(**rf_kw), ctx.n_classes, ctx.train_cfg.n_jobs)


_TREE_SPACE = {"n_estimators": [100, 200, 400], "max_depth": [None, 12, 24], "min_samples_leaf": [1, 2, 4],
               "max_features": ["sqrt", "log2", 0.3]}

SPECS: Dict[str, ModelSpec] = {s.key: s for s in [
    # ---- A. classical machine learning (engineered features)
    ModelSpec("rf", "Random Forest", "ml", "features", ("sklearn",), ml.build_rf, space=_TREE_SPACE),
    ModelSpec("extra_trees", "Extra Trees", "ml", "features", ("sklearn",), ml.build_extra_trees, space=_TREE_SPACE),
    ModelSpec("xgboost", "XGBoost", "ml", "features", ("xgboost",), ml.build_xgboost, pip="xgboost",
              space={"max_depth": [3, 5, 7], "learning_rate": [0.04, 0.08, 0.15], "subsample": [0.7, 0.9],
                     "colsample_bytree": [0.6, 0.8, 1.0]}),
    ModelSpec("lightgbm", "LightGBM", "ml", "features", ("lightgbm",), ml.build_lightgbm, pip="lightgbm",
              space={"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.06, 0.12], "colsample_bytree": [0.6, 0.8, 1.0],
                     "min_child_samples": [10, 20, 40]}),
    ModelSpec("catboost", "CatBoost", "ml", "features", ("catboost",), ml.build_catboost, pip="catboost",
              space={"depth": [4, 6, 8], "learning_rate": [0.05, 0.1, 0.2], "l2_leaf_reg": [1, 3, 9]}),
    ModelSpec("svm_rbf", "SVM (RBF)", "ml", "features", ("sklearn",), ml.build_svm_rbf,
              space={"C": [1, 3, 10, 30, 100], "gamma": ["scale", 0.003, 0.01, 0.03]}),
    ModelSpec("svm_linear", "SVM (linear)", "ml", "features", ("sklearn",), ml.build_svm_linear,
              space={"C": [0.03, 0.1, 0.3, 1, 3, 10]}),
    ModelSpec("knn", "k-NN", "ml", "features", ("sklearn",), ml.build_knn,
              space={"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]}),
    ModelSpec("logreg", "Logistic Regression", "ml", "features", ("sklearn",), ml.build_logreg,
              space={"C": [0.01, 0.1, 1, 10, 100]}),
    ModelSpec("lda", "LDA", "ml", "features", ("sklearn",), ml.build_lda,
              space={"shrinkage": ["auto", 0.1, 0.5]}),
    ModelSpec("gradient_boosting", "Gradient Boosting", "ml", "features", ("sklearn",), ml.build_gradient_boosting,
              space={"n_estimators": [60, 100, 200], "max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1, 0.2]}),
    ModelSpec("mlp", "MLP", "ml", "features", ("sklearn",), ml.build_mlp,
              space={"hidden_layer_sizes": [(64,), (128, 64), (256, 128)], "alpha": [1e-4, 1e-3, 1e-2]}),
    # ---- B. deep learning (raw windows)
    ModelSpec("cnn1d", "1D CNN", "dl", "raw", ("torch",), _torch_factory("cnn1d", "dl"), pip="torch"),
    ModelSpec("cnn2d", "2D CNN", "dl", "raw", ("torch",), _torch_factory("cnn2d", "dl"), pip="torch"),
    ModelSpec("lstm", "LSTM", "dl", "raw", ("torch",), _torch_factory("lstm", "dl"), pip="torch"),
    ModelSpec("bilstm", "BiLSTM", "dl", "raw", ("torch",), _torch_factory("bilstm", "dl"), pip="torch"),
    ModelSpec("gru", "GRU", "dl", "raw", ("torch",), _torch_factory("gru", "dl"), pip="torch"),
    ModelSpec("transformer", "Transformer", "dl", "raw", ("torch",), _torch_factory("transformer", "dl"), pip="torch"),
    # ---- C. hybrids
    ModelSpec("cnn_lstm", "CNN-LSTM", "hybrid", "raw", ("torch",), _torch_factory("cnn_lstm", "hybrid"), pip="torch"),
    ModelSpec("cnn_gru", "CNN-GRU", "hybrid", "raw", ("torch",), _torch_factory("cnn_gru", "hybrid"), pip="torch"),
    ModelSpec("cnn_bilstm", "CNN-BiLSTM", "hybrid", "raw", ("torch",), _torch_factory("cnn_bilstm", "hybrid"), pip="torch"),
    ModelSpec("cnn_tcn", "CNN-TCN", "hybrid", "raw", ("torch",), _torch_factory("cnn_tcn", "hybrid"), pip="torch"),
    ModelSpec("cnn_attention", "CNN-Attention", "hybrid", "raw", ("torch",), _torch_factory("cnn_attention", "hybrid"), pip="torch"),
    ModelSpec("cnn_transformer", "CNN-Transformer", "hybrid", "raw", ("torch",), _torch_factory("cnn_transformer", "hybrid"), pip="torch"),
    ModelSpec("cnn_rf", "CNN + Random Forest", "hybrid", "raw", ("torch", "sklearn"), _cnn_rf, pip="torch",
              description="CNN embeddings -> Random Forest classifier"),
]}


def select_models(enabled, families: List[str], exclude: List[str]) -> Tuple[List[ModelSpec], List[Tuple[str, str]]]:
    """Returns (runnable specs, [(key, reason it was skipped)])."""
    if enabled == "all" or enabled is None:
        keys = list(SPECS)
    else:
        keys = list(enabled)
        unknown = [k for k in keys if k not in SPECS]
        if unknown:
            raise KeyError(f"Unknown model key(s) {unknown}. Available: {list(SPECS)}")
    run, skipped = [], []
    for k in keys:
        spec = SPECS[k]
        if spec.family not in families:
            skipped.append((k, f"family '{spec.family}' not enabled"))
        elif k in exclude:
            skipped.append((k, "excluded in config"))
        else:
            missing = [m for m in spec.requires if not have(m)]
            if missing:
                hint = f" (pip install {spec.pip})" if spec.pip else ""
                why = "; ".join(f"{m}: {import_error(m)}" for m in missing)
                skipped.append((k, f"required library unusable - {why}{hint}"))
            else:
                run.append(spec)
    return run, skipped
