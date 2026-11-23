"""Dataset analysis, feature-type inference, leakage-safe splitting and the preprocessing pipeline.

Leakage rules enforced here:
* the train/test split happens **before** any statistic is learned;
* imputation, scaling, encoding, constant-feature removal and ``SelectKBest`` are fitted on the training
  rows only (``fit_preprocessing``) and merely applied to the test rows;
* every model gets exactly the same split (same seed) and the same preprocessing code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler, StandardScaler)

NUMERIC, CATEGORICAL, IGNORE = "numeric", "categorical", "ignore"


@dataclass
class PreprocessOptions:
    """Everything the GUI's *Configure* tab controls (JSON-serialisable, stored with every model)."""
    task: str = "auto"                  # auto | classification | regression
    test_size: float = 0.2
    seed: int = 42
    drop_duplicates: bool = True
    drop_constant: bool = True          # remove zero-variance features (fitted on train)
    impute_numeric: str = "median"      # median | mean | most_frequent
    impute_categorical: str = "most_frequent"
    scaling: str = "standard"           # standard | minmax | robust | none
    encoding: str = "onehot"            # onehot | ordinal
    max_categories: int = 30            # onehot: rarer levels are pooled into one "infrequent" column
    feature_selection: str = "none"     # none | kbest
    k_best: int = 20
    class_weight: bool = True           # balanced class weights where the model supports them
    dl_epochs: int = 40                 # deep-learning models only

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------ analysis
def analyze(df: pd.DataFrame) -> dict:
    """Summary shown in the *Data* tab."""
    miss = df.isna().sum()
    try:
        dups = int(df.duplicated().sum())
    except TypeError:                                    # unhashable cells (lists / dicts)
        dups = int(df.astype(str).duplicated().sum())
    columns = []
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        columns.append({"name": str(c), "dtype": str(s.dtype), "missing": int(miss[c]),
                        "missing_pct": float(100.0 * miss[c] / max(len(df), 1)),
                        "unique": int(nn.astype(str).nunique()) if len(nn) else 0,
                        "example": "" if nn.empty else str(nn.iloc[0])[:40]})
    try:
        stats = df.describe(include="all").T
    except Exception:
        stats = pd.DataFrame()
    return {"rows": int(len(df)), "cols": int(df.shape[1]), "missing_cells": int(miss.sum()),
            "rows_with_missing": int(df.isna().any(axis=1).sum()), "duplicates": dups,
            "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6), "columns": columns, "stats": stats}


def infer_column_type(s: pd.Series) -> str:
    name = str(s.name).strip().lower()
    nn = s.dropna()
    if len(nn) == 0:
        return IGNORE
    if pd.api.types.is_bool_dtype(s):
        return CATEGORICAL
    if pd.api.types.is_datetime64_any_dtype(s):
        return IGNORE
    if pd.api.types.is_numeric_dtype(s):
        if (name in ("id", "index", "unnamed: 0") or name.endswith("_id")) and nn.nunique() >= 0.95 * len(nn) \
                and pd.api.types.is_integer_dtype(s):
            return IGNORE                                # row identifier
        return NUMERIC
    conv = pd.to_numeric(nn, errors="coerce")
    if conv.notna().mean() >= 0.95:
        return NUMERIC                                   # numbers stored as text
    nun = nn.astype(str).nunique()
    if nun > 50 and nun / len(nn) > 0.5:
        return IGNORE                                    # identifier / free text
    return CATEGORICAL


def infer_types(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, str]:
    return {str(c): infer_column_type(df[c]) for c in df.columns if str(c) != target}


def detect_task(y: pd.Series) -> str:
    nn = y.dropna()
    if not pd.api.types.is_numeric_dtype(nn) or pd.api.types.is_bool_dtype(nn):
        return "classification"
    nun = nn.nunique()
    integer_like = bool(np.all(np.mod(nn.to_numpy(dtype=float), 1) == 0))
    if nun <= 10 or (integer_like and (nun <= 20 or nun / max(len(nn), 1) < 0.02)):
        return "classification"
    return "regression"


# --------------------------------------------------------------------------------------- data frame
def coerce_columns(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    """Give every feature a clean dtype: floats (inf -> NaN) / strings (missing stays NaN)."""
    out = pd.DataFrame(index=df.index)
    for c in numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    for c in categorical:
        s = df[c]
        out[c] = s.astype(str).where(s.notna(), np.nan).astype(object)
    return out[numeric + categorical]


@dataclass
class PreparedData:
    task: str
    target: str
    numeric: List[str]
    categorical: List[str]
    classes: List[str]                     # class names (classification) in label-index order
    label_encoder: Optional[LabelEncoder]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    warnings: List[str]
    n_dropped_duplicates: int = 0
    n_dropped_missing_target: int = 0


def prepare_data(df: pd.DataFrame, target: str, feature_types: Dict[str, str],
                 opts: PreprocessOptions) -> PreparedData:
    """Clean the frame, decide the task, then split. Nothing is *learned* from the data except the
    label vocabulary (class names), which carries no information about the test labels' relationship to X."""
    warnings: List[str] = []
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    numeric = [c for c, t in feature_types.items() if t == NUMERIC and c != target and c in df.columns]
    categorical = [c for c, t in feature_types.items() if t == CATEGORICAL and c != target and c in df.columns]
    if not numeric and not categorical:
        raise ValueError("No feature columns selected.")

    work = df[numeric + categorical + [target]].copy()
    n0 = len(work)
    work = work[work[target].notna()]
    dropped_missing = n0 - len(work)
    if dropped_missing:
        warnings.append(f"Dropped {dropped_missing} rows with a missing target.")
    dropped_dups = 0
    if opts.drop_duplicates:
        n1 = len(work)
        try:
            work = work.drop_duplicates()
        except TypeError:
            work = work.loc[~work.astype(str).duplicated()]
        dropped_dups = n1 - len(work)
        if dropped_dups:
            warnings.append(f"Dropped {dropped_dups} duplicate rows (before splitting, so no row is in both sets).")

    y_raw = work[target]
    task = opts.task if opts.task in ("classification", "regression") else detect_task(y_raw)
    X = coerce_columns(work, numeric, categorical)

    le: Optional[LabelEncoder] = None
    classes: List[str] = []
    if task == "classification":
        y_str = y_raw.astype(str)
        counts = y_str.value_counts()
        rare = counts[counts < 2].index
        if len(rare):
            warnings.append(f"Dropped {len(rare)} class(es) with a single sample: {list(rare)[:5]}")
            keep = ~y_str.isin(rare)
            X, y_raw, y_str = X[keep.to_numpy()], y_raw[keep], y_str[keep]
        le = LabelEncoder().fit(y_str)
        classes = [str(c) for c in le.classes_]
        if len(classes) < 2:
            raise ValueError("Classification needs at least two classes in the target column.")
        y = le.transform(y_str).astype(int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        if not ok.all():
            warnings.append(f"Dropped {int((~ok).sum())} rows with a non-numeric target.")
            X, y = X[ok], y[ok]

    if len(X) < 10:
        raise ValueError(f"Only {len(X)} usable rows - too few to train and test.")
    stratify = None
    if task == "classification":
        n_test = int(np.ceil(len(y) * opts.test_size))
        if np.bincount(y).min() >= 2 and n_test >= len(classes):
            stratify = y
        else:
            warnings.append("Class sizes too small for a stratified split; used a plain random split.")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=opts.test_size, random_state=opts.seed,
                                              stratify=stratify)
    return PreparedData(task, target, numeric, categorical, classes, le, X_tr.reset_index(drop=True),
                        X_te.reset_index(drop=True), np.asarray(y_tr), np.asarray(y_te), warnings,
                        dropped_dups, dropped_missing)


# ------------------------------------------------------------------------------------- preprocessing
def build_column_transformer(opts: PreprocessOptions, numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    parts = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy=opts.impute_numeric, keep_empty_features=True))]
        scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}.get(opts.scaling)
        if scaler:
            steps.append(("scale", scaler()))
        parts.append(("num", Pipeline(steps), numeric))
    if categorical:
        steps = [("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]
        if opts.encoding == "ordinal":
            steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        else:
            steps.append(("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False,
                                                  max_categories=max(2, opts.max_categories))))
        parts.append(("cat", Pipeline(steps), categorical))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def fit_preprocessing(opts: PreprocessOptions, numeric: List[str], categorical: List[str],
                      X_train: pd.DataFrame, y_train: np.ndarray, task: str) -> Tuple[Pipeline, np.ndarray]:
    """Fit all preprocessing on the TRAIN rows and return (fitted pipeline, transformed train matrix)."""
    ct = build_column_transformer(opts, numeric, categorical)
    Xt = ct.fit_transform(X_train, y_train)
    steps = [("prep", ct)]
    if opts.drop_constant:
        vt = VarianceThreshold(0.0).fit(Xt)
        if vt.get_support().any():
            Xt = vt.transform(Xt)
            steps.append(("variance", vt))
    if opts.feature_selection == "kbest" and Xt.shape[1] > 1:
        k = int(max(1, min(opts.k_best, Xt.shape[1])))
        sel = SelectKBest(f_classif if task == "classification" else f_regression, k=k).fit(Xt, y_train)
        Xt = sel.transform(Xt)
        steps.append(("select", sel))
    return Pipeline(steps), np.asarray(Xt, dtype=np.float64)


def feature_names(pre: Pipeline) -> List[str]:
    """Names of the columns the model sees (best effort)."""
    try:
        names = np.asarray(pre.named_steps["prep"].get_feature_names_out(), dtype=object)
        for key in ("variance", "select"):
            if key in pre.named_steps:
                names = names[pre.named_steps[key].get_support()]
        return [str(n) for n in names]
    except Exception:
        return []
