from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.base import clone
import joblib

from .features import build_preprocessor

@dataclass
class CVResult:
    oof_proba: np.ndarray
    auc_mean: float
    auc_std: float
    ap_mean: float
    ap_std: float
    best_params: dict

def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(min_samples_leaf=20, class_weight="balanced")

def fit_time_aware(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> tuple[Pipeline, CVResult]:
    pre = build_preprocessor(X)
    model = build_model()
    pipe = Pipeline([("pre", pre), ("model", model)])

    grid = {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [None, 6, 12],
    }
    if "TransactionDT" in X.columns:
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = cv.split(X, y)
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
        splits = cv.split(X, y)

    search = GridSearchCV(pipe, grid, scoring="average_precision", cv=splits, n_jobs=-1)
    search.fit(X, y)
    best = search.best_estimator_

    oof = np.zeros(len(y), dtype=float)
    fold_aucs, fold_aps = [], []
    for tr, te in cv.split(X, y):
        m = clone(best).fit(X.iloc[tr], y.iloc[tr])
        proba = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = proba
        fold_aucs.append(roc_auc_score(y.iloc[te], proba))
        fold_aps.append(average_precision_score(y.iloc[te], proba))

    res = CVResult(
        oof_proba=oof,
        auc_mean=float(np.mean(fold_aucs)),
        auc_std=float(np.std(fold_aucs)),
        ap_mean=float(np.mean(fold_aps)),
        ap_std=float(np.std(fold_aps)),
        best_params=search.best_params_,
    )
    return best, res

def save_model(pipe: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)

def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
