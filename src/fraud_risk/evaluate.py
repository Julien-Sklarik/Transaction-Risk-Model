from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

@dataclass
class ThresholdChoice:
    threshold: float
    f1: float
    precision: float
    recall: float

def choose_threshold_from_pr(y_true: np.ndarray, proba: np.ndarray) -> ThresholdChoice:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    idx = np.argmax(f1[:-1]) if len(thr) > 0 else 0
    return ThresholdChoice(
        threshold=float(thr[idx] if len(thr) else 0.5),
        f1=float(f1[idx]),
        precision=float(prec[idx]),
        recall=float(rec[idx]),
    )

def compute_curves(y_true: np.ndarray, proba: np.ndarray) -> dict:
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "tpr": tpr,
        "ap": float(average_precision_score(y_true, proba)),
        "auc": float(roc_auc_score(y_true, proba)),
    }

def top_permutation_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n: int = 10) -> pd.Series:
    r = permutation_importance(pipe, X, y, n_repeats=3, random_state=0, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=pipe.feature_names_in_).sort_values(ascending=False)
    return imp.head(n)
