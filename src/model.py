"""Train, cross-validate and evaluate the bot detector.

Design choices that matter for honesty of the numbers:

* **Every metric is out-of-fold.**  We use ``cross_val_predict`` with
  stratified 5-fold CV, so each user is scored by a model that never saw them
  during training.  No user is evaluated against a model fit on their own row.
* **Class imbalance is handled explicitly** (``class_weight="balanced"``).  The
  original pipeline left it on the default and paid for it with 0.48 recall on
  the minority (bot) class.
* **Several models are compared** so the choice is evidence-based, not assumed.

The headline model is a balanced Random Forest: it keeps the original project's
model family, exposes interpretable feature importances, and recovers the
recall the first version was losing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

RANDOM_STATE = 42
N_SPLITS = 5

# Reported by the original, pre-polish pipeline (RF defaults, 2 of 4 datasets).
# Kept here so the showcase can quantify the improvement.
BASELINE = {
    "label": "Original (RF defaults, 2 datasets)",
    "accuracy": 0.91,
    "bot_precision": 0.93,
    "bot_recall": 0.48,
    "bot_f1": 0.63,
}


def candidate_models() -> dict[str, Any]:
    """The model zoo we benchmark against each other."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "Random Forest (default)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


SELECTED_MODEL = "Random Forest (balanced)"


def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


@dataclass
class Evaluation:
    """All numbers needed to describe one model's out-of-fold performance."""

    name: str
    proba: np.ndarray  # out-of-fold P(bot) per user
    pred: np.ndarray
    roc_auc: float
    pr_auc: float
    accuracy: float
    bot_precision: float
    bot_recall: float
    bot_f1: float

    def summary(self) -> dict[str, float]:
        return {
            "name": self.name,
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "accuracy": round(self.accuracy, 4),
            "bot_precision": round(self.bot_precision, 4),
            "bot_recall": round(self.bot_recall, 4),
            "bot_f1": round(self.bot_f1, 4),
        }


def evaluate_model(name: str, model: Any, X: np.ndarray, y: np.ndarray) -> Evaluation:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    return Evaluation(
        name=name,
        proba=proba,
        pred=pred,
        roc_auc=roc_auc_score(y, proba),
        pr_auc=average_precision_score(y, proba),
        accuracy=float((pred == y).mean()),
        bot_precision=precision,
        bot_recall=recall,
        bot_f1=_f1(precision, recall),
    )


def cross_validated_importances(model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean impurity-based importance across folds (more stable than one fit)."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    importances = []
    for train_idx, _ in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        importances.append(model.feature_importances_)
    return np.mean(importances, axis=0)


@dataclass
class Results:
    """Everything the export layer and the README need."""

    n_users: int
    n_bots: int
    n_features: int
    evaluations: dict[str, Evaluation]
    selected: str
    importances: list[tuple[str, float]]
    feature_table: pd.DataFrame = field(repr=False)

    @property
    def best(self) -> Evaluation:
        return self.evaluations[self.selected]


def run(feature_table: pd.DataFrame) -> Results:
    """Benchmark every candidate, then build the full results bundle."""
    X = feature_table[FEATURE_NAMES].to_numpy()
    y = feature_table["is_bot"].to_numpy()

    evaluations = {
        name: evaluate_model(name, model, X, y)
        for name, model in candidate_models().items()
    }

    # Feature importances come from the selected (interpretable) model.
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    )
    importances = cross_validated_importances(rf, X, y)
    ranked = sorted(
        zip(FEATURE_NAMES, importances), key=lambda kv: kv[1], reverse=True
    )

    return Results(
        n_users=len(feature_table),
        n_bots=int(y.sum()),
        n_features=len(FEATURE_NAMES),
        evaluations=evaluations,
        selected=SELECTED_MODEL,
        importances=[(name, float(score)) for name, score in ranked],
        feature_table=feature_table,
    )


def curve_points(y: np.ndarray, proba: np.ndarray, max_points: int = 80) -> dict[str, Any]:
    """ROC and precision-recall curves, down-sampled for a lightweight payload."""
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)

    def thin(arr: np.ndarray) -> list[float]:
        if len(arr) <= max_points:
            return [round(float(v), 4) for v in arr]
        idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
        return [round(float(arr[i]), 4) for i in idx]

    return {
        "roc": {"fpr": thin(fpr), "tpr": thin(tpr)},
        "pr": {"precision": thin(prec), "recall": thin(rec)},
    }


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
