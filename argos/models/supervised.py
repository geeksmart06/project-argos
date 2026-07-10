"""
Supervised fraud classifier.

Trained on a LABELED dataset (e.g. Kaggle Credit Card Fraud Detection,
or PaySim) — see scripts/train_supervised.py. This is what actually
learns fraud *patterns* instead of just flagging statistical outliers.

Raw XGBoost outputs are not well-calibrated probabilities, so this
wraps the model in CalibratedClassifierCV (isotonic regression) —
without this step, a "78% confidence" score doesn't actually mean
"78% of similar past cases were fraud."
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class TrainingMetrics:
    auc_roc: float
    auc_pr: float
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float
    n_train: int
    n_test: int
    n_fraud_test: int

    def as_dict(self) -> dict:
        return self.__dict__

    def summary(self) -> str:
        return (
            f"AUC-ROC: {self.auc_roc:.4f} | AUC-PR: {self.auc_pr:.4f} | "
            f"Precision@{self.threshold:.2f}: {self.precision_at_threshold:.4f} | "
            f"Recall@{self.threshold:.2f}: {self.recall_at_threshold:.4f} "
            f"(test set: {self.n_test} rows, {self.n_fraud_test} labeled fraud)"
        )


class SupervisedFraudModel:
    def __init__(self):
        self.model: Optional[CalibratedClassifierCV] = None
        self.feature_cols: Optional[list] = None

    def train(self, X: np.ndarray, y: np.ndarray, feature_cols: list,
              test_size: float = 0.2, threshold: float = 0.5,
              random_state: int = 42) -> TrainingMetrics:
        self.feature_cols = feature_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        base_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
        )

        # Isotonic calibration so predict_proba is a genuine probability,
        # not just a ranking score.
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        self.model = calibrated

        probs = calibrated.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, probs)
        auc_pr = average_precision_score(y_test, probs)

        preds_at_threshold = (probs >= threshold).astype(int)
        tp = int(((preds_at_threshold == 1) & (y_test == 1)).sum())
        fp = int(((preds_at_threshold == 1) & (y_test == 0)).sum())
        fn = int(((preds_at_threshold == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return TrainingMetrics(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            precision_at_threshold=precision,
            recall_at_threshold=recall,
            threshold=threshold,
            n_train=len(X_train),
            n_test=len(X_test),
            n_fraud_test=int(y_test.sum()),
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained/loaded. Call train() or load().")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str) -> "SupervisedFraudModel":
        if not os.path.exists(path):
            raise FileNotFoundError(f"No supervised model found at {path}")
        payload = joblib.load(path)
        instance = cls()
        instance.model = payload["model"]
        instance.feature_cols = payload["feature_cols"]
        return instance

    @staticmethod
    def is_available(path: str) -> bool:
        return os.path.exists(path)
