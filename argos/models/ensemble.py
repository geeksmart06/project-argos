"""
Ensemble scoring.

Combines (when available) the calibrated supervised probability with
the unsupervised anomaly score and simple rule-based boosts, into one
final score plus a human-readable list of reasons per transaction.

IMPORTANT — honesty about what the score means:
  - If a trained supervised model is loaded, the output column is
    `fraud_probability` and IS a calibrated probability.
  - If no supervised model is available, the pipeline runs in
    unsupervised-only mode. The output column is `anomaly_score` and
    is explicitly NOT a probability of fraud — it only reflects
    statistical unusualness within this batch. The pipeline surfaces
    this distinction to the caller so it doesn't get misrepresented
    downstream (dashboard, reports, resume, interviews, etc).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .supervised import SupervisedFraudModel
from .unsupervised import UnsupervisedAnomalyScorer


@dataclass
class EnsembleConfig:
    flag_threshold: float = 0.5
    high_risk_threshold: float = 0.8
    supervised_weight: float = 0.6
    unsupervised_weight: float = 0.25
    rules_weight: float = 0.15


def _rule_boost(row: pd.Series) -> float:
    boost = 0.0
    if row.get("new_payee_large_amount", 0) == 1:
        boost += 0.4
    if row.get("velocity_count_1h", 0) >= 3:
        boost += 0.3
    if row.get("is_night", 0) == 1 and row.get("abs_amount", 0) > row.get("payee_median", 0) * 3:
        boost += 0.2
    if row.get("is_round_amount", 0) == 1 and row.get("abs_amount", 0) > 10000:
        boost += 0.1
    return min(boost, 1.0)


def _reasons_for_row(row: pd.Series, mode: str) -> str:
    reasons = []
    if row.get("new_payee_large_amount", 0) == 1:
        reasons.append("Large first-time transaction to a new payee")
    if row.get("velocity_count_1h", 0) >= 3:
        reasons.append(f"{int(row['velocity_count_1h'])} transactions within 1 hour")
    if row.get("is_night", 0) == 1:
        reasons.append("Occurred late at night")
    if row.get("payee_rel_dev", 0) > 5:
        reasons.append("Amount far from this payee's usual amount")
    if row.get("is_round_amount", 0) == 1 and row.get("abs_amount", 0) > 10000:
        reasons.append("Suspiciously round large amount")
    if mode == "supervised" and row.get("fraud_probability", 0) > 0.7:
        reasons.append("Model flagged strong fraud pattern match")
    elif mode == "unsupervised" and row.get("anomaly_score", 0) > 0.7:
        reasons.append("Statistically unusual vs. rest of this statement")
    return "; ".join(reasons)


class EnsembleScorer:
    def __init__(self, config: Optional[EnsembleConfig] = None,
                 supervised_model_path: Optional[str] = None,
                 contamination: float = 0.02):
        self.config = config or EnsembleConfig()
        self.supervised_model: Optional[SupervisedFraudModel] = None
        self.mode = "unsupervised"

        if supervised_model_path and SupervisedFraudModel.is_available(supervised_model_path):
            self.supervised_model = SupervisedFraudModel.load(supervised_model_path)
            self.mode = "supervised"

        self.unsupervised = UnsupervisedAnomalyScorer(contamination=contamination)

    def score(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        X = out[feature_cols].values
        rule_boosts = out.apply(_rule_boost, axis=1).values

        if self.mode == "supervised":
            sup_feature_cols = self.supervised_model.feature_cols or feature_cols
            missing = [c for c in sup_feature_cols if c not in out.columns]
            if missing:
                raise ValueError(
                    f"Supervised model expects features {missing} that are not "
                    "present in the engineered dataframe. Re-train or check "
                    "feature_engineer output."
                )
            X_sup = out[sup_feature_cols].values
            fraud_prob = self.supervised_model.predict_proba(X_sup)
            anomaly_score = self.unsupervised.score(X)

            final = (
                self.config.supervised_weight * fraud_prob
                + self.config.unsupervised_weight * anomaly_score
                + self.config.rules_weight * rule_boosts
            )
            out["fraud_probability"] = fraud_prob
            out["anomaly_score"] = anomaly_score
            out["suspicion_score"] = np.clip(final, 0, 1)
        else:
            anomaly_score = self.unsupervised.score(X)
            final = (
                (self.config.unsupervised_weight + self.config.supervised_weight) * anomaly_score
                + self.config.rules_weight * rule_boosts
            )
            out["anomaly_score"] = anomaly_score
            out["suspicion_score"] = np.clip(final, 0, 1)

        out["flagged"] = (out["suspicion_score"] >= self.config.flag_threshold).astype(int)
        out["high_risk"] = (out["suspicion_score"] >= self.config.high_risk_threshold).astype(int)
        out["reasons"] = out.apply(lambda r: _reasons_for_row(r, self.mode), axis=1)
        out.attrs["scoring_mode"] = self.mode
        return out
