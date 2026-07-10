from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from argos.features.engineer import engineer_features
from argos.models.ensemble import EnsembleConfig, EnsembleScorer


def _make_df(n_normal=40):
    base = datetime(2026, 1, 1, 9, 0)
    rows = []
    for i in range(n_normal):
        rows.append({
            "date": base + timedelta(hours=i * 6),
            "description": "Regular Merchant",
            "amount": -np.random.uniform(100, 300),
            "balance": None,
        })
    # one obvious outlier
    rows.append({
        "date": base + timedelta(hours=n_normal * 6),
        "description": "Unknown Wire",
        "amount": -80000.0,
        "balance": None,
    })
    return pd.DataFrame(rows)


def test_ensemble_runs_unsupervised_only_without_trained_model():
    df, feature_cols = engineer_features(_make_df())
    scorer = EnsembleScorer(config=EnsembleConfig(), supervised_model_path="models/does_not_exist.pkl")
    assert scorer.mode == "unsupervised"

    scored = scorer.score(df, feature_cols)
    assert "anomaly_score" in scored.columns
    assert "fraud_probability" not in scored.columns
    assert "suspicion_score" in scored.columns
    assert scored["suspicion_score"].between(0, 1).all()


def test_large_outlier_scores_higher_than_typical_transaction():
    df, feature_cols = engineer_features(_make_df())
    scorer = EnsembleScorer(supervised_model_path="models/does_not_exist.pkl")
    scored = scorer.score(df, feature_cols)

    outlier_score = scored.iloc[-1]["suspicion_score"]
    median_normal_score = scored.iloc[:-1]["suspicion_score"].median()
    assert outlier_score > median_normal_score


def test_flagged_and_high_risk_are_consistent_with_threshold():
    df, feature_cols = engineer_features(_make_df())
    config = EnsembleConfig(flag_threshold=0.5, high_risk_threshold=0.8)
    scorer = EnsembleScorer(config=config, supervised_model_path="models/does_not_exist.pkl")
    scored = scorer.score(df, feature_cols)

    assert (scored.loc[scored["flagged"] == 1, "suspicion_score"] >= 0.5).all()
    assert (scored.loc[scored["high_risk"] == 1, "suspicion_score"] >= 0.8).all()
    # every high_risk row must also be flagged
    assert (scored.loc[scored["high_risk"] == 1, "flagged"] == 1).all()
