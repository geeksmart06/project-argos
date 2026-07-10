from datetime import datetime, timedelta

import pandas as pd

from argos.features.engineer import engineer_features


def _sample_df():
    base = datetime(2026, 1, 1, 10, 0)
    rows = [
        {"date": base, "description": "Coffee Shop", "amount": -250.0, "balance": None},
        {"date": base + timedelta(hours=1), "description": "Coffee Shop", "amount": -260.0, "balance": None},
        {"date": base + timedelta(days=1, hours=2), "description": "New Payee XYZ", "amount": -50000.0, "balance": None},
    ]
    return pd.DataFrame(rows)


def test_engineer_features_adds_expected_columns():
    df, feature_cols = engineer_features(_sample_df())
    for col in ["abs_amount", "log_amount", "hour", "payee_count",
                "is_new_payee", "new_payee_large_amount", "velocity_count_1h"]:
        assert col in df.columns
    assert set(feature_cols).issubset(df.columns)


def test_repeat_payee_has_higher_count_than_new_payee():
    df, _ = engineer_features(_sample_df())
    coffee_rows = df[df["description"] == "Coffee Shop"]
    new_payee_row = df[df["description"] == "New Payee XYZ"]
    assert (coffee_rows["payee_count"] == 2).all()
    assert (new_payee_row["payee_count"] == 1).all()


def test_new_payee_large_amount_flagged():
    df, _ = engineer_features(_sample_df())
    new_payee_row = df[df["description"] == "New Payee XYZ"].iloc[0]
    assert new_payee_row["new_payee_large_amount"] == 1


def test_no_nan_or_inf_in_feature_columns():
    df, feature_cols = engineer_features(_sample_df())
    assert not df[feature_cols].isna().any().any()
    assert not (df[feature_cols].abs() == float("inf")).any().any()
