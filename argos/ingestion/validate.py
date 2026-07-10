"""
Post-ingestion validation. Runs regardless of whether data came from
CSV, Excel, or PDF, so downstream feature engineering always receives
a clean, minimally-guaranteed schema.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"date", "description", "amount"}


class ValidationError(Exception):
    pass


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValidationError(f"Missing required columns after parsing: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows that are structurally unusable and coerces dtypes.
    Unlike the original prototype, this NEVER fabricates a date for a
    row that failed to parse — such rows are dropped and should show
    up in the ParseReport from ingestion instead.
    """
    validate_schema(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["date", "amount"]).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        out.attrs["rows_dropped_in_cleaning"] = dropped
    return out
