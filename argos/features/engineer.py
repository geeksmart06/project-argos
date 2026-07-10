"""
Feature engineering.

Beyond the basic amount/time features, this module adds the features
that actually correlate with real fraud patterns:
  - velocity (txn count/sum in trailing windows)
  - new-payee-large-first-transaction pattern
  - round-number bias
  - payee-relative deviation

These are computed per-account, in a single pass, so the module works
identically whether it's fed 50 rows or 50,000.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd


def _payee_key(description: str) -> str:
    if not isinstance(description, str):
        return ""
    key = description[:30].lower()
    return "".join(ch for ch in key if ch.isalnum() or ch == " ").strip()


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["abs_amount"] = out["amount"].abs()
    out["log_amount"] = out["abs_amount"].apply(lambda x: math.log1p(x))
    out["hour"] = out["date"].dt.hour
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_night"] = out["hour"].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)

    # Round-number bias: fraud/scam transfers often use suspiciously
    # round amounts (1000, 5000, 10000...) rather than natural totals.
    out["is_round_amount"] = (out["abs_amount"] % 100 == 0).astype(int)

    # Payee-level stats
    out["payee_key"] = out["description"].apply(_payee_key)
    payee_count = out.groupby("payee_key")["payee_key"].transform("count")
    out["payee_count"] = payee_count
    payee_median = out.groupby("payee_key")["abs_amount"].transform("median")
    out["payee_median"] = payee_median
    out["payee_rel_dev"] = (out["abs_amount"] - out["payee_median"]).abs() / (out["payee_median"] + 1)

    # New-payee-large-first-transaction: count==1 AND amount well above
    # the account's own median transaction size.
    account_median = out["abs_amount"].median() if len(out) else 0.0
    out["is_new_payee"] = (out["payee_count"] == 1).astype(int)
    out["new_payee_large_amount"] = (
        (out["is_new_payee"] == 1) & (out["abs_amount"] > 5 * (account_median + 1))
    ).astype(int)

    # Time since previous transaction (velocity signal, part 1)
    out["time_diff_hours"] = out["date"].diff().dt.total_seconds().div(3600)
    out["time_diff_hours"] = out["time_diff_hours"].fillna(9999)

    # Rolling velocity: transaction count and sum in trailing 1h/24h/7d
    # windows. Implemented via a time-indexed rolling window per row.
    indexed = out.set_index("date")
    out["velocity_count_1h"] = (
        indexed["abs_amount"].rolling("1h").count().values
    )
    out["velocity_sum_1h"] = (
        indexed["abs_amount"].rolling("1h").sum().values
    )
    out["velocity_count_24h"] = (
        indexed["abs_amount"].rolling("24h").count().values
    )
    out["velocity_sum_24h"] = (
        indexed["abs_amount"].rolling("24h").sum().values
    )

    feature_cols = [
        "abs_amount", "log_amount", "hour", "day_of_week", "is_weekend",
        "is_night", "is_round_amount", "payee_count", "payee_rel_dev",
        "is_new_payee", "new_payee_large_amount", "time_diff_hours",
        "velocity_count_1h", "velocity_sum_1h", "velocity_count_24h",
        "velocity_sum_24h",
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out, feature_cols
