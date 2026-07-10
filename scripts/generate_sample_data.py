#!/usr/bin/env python3
"""
Generates a synthetic bank-statement CSV with a handful of injected
anomalies, so the pipeline/dashboard can be demoed without needing a
real statement or a Kaggle download.

Usage:
    python scripts/generate_sample_data.py --out data/sample_statement.csv
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

MERCHANTS = [
    "Swiggy", "Amazon", "Zomato", "Uber", "Netflix Subscription",
    "Local Grocery Mart", "Electricity Board", "Rent Payment",
    "Spotify", "Gym Membership", "Coffee Shop", "Fuel Station",
]


def generate(n_normal: int = 300, n_anomalies: int = 12) -> pd.DataFrame:
    rows = []
    start = datetime.now() - timedelta(days=90)

    for _ in range(n_normal):
        merchant = random.choice(MERCHANTS)
        dt = start + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(8, 21),
            minutes=random.randint(0, 59),
        )
        base_amounts = {
            "Rent Payment": 15000, "Electricity Board": 1800, "Netflix Subscription": 499,
            "Spotify": 119, "Gym Membership": 1200,
        }
        amt = base_amounts.get(merchant, random.uniform(100, 2000))
        amt = round(amt * random.uniform(0.9, 1.1), 2)
        rows.append({"Date": dt, "Description": merchant, "Debit": amt, "Credit": "", "Balance": ""})

    # Injected anomalies: fraud-like patterns.
    # Each "new payee" style anomaly gets a UNIQUE payee identifier —
    # real fraud rarely repeats the exact same payee string, and reusing
    # one name across rows would defeat our own new-payee/velocity
    # features (they'd stop looking "new" after the first occurrence).
    for i in range(n_anomalies // 3):
        day_offset = random.randint(0, 90)
        dt = (start + timedelta(days=day_offset)).replace(
            hour=random.choice([1, 2, 3]), minute=random.randint(0, 59)
        )
        rows.append({"Date": dt, "Description": f"Unknown Wire Transfer REF{1000+i}",
                     "Debit": round(random.uniform(40000, 90000), 2), "Credit": "", "Balance": ""})

    # Rapid-fire small transfers to the SAME payee within an hour —
    # this one is intentionally a repeated payee, since velocity (not
    # novelty) is the signal we want this pattern to demonstrate.
    for _ in range(n_anomalies // 3):
        day_offset = random.randint(0, 90)
        base_dt = (start + timedelta(days=day_offset)).replace(hour=14, minute=0)
        for i in range(4):
            dt = base_dt + timedelta(minutes=i * 7)
            rows.append({"Date": dt, "Description": "QuickPay Transfer",
                         "Debit": round(random.uniform(9000, 9999), 2), "Credit": "", "Balance": ""})

    for i in range(n_anomalies - 2 * (n_anomalies // 3)):
        day_offset = random.randint(0, 90)
        dt = (start + timedelta(days=day_offset)).replace(hour=random.randint(9, 20), minute=random.randint(0, 59))
        rows.append({"Date": dt, "Description": f"New Merchant {1000+i} Pvt Ltd",
                     "Debit": 50000.0, "Credit": "", "Balance": ""})

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%d-%m-%Y %H:%M")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_statement.csv")
    parser.add_argument("--n-normal", type=int, default=300)
    parser.add_argument("--n-anomalies", type=int, default=12)
    args = parser.parse_args()

    df = generate(args.n_normal, args.n_anomalies)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic transactions to {args.out}")
