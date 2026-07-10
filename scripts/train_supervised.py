#!/usr/bin/env python3
"""
Trains the supervised fraud classifier on a LABELED dataset.

This is deliberately generic about schema: it auto-detects numeric
feature columns and excludes the label column (plus any obvious
ID/string columns), so it works with either of the standard public
benchmarks:

  - Kaggle "Credit Card Fraud Detection"
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    (columns: Time, V1..V28, Amount, Class)

  - PaySim synthetic mobile money dataset
    https://www.kaggle.com/datasets/ealaxi/paysim1
    (columns: step, type, amount, oldbalanceOrg, newbalanceOrig,
     oldbalanceDest, newbalanceDest, isFraud, ...)
    NOTE: encode the categorical 'type' column to numeric before
    training, or pass --feature-cols explicitly.

IMPORTANT — read this before assuming this model works on your own
uploaded bank statements out of the box:

These public datasets are already in their OWN feature space (PCA
components, or raw numeric transaction fields). They do NOT contain
'description' text, payee history, or velocity features the way
argos/features/engineer.py produces for a real statement upload.

So: this script proves out a properly calibrated, correctly-evaluated
supervised model (with honest precision/recall/AUC-PR, not "accuracy")
on a real benchmark — which is the right thing to have on a resume/repo.
But EnsembleScorer will only use a trained model in production if its
feature_cols match what engineer_features() outputs. If they don't
match (which they won't, out of the box, against Kaggle/PaySim), the
pipeline correctly falls back to unsupervised + rules and labels its
output as `anomaly_score`, not `fraud_probability` — see
argos/models/ensemble.py. To close that gap for real, you'd need
labeled data in the SAME feature space as engineer_features(), which
is realistically an internal/proprietary dataset, not a public one.

Usage:
    python scripts/train_supervised.py path/to/creditcard.csv --label-col Class
    python scripts/train_supervised.py path/to/creditcard.csv --label-col Class \
        --feature-cols Time V1 V2 V3 Amount --output models/supervised_model.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from argos.models.supervised import SupervisedFraudModel


def auto_detect_feature_cols(df: pd.DataFrame, label_col: str) -> list:
    exclude_patterns = ("id", "name", "nameorig", "namedest")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [
        c for c in numeric_cols
        if c != label_col and not any(p in c.lower() for p in exclude_patterns)
    ]
    return feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--label-col", default="Class")
    parser.add_argument("--feature-cols", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="models/supervised_model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    if args.label_col not in df.columns:
        raise SystemExit(f"Label column '{args.label_col}' not found. Columns: {list(df.columns)}")

    feature_cols = args.feature_cols or auto_detect_feature_cols(df, args.label_col)
    if not feature_cols:
        raise SystemExit("No usable numeric feature columns detected. Pass --feature-cols explicitly.")

    print(f"Using {len(feature_cols)} feature columns: {feature_cols}")
    print(f"Label distribution:\n{df[args.label_col].value_counts()}\n")

    X = df[feature_cols].fillna(0.0).values
    y = df[args.label_col].values

    model = SupervisedFraudModel()
    metrics = model.train(X, y, feature_cols=feature_cols, threshold=args.threshold)

    print("\n--- Held-out test set metrics (NOT accuracy — see README for why) ---")
    print(metrics.summary())

    model.save(args.output)
    print(f"\nSaved trained + calibrated model to {args.output}")

    metrics_path = str(Path(args.output).with_suffix(".metrics.json"))
    with open(metrics_path, "w") as f:
        json.dump(metrics.as_dict(), f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
