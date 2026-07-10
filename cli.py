#!/usr/bin/env python3
"""
Project Argos CLI.

Usage:
    python cli.py analyze path/to/statement.csv [--output-dir reports]
    python cli.py train path/to/labeled_dataset.csv --label-col Class
    python cli.py dashboard
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from argos.config import ArgosConfig
from argos.pipeline import FraudDetectionPipeline


def cmd_analyze(args: argparse.Namespace) -> None:
    config = ArgosConfig.load(args.config) if args.config else ArgosConfig.load()
    pipeline = FraudDetectionPipeline(config=config)
    result = pipeline.run(args.input_path, output_dir=args.output_dir)

    print(f"\nScoring mode: {result.scoring_mode}")
    print(result.parse_report.summary())
    print(f"Scored CSV:  {result.csv_path}")
    print(f"HTML report: {result.html_report_path}")

    flagged = result.scored_df[result.scored_df["flagged"] == 1]
    print(f"\n{len(flagged)}/{len(result.scored_df)} transactions flagged "
          f"(threshold={config.get('thresholds.flag_score', 0.5)})")


def cmd_train(args: argparse.Namespace) -> None:
    # Delegates to scripts/train_supervised.py so training logic
    # lives in one place and can also be run standalone.
    cmd = [sys.executable, "scripts/train_supervised.py",
           args.dataset_path, "--label-col", args.label_col]
    if args.output:
        cmd += ["--output", args.output]
    subprocess.run(cmd, check=True)


def cmd_dashboard(args: argparse.Namespace) -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project Argos — fraud & anomaly detector")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Score a bank statement (CSV/XLSX/PDF)")
    p_analyze.add_argument("input_path")
    p_analyze.add_argument("--output-dir", default=None)
    p_analyze.add_argument("--config", default=None)
    p_analyze.set_defaults(func=cmd_analyze)

    p_train = sub.add_parser("train", help="Train the supervised model on a labeled dataset")
    p_train.add_argument("dataset_path")
    p_train.add_argument("--label-col", default="Class")
    p_train.add_argument("--output", default=None)
    p_train.set_defaults(func=cmd_train)

    p_dash = sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
