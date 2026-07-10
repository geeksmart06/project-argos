"""
FraudDetectionPipeline — orchestrates ingestion -> validation ->
feature engineering -> scoring -> alerting -> reporting.

This is the single entry point both the CLI and the dashboard call
into, so both surfaces stay in sync with one implementation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .alerts.notifier import Notifier
from .config import ArgosConfig
from .features.engineer import engineer_features
from .ingestion import csv_loader, pdf_loader
from .ingestion.validate import clean
from .models.ensemble import EnsembleConfig, EnsembleScorer
from .reporting.report_generator import generate_html_report


@dataclass
class PipelineResult:
    scored_df: pd.DataFrame
    parse_report: object
    csv_path: str
    html_report_path: str
    scoring_mode: str


class FraudDetectionPipeline:
    def __init__(self, config: Optional[ArgosConfig] = None):
        self.config = config or ArgosConfig.load()

        ensemble_config = EnsembleConfig(
            flag_threshold=self.config.get("thresholds.flag_score", 0.5),
            high_risk_threshold=self.config.get("thresholds.high_risk_score", 0.8),
            supervised_weight=self.config.get("ensemble.supervised_weight", 0.6),
            unsupervised_weight=self.config.get("ensemble.unsupervised_weight", 0.25),
            rules_weight=self.config.get("ensemble.rules_weight", 0.15),
        )
        model_path = self.config.get("supervised.model_path", "models/supervised_model.pkl")
        use_supervised = self.config.get("supervised.use_if_available", True)

        self.scorer = EnsembleScorer(
            config=ensemble_config,
            supervised_model_path=model_path if use_supervised else None,
            contamination=self.config.get("unsupervised.contamination", 0.02),
        )
        self.notifier = Notifier(
            channel=self.config.get("alerts.channel", "console"),
            webhook_url=self.config.get("alerts.webhook_url"),
            email_config=self.config.get("alerts.email", {}),
        )

    def run(self, input_path: str, output_dir: Optional[str] = None) -> PipelineResult:
        output_dir = output_dir or self.config.get("reporting.output_dir", "reports")
        os.makedirs(output_dir, exist_ok=True)

        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".pdf":
            raw_df, parse_report = pdf_loader.load_transactions(input_path)
        elif ext in (".csv", ".xlsx", ".xls"):
            raw_df, parse_report = csv_loader.load_transactions(input_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .csv, .xlsx, .xls, or .pdf")

        if raw_df.empty:
            raise ValueError(
                f"No transactions could be parsed from {input_path}. "
                f"Parse report: {parse_report.summary()}"
            )

        cleaned_df = clean(raw_df)
        featured_df, feature_cols = engineer_features(cleaned_df)
        scored_df = self.scorer.score(featured_df, feature_cols)

        basename = os.path.splitext(os.path.basename(input_path))[0]
        csv_out = os.path.join(output_dir, f"{basename}_scored.csv")
        html_out = os.path.join(output_dir, f"{basename}_report.html")

        scored_df.to_csv(csv_out, index=False)
        generate_html_report(
            scored_df, input_path, html_out,
            top_n=self.config.get("reporting.top_n_flagged", 25),
        )

        if self.config.get("alerts.enabled", True):
            high_risk = scored_df[scored_df["high_risk"] == 1]
            self.notifier.notify_high_risk(high_risk, os.path.basename(input_path))

        return PipelineResult(
            scored_df=scored_df,
            parse_report=parse_report,
            csv_path=csv_out,
            html_report_path=html_out,
            scoring_mode=self.scorer.mode,
        )
