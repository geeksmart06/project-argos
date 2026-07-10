#!/usr/bin/env python3
"""
Project Argos — Streamlit dashboard.

Upload a statement, get flagged transactions, alerts, and a downloadable
report, all in the browser. Run with:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from argos.config import ArgosConfig
from argos.pipeline import FraudDetectionPipeline

st.set_page_config(page_title="Project Argos", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_pipeline() -> FraudDetectionPipeline:
    return FraudDetectionPipeline(config=ArgosConfig.load())


def save_upload_to_tmp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def main():
    st.title("🛡️ Project Argos")
    st.caption("Financial fraud & anomaly detector — upload a statement, review what got flagged.")

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload statement (CSV, XLSX, or PDF)", type=["csv", "xlsx", "xls", "pdf"])
        use_sample = st.button("Use sample demo data instead")
        st.divider()
        st.caption(
            "PDF parsing is best-effort — CSV/Excel exports from your bank "
            "are far more reliable. See the parse report after upload."
        )

    input_path = None
    display_name = None

    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_statement.csv")
        if not os.path.exists(sample_path):
            st.error("Sample data not found. Run: python scripts/generate_sample_data.py")
            return
        input_path = sample_path
        display_name = "sample_statement.csv (synthetic demo data)"
    elif uploaded is not None:
        input_path = save_upload_to_tmp(uploaded)
        display_name = uploaded.name

    if not input_path:
        st.info("Upload a statement or click **Use sample demo data** in the sidebar to get started.")
        return

    pipeline = get_pipeline()

    with st.spinner("Parsing and scoring transactions..."):
        try:
            result = pipeline.run(input_path, output_dir=tempfile.mkdtemp())
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    df = result.scored_df
    report = result.parse_report

    # --- Parse feedback ---
    with st.expander(f"Parse report — {display_name}", expanded=report.skipped_count > 0):
        st.write(report.summary())
        if report.skipped_count > 0:
            st.warning(f"{report.skipped_count} row(s) could not be parsed and were skipped.")
            st.dataframe(pd.DataFrame(report.skipped_rows), use_container_width=True)

    # --- Scoring mode disclaimer ---
    if result.scoring_mode == "unsupervised":
        st.info(
            "**No trained supervised model found** — scores below are "
            "`anomaly_score` values (statistical unusualness within this "
            "statement), not calibrated fraud probabilities. "
            "Train one with `python cli.py train ...` for calibrated probabilities.",
            icon="ℹ️",
        )
    else:
        st.success("Using calibrated `fraud_probability` from the trained supervised model.", icon="✅")

    # --- High risk alert banner ---
    high_risk_df = df[df["high_risk"] == 1]
    if not high_risk_df.empty:
        st.error(
            f"🚨 {len(high_risk_df)} high-risk transaction(s) detected "
            f"(score ≥ {pipeline.config.get('thresholds.high_risk_score', 0.8)})",
            icon="🚨",
        )

    # --- Summary cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analyzed", len(df))
    c2.metric("Flagged", int(df["flagged"].sum()))
    c3.metric("High risk", int(df["high_risk"].sum()))
    flagged_amount = df.loc[df["flagged"] == 1, "amount"].abs().sum()
    c4.metric("Flagged amount", f"₹{flagged_amount:,.0f}")

    st.divider()

    # --- Charts ---
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Suspicion score distribution")
        fig = px.histogram(df, x="suspicion_score", nbins=30,
                            color=df["flagged"].map({0: "normal", 1: "flagged"}),
                            color_discrete_map={"normal": "#4C78A8", "flagged": "#E45756"})
        fig.update_layout(legend_title_text="", xaxis_title="Suspicion score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("Transactions over time")
        fig2 = px.scatter(df, x="date", y="abs_amount",
                           color=df["flagged"].map({0: "normal", 1: "flagged"}),
                           color_discrete_map={"normal": "#4C78A8", "flagged": "#E45756"},
                           size="suspicion_score", hover_data=["description", "reasons"])
        fig2.update_layout(legend_title_text="", xaxis_title="Date", yaxis_title="Amount")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Flagged transactions table ---
    st.subheader("Flagged transactions")
    risk_filter = st.multiselect("Filter by risk level", ["HIGH", "MEDIUM"], default=["HIGH", "MEDIUM"])
    flagged = df[df["flagged"] == 1].copy()
    flagged["risk_level"] = flagged["high_risk"].map({1: "HIGH", 0: "MEDIUM"})
    flagged = flagged[flagged["risk_level"].isin(risk_filter)]
    flagged = flagged.sort_values("suspicion_score", ascending=False)

    display_cols = ["date", "description", "amount", "suspicion_score", "risk_level", "reasons"]
    st.dataframe(flagged[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # --- Downloads ---
    dl1, dl2 = st.columns(2)
    with open(result.csv_path, "rb") as f:
        dl1.download_button("Download scored CSV", f, file_name=os.path.basename(result.csv_path))
    with open(result.html_report_path, "rb") as f:
        dl2.download_button("Download HTML report", f, file_name=os.path.basename(result.html_report_path))


if __name__ == "__main__":
    main()
