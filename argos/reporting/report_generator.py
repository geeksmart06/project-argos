"""
HTML report generation. Reads the shared Jinja2 template so report
styling lives in one place, not inlined in Python strings.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def generate_html_report(df: pd.DataFrame, filename: str, out_path: str, top_n: int = 25) -> str:
    scoring_mode = df.attrs.get("scoring_mode", "unsupervised")
    score_col = "fraud_probability" if scoring_mode == "supervised" else "anomaly_score"

    flagged_df = df[df["flagged"] == 1].sort_values("suspicion_score", ascending=False)

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("report.html")

    html = template.render(
        filename=os.path.basename(filename),
        generated=datetime.now().isoformat(timespec="seconds"),
        total=len(df),
        flagged_count=int(df["flagged"].sum()),
        high_risk_count=int(df["high_risk"].sum()) if "high_risk" in df.columns else 0,
        scoring_mode=scoring_mode,
        score_col=score_col,
        flagged=flagged_df.head(top_n).to_dict("records"),
        top_n=min(top_n, len(flagged_df)),
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
