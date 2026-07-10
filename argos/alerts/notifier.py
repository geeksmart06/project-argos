"""
Alerting. Pluggable channels so a GitHub reader can wire in whatever
they actually use (console for local runs, email/webhook for anything
closer to production). Each channel fails soft — a broken webhook
should never crash the analysis pipeline.
"""
from __future__ import annotations

import json
import smtplib
from email.mime.text import MIMEText
from typing import List, Optional

import pandas as pd
import requests


class Notifier:
    def __init__(self, channel: str = "console", webhook_url: Optional[str] = None,
                 email_config: Optional[dict] = None):
        self.channel = channel
        self.webhook_url = webhook_url
        self.email_config = email_config or {}

    def notify_high_risk(self, high_risk_df: pd.DataFrame, statement_name: str) -> None:
        if high_risk_df.empty:
            return

        message = self._build_message(high_risk_df, statement_name)

        if self.channel == "console":
            print("\n" + "=" * 60)
            print("  ALERT — HIGH RISK TRANSACTIONS DETECTED")
            print("=" * 60)
            print(message)
        elif self.channel == "webhook":
            self._send_webhook(high_risk_df, statement_name, message)
        elif self.channel == "email":
            self._send_email(message)
        else:
            print(f"[notifier] Unknown channel '{self.channel}', falling back to console.")
            print(message)

    def _build_message(self, df: pd.DataFrame, statement_name: str) -> str:
        lines = [f"Statement: {statement_name}", f"{len(df)} high-risk transaction(s) found:\n"]
        for _, row in df.iterrows():
            lines.append(
                f"  - {row['date']} | {row['description'][:40]:40s} | "
                f"amount: {row['amount']:.2f} | score: {row['suspicion_score']:.2f} | "
                f"{row.get('reasons', '')}"
            )
        return "\n".join(lines)

    def _send_webhook(self, df: pd.DataFrame, statement_name: str, message: str) -> None:
        if not self.webhook_url:
            print("[notifier] Webhook channel selected but no webhook_url configured.")
            return
        payload = {
            "statement": statement_name,
            "high_risk_count": len(df),
            "summary": message,
            "transactions": json.loads(df.to_json(orient="records", date_format="iso")),
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
        except requests.RequestException as e:
            print(f"[notifier] Webhook delivery failed (analysis continues): {e}")

    def _send_email(self, message: str) -> None:
        cfg = self.email_config
        required = ["smtp_host", "smtp_port", "sender", "recipient"]
        if not all(cfg.get(k) for k in required):
            print("[notifier] Email channel selected but config incomplete; skipping send.")
            return
        try:
            msg = MIMEText(message)
            msg["Subject"] = "Project Argos — High Risk Transactions Detected"
            msg["From"] = cfg["sender"]
            msg["To"] = cfg["recipient"]
            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.starttls()
                if cfg.get("username") and cfg.get("password"):
                    server.login(cfg["username"], cfg["password"])
                server.sendmail(cfg["sender"], [cfg["recipient"]], msg.as_string())
        except Exception as e:
            print(f"[notifier] Email delivery failed (analysis continues): {e}")
