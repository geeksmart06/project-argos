"""
CSV / Excel ingestion.

This is the PRIMARY ingestion path. Every bank offers a CSV or Excel
export of statements, and it is dramatically more reliable than
scraping tables out of PDFs. Column names vary a lot bank-to-bank, so
this module auto-detects the date/description/amount/balance columns
instead of assuming a fixed schema.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

# Candidate column-name patterns, checked in order, per logical field.
COLUMN_PATTERNS = {
    "date": [r"^date$", r"txn.?date", r"transaction.?date", r"value.?date", r"posting.?date"],
    "description": [r"description", r"narration", r"particulars", r"details", r"remarks", r"merchant"],
    "debit": [r"debit", r"withdrawal", r"^dr$", r"amount.?dr"],
    "credit": [r"credit", r"deposit", r"^cr$", r"amount.?cr"],
    "amount": [r"^amount$", r"^amt$", r"transaction.?amount"],
    "balance": [r"balance", r"closing.?bal"],
}


@dataclass
class ParseReport:
    total_rows: int = 0
    parsed_rows: int = 0
    skipped_rows: List[dict] = field(default_factory=list)
    detected_columns: dict = field(default_factory=dict)

    def add_skip(self, row_index: int, reason: str):
        self.skipped_rows.append({"row_index": row_index, "reason": reason})

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_rows)

    def summary(self) -> str:
        return (
            f"Parsed {self.parsed_rows}/{self.total_rows} rows "
            f"({self.skipped_count} skipped). "
            f"Detected columns: {self.detected_columns}"
        )


def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    normalized = {c: re.sub(r"[^a-z]", "", c.lower()) for c in columns}
    for pattern in patterns:
        for original, norm in normalized.items():
            if re.search(pattern.replace(".?", ""), norm):
                return original
    return None


def _detect_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    detected = {}
    for field_name, patterns in COLUMN_PATTERNS.items():
        detected[field_name] = _find_column(cols, patterns)
    return detected


def _to_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in {"-", "."}:
        return None
    try:
        v = float(s)
        return -abs(v) if negative else v
    except ValueError:
        return None


def load_transactions(path: str) -> tuple[pd.DataFrame, ParseReport]:
    """
    Load a CSV/XLS/XLSX bank statement export into a standardized
    DataFrame: [date, description, amount, balance, orig_row].

    Returns (dataframe, ParseReport) — the report tells the caller
    exactly what was skipped and why, instead of failing silently.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, dtype=str)
    else:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)

    report = ParseReport(total_rows=len(raw))
    detected = _detect_columns(raw)
    report.detected_columns = detected

    if not detected.get("date") or not detected.get("description"):
        raise ValueError(
            "Could not detect required 'date' and 'description' columns. "
            f"Available columns: {list(raw.columns)}. "
            "Rename columns or extend COLUMN_PATTERNS in csv_loader.py."
        )

    has_split_amount = detected.get("debit") or detected.get("credit")
    has_single_amount = detected.get("amount")

    if not has_split_amount and not has_single_amount:
        raise ValueError(
            "Could not detect an amount column (looked for debit/credit or a "
            f"single amount column). Available columns: {list(raw.columns)}"
        )

    rows = []
    for idx, row in raw.iterrows():
        date_raw = row.get(detected["date"], "")
        desc = row.get(detected["description"], "")

        try:
            dt = dateparser.parse(str(date_raw), dayfirst=True)
        except Exception:
            report.add_skip(idx, f"unparseable date: '{date_raw}'")
            continue

        if has_single_amount:
            amount = _to_float(row.get(detected["amount"]))
        else:
            debit = _to_float(row.get(detected.get("debit"), "")) if detected.get("debit") else None
            credit = _to_float(row.get(detected.get("credit"), "")) if detected.get("credit") else None
            if debit:
                amount = -abs(debit)
            elif credit:
                amount = abs(credit)
            else:
                amount = None

        if amount is None:
            report.add_skip(idx, f"unparseable/missing amount in row: {dict(row)}")
            continue

        balance = _to_float(row.get(detected.get("balance"), "")) if detected.get("balance") else None

        rows.append({
            "date": dt,
            "description": str(desc).strip(),
            "amount": amount,
            "balance": balance,
            "orig_row": idx,
        })

    df = pd.DataFrame(rows)
    report.parsed_rows = len(df)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df, report
