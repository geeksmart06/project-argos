"""
PDF ingestion — SECONDARY / best-effort path.

PDFs are not a reliable source of tabular data: layout, fonts, and
column spacing vary bank to bank, and text-extraction libraries can't
always recover the true column boundaries. This module tries table
extraction first (camelot, if installed) and falls back to a
whitespace-based text heuristic, but callers should always inspect the
returned ParseReport before trusting the output. Prefer csv_loader for
anything that matters.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from dateutil import parser as dateparser

from .csv_loader import ParseReport, _to_float

try:
    import camelot  # type: ignore
    _HAS_CAMELOT = True
except ImportError:
    _HAS_CAMELOT = False

import pdfplumber


def _try_camelot(path: str) -> Optional[pd.DataFrame]:
    if not _HAS_CAMELOT:
        return None
    try:
        tables = camelot.read_pdf(path, pages="all", flavor="lattice")
        if len(tables) == 0:
            tables = camelot.read_pdf(path, pages="all", flavor="stream")
        if len(tables) == 0:
            return None
        frames = [t.df for t in tables]
        combined = pd.concat(frames, ignore_index=True)
        return combined
    except Exception:
        return None


def _fallback_text_extraction(path: str) -> list[str]:
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
    return lines


def load_transactions(path: str) -> tuple[pd.DataFrame, ParseReport]:
    """
    Best-effort PDF statement parsing.

    Returns (dataframe, ParseReport). Always check report.skipped_count
    and report.summary() — a "successful" run may still have silently
    dropped a meaningful fraction of rows.
    """
    camelot_df = _try_camelot(path)
    report = ParseReport()

    if camelot_df is not None and not camelot_df.empty:
        return _parse_table_rows(camelot_df, report)

    # Fallback: heuristic text-line splitting. Least reliable path.
    lines = _fallback_text_extraction(path)
    report.total_rows = len(lines)
    rows = []
    for i, ln in enumerate(lines):
        if re.search(r"date\s+description", ln, re.I):
            continue
        tokens = re.split(r"\s{2,}|\t", ln)
        if len(tokens) < 3:
            report.add_skip(i, f"could not tokenize line into >=3 columns: '{ln}'")
            continue

        date_raw, desc = tokens[0], tokens[1]
        try:
            dt = dateparser.parse(date_raw, dayfirst=True)
        except Exception:
            report.add_skip(i, f"unparseable date: '{date_raw}'")
            continue

        amount = None
        for tok in tokens[2:]:
            amt = _to_float(tok)
            if amt is not None:
                amount = amt
                break
        if amount is None:
            report.add_skip(i, f"no numeric amount found in line: '{ln}'")
            continue

        rows.append({"date": dt, "description": desc.strip(), "amount": amount,
                      "balance": None, "orig_row": i})

    df = pd.DataFrame(rows)
    report.parsed_rows = len(df)
    report.detected_columns = {"method": "text-heuristic (fallback — verify results)"}
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df, report


def _parse_table_rows(raw_table: pd.DataFrame, report: ParseReport) -> tuple[pd.DataFrame, ParseReport]:
    report.total_rows = len(raw_table)
    rows = []
    for i, row in raw_table.iterrows():
        cells = [str(c).strip() for c in row.tolist()]
        dt = None
        for cell in cells:
            try:
                dt = dateparser.parse(cell, dayfirst=True, fuzzy=False)
                break
            except Exception:
                continue
        if dt is None:
            report.add_skip(i, "no parseable date found in row")
            continue

        amount = None
        for cell in cells:
            amt = _to_float(cell)
            if amt is not None:
                amount = amt
        if amount is None:
            report.add_skip(i, "no parseable amount found in row")
            continue

        desc_candidates = [c for c in cells if c and not re.match(r"^[\d.,\-()]+$", c)]
        description = max(desc_candidates, key=len) if desc_candidates else ""

        rows.append({"date": dt, "description": description, "amount": amount,
                      "balance": None, "orig_row": i})

    df = pd.DataFrame(rows)
    report.parsed_rows = len(df)
    report.detected_columns = {"method": "camelot table extraction"}
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df, report
