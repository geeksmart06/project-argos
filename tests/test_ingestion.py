import os
import tempfile

import pandas as pd
import pytest

from argos.ingestion import csv_loader


def _write_csv(rows, columns):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(tmp.name, index=False)
    return tmp.name


def test_loads_debit_credit_style_csv():
    path = _write_csv(
        rows=[
            ["01-01-2026", "Coffee Shop", "250", "", "9750"],
            ["02-01-2026", "Salary Credit", "", "50000", "59750"],
        ],
        columns=["Date", "Narration", "Debit", "Credit", "Balance"],
    )
    df, report = csv_loader.load_transactions(path)
    os.unlink(path)

    assert len(df) == 2
    assert report.parsed_rows == 2
    assert report.skipped_count == 0
    assert df.iloc[0]["amount"] == -250.0
    assert df.iloc[1]["amount"] == 50000.0


def test_loads_single_amount_column_csv():
    path = _write_csv(
        rows=[["01-01-2026", "Grocery", "-500"], ["02-01-2026", "Refund", "200"]],
        columns=["Transaction Date", "Details", "Amount"],
    )
    df, report = csv_loader.load_transactions(path)
    os.unlink(path)

    assert len(df) == 2
    assert df.iloc[0]["amount"] == -500.0


def test_skips_unparseable_rows_without_crashing():
    path = _write_csv(
        rows=[
            ["not-a-date", "Bad Row", "100"],
            ["03-01-2026", "Good Row", "300"],
        ],
        columns=["Date", "Description", "Amount"],
    )
    df, report = csv_loader.load_transactions(path)
    os.unlink(path)

    assert len(df) == 1
    assert report.skipped_count == 1
    assert "unparseable date" in report.skipped_rows[0]["reason"]


def test_missing_required_columns_raises():
    path = _write_csv(rows=[["a", "b"]], columns=["Foo", "Bar"])
    with pytest.raises(ValueError):
        csv_loader.load_transactions(path)
    os.unlink(path)
