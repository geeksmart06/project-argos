# Project Argos

A modular fraud & anomaly detector for bank statements — upload a CSV/Excel/PDF
statement, get flagged transactions with explanations, alerts on high-risk
activity, and a downloadable report. Includes a Streamlit dashboard.

This is a rebuild of an earlier prototype that parsed PDFs with regex and
scored transactions with unsupervised models only. This version fixes that
prototype's core problems and adds a supervised-learning path, a proper
ingestion layer with parse-error reporting, and a web dashboard.

## Why this exists / what changed from v1

The original script had a few silent failure modes worth naming, because
they're common mistakes in fraud-detection side projects generally:

- **Fabricated dates.** Unparseable dates were replaced with `pd.date_range`
  placeholders, which then fed into time-based features as if they were real.
  This version drops unparseable rows and reports exactly which ones and why.
- **Silent zeros.** Failed amount parsing defaulted to `0.0`, indistinguishable
  from a genuine zero-amount transaction. Now every skipped row is logged in
  a `ParseReport`.
- **Fixed contamination rate.** `IsolationForest(contamination=0.02)` assumes
  2% of every statement is fraud, regardless of what's actually in it. Still
  a real limitation of unsupervised-only scoring — see "Scoring modes" below.
- **Anomaly ≠ fraud.** The old script called its output a "suspicion score"
  with no distinction between "statistically unusual" and "matches known
  fraud patterns." This version makes that distinction explicit in the code,
  the output columns, and the report.

## Architecture

```
statement (csv/xlsx/pdf)
        │
        ▼
  argos/ingestion/          CSV/Excel = primary path (schema auto-detect)
  (csv_loader, pdf_loader)   PDF = best-effort fallback (camelot → text heuristic)
        │  → returns (DataFrame, ParseReport)
        ▼
  argos/ingestion/validate.py    drops structurally invalid rows, no fabrication
        │
        ▼
  argos/features/engineer.py     velocity, payee history, round-number bias,
        │                        new-payee-large-amount, time-of-day features
        ▼
  argos/models/
    ├── unsupervised.py     IsolationForest + LOF ensemble → anomaly_score
    ├── supervised.py       XGBoost + isotonic calibration → fraud_probability
    └── ensemble.py         combines both (+ rule boosts) → suspicion_score
        │
        ├──► argos/alerts/notifier.py     console / email / webhook
        └──► argos/reporting/             scored CSV + HTML report
        │
        ▼
  cli.py  (terminal)   /   dashboard/app.py  (Streamlit)
```

Both the CLI and dashboard call the same `FraudDetectionPipeline`
(`argos/pipeline.py`), so they can't drift out of sync.

## Scoring modes — read this before trusting a number

The pipeline runs in one of two modes, and **the output column name tells
you which one you're looking at**:

| Mode | Trigger | Output column | What it actually means |
|---|---|---|---|
| `unsupervised` | No trained model at `models/supervised_model.pkl` (default state) | `anomaly_score` | Statistically unusual *relative to this batch*. Not a fraud probability. |
| `supervised` | A trained, schema-matching model is present | `fraud_probability` | A calibrated probability, from a model trained on labeled fraud data. |

**Why "accuracy" is not reported anywhere in this project:** fraud is a
heavily imbalanced problem (often <1% positive class). A model that predicts
"not fraud" for everything scores 99%+ accuracy while catching zero fraud.
Instead, `scripts/train_supervised.py` reports **precision, recall, AUC-ROC,
and AUC-PR** on a held-out test split — see `TrainingMetrics` in
`argos/models/supervised.py`.

**Why the supervised path won't just work on your own uploaded statement
out of the box:** public labeled fraud datasets (Kaggle Credit Card Fraud,
PaySim) are in their own feature space — PCA components or raw numeric
transaction fields. They don't contain `description` text, payee history, or
velocity features the way `argos/features/engineer.py` produces for a real
statement. So `EnsembleScorer` will refuse to apply a mismatched model —
it raises a clear `ValueError` naming the missing features rather than
silently producing garbage scores (see `test_ensemble.py` and
`ensemble.py`). Training `scripts/train_supervised.py` on those datasets is
still worth doing — it demonstrates a properly calibrated, honestly evaluated
supervised classifier — but closing the loop on arbitrary real statements
would need labeled fraud data in the *same* engineered feature space, which
in practice means proprietary/internal data, not a public benchmark.

## Quickstart

```bash
git clone <your-repo-url>
cd project-argos
pip install -r requirements.txt

# Generate synthetic demo data (no real bank data needed to try this out)
python scripts/generate_sample_data.py --out data/sample_statement.csv

# CLI
python cli.py analyze data/sample_statement.csv

# Dashboard
python cli.py dashboard
# or: streamlit run dashboard/app.py
```

On the demo data, the pipeline (unsupervised mode) correctly ranks 8/9
flagged transactions as the deliberately injected anomalies (large
first-time payees, round suspicious amounts, off-hours wire transfers), with
one plausible false positive on a normal but atypically large food-delivery
order — a realistic outcome for anomaly detection, and exactly the kind of
result that should go through human review rather than an automated block.
A rapid-fire small-transfer velocity pattern was also injected and scores
noticeably higher than normal traffic, but stayed under the default 0.5 flag
threshold in this run — a good illustration of the precision/recall
trade-off `thresholds.flag_score` in `config.yaml` controls.

## Training the supervised model

```bash
# Download a labeled dataset first, e.g.:
#   Kaggle Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#   PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1

python cli.py train path/to/creditcard.csv --label-col Class
```

This prints precision/recall/AUC-ROC/AUC-PR on a held-out split, saves the
calibrated model to `models/supervised_model.pkl`, and saves metrics to
`models/supervised_model.metrics.json`.

## Ingestion notes

- **CSV/Excel is the reliable path.** Column names are auto-detected via
  regex against common bank export headers (date/narration/debit/credit/
  balance in various phrasings). If detection fails, the error names exactly
  which required field it couldn't find and lists your actual columns.
- **PDF is best-effort.** Tries `camelot` table extraction first (if
  installed — it needs Ghostscript, so it's optional, see
  `requirements.txt`), then falls back to a whitespace-based text-line
  heuristic. Always check `ParseReport.summary()` / the dashboard's parse
  report panel before trusting PDF-derived results — bank PDF layouts vary
  enough that silent misparsing is a real risk.

## Project structure

```
argos/                  core package (pip installable — pyproject.toml)
  ingestion/             csv_loader.py, pdf_loader.py, validate.py
  features/               engineer.py
  models/                 unsupervised.py, supervised.py, ensemble.py
  alerts/                 notifier.py
  reporting/              report_generator.py, templates/report.html
  pipeline.py             orchestrates the above
  config.py               loads config.yaml
cli.py                   analyze / train / dashboard subcommands
dashboard/app.py         Streamlit UI
scripts/
  generate_sample_data.py
  train_supervised.py
tests/                   pytest — ingestion, features, ensemble
config.yaml              thresholds, weights, alert channel, all in one place
```

## Roadmap / honest limitations

- No feedback loop yet — a "mark as false positive / confirmed fraud" action
  in the dashboard that retrains the model over time is the highest-value
  next addition.
- PDF ingestion remains inherently fragile; for production use, prefer
  direct data feeds (e.g. India's Account Aggregator framework — Setu,
  Finvu, OneMoney — over parsing PDFs at all).
- Batch-relative unsupervised scoring means the same transaction can score
  differently depending on what else is in the statement it's evaluated
  with. This is expected behavior, not a bug, but worth knowing.
- Real-world fraud tactics adapt to detection systems over time; any model
  here will need periodic retraining, not a one-time train-and-forget.

## License

MIT — see LICENSE.
