# Credit Default Risk Prediction

![CI](https://github.com/Swayam13-exe/Credit-Risk-Prediction-System/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)
![Model](https://img.shields.io/badge/model-XGBoost-orange.svg)

**Live demo:** [swayam-credit-risk-prediction-system.streamlit.app](https://swayam-credit-risk-prediction-system.streamlit.app) · **API docs (when running locally):** `http://localhost:8000/docs`

An end-to-end machine learning system for predicting consumer credit card
default risk — covering data cleaning, feature engineering, model
comparison, class-imbalance handling, explainability, a served API,
containerization, and post-deployment drift monitoring.

Every design decision below (imbalance handling, metric choice,
explainability, monitoring) mirrors what is actually required of a credit
scoring model in a regulated fintech setting.

## Table of contents

- [Live demo](#live-demo)
- [Results at a glance](#results-at-a-glance)
- [Dataset](#dataset)
- [Explainability](#explainability)
- [Architecture](#architecture)
- [Repo structure](#repo-structure)
- [Running it](#running-it)
- [Monitoring](#monitoring)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Next steps](#next-steps-roadmap)
- [Author](#author)
- [License](#license)

## Live demo

Try it yourself — fill in a client profile and get a live prediction with
a SHAP explanation of why:

**[swayam-credit-risk-prediction-system.streamlit.app](https://swayam-credit-risk-prediction-system.streamlit.app)**

*(Hosted on Streamlit Community Cloud's free tier — the app may take a
few seconds to wake up if it hasn't been visited recently.)*

The interactive demo (`demo/streamlit_app.py`) is a UI wrapper around the
same trained pipeline served by the FastAPI app below — same model, same
features, same predictions.

## Results at a glance

Trained and evaluated on a held-out 20% test split (6,000 clients never
seen during training or model selection):

| Model | CV ROC-AUC | CV PR-AUC | CV F1 |
|---|---|---|---|
| Logistic Regression (baseline) | 0.761 | 0.509 | 0.526 |
| Random Forest | 0.788 | 0.561 | 0.544 |
| **XGBoost (selected)** | 0.786 | **0.563** | 0.540 |

**Held-out test set (XGBoost, final):**
- ROC-AUC: **0.779**
- PR-AUC (average precision): **0.558**
- F1 at tuned threshold (0.576): **0.547**
- Confusion matrix: 761/1,327 actual defaulters caught (57.4% recall), 693 false positives

XGBoost was selected over Random Forest despite a marginally lower CV
ROC-AUC because it scored higher on **PR-AUC**, which is the more
informative metric here: the dataset's ~22% default rate means ROC-AUC
can look deceptively good while precision on the minority (default) class
stays weak. XGBoost also trains ~12x faster than the Random Forest here
(7.6s vs 95.9s), which matters for retraining cadence in production.

These numbers are in line with published benchmarks on this dataset —
comparable academic work using ensembles + SHAP on this same data reports
ROC-AUC in the 0.73–0.79 range and identifies repayment-status history as
the dominant predictor, which matches what we find below.

## Dataset

**UCI "Default of Credit Card Clients" dataset** (Yeh, I. C., & Lien, C. H.,
2009, *Expert Systems with Applications*) — 30,000 credit card clients in
Taiwan, April–September 2005, 23 raw features, binary target (defaulted on
next month's payment or not).

Chosen deliberately over larger alternatives (e.g. Home Credit Default
Risk's 300K-row, 7-table relational schema) because it's a single,
clean-ish table that still requires real data-quality handling and
feature engineering to do well — better suited to demonstrating engineering
judgment than to hiding it behind a huge join pipeline.

### Data quality issues found and corrected (`src/features.py::clean`)
- `EDUCATION` contains undocumented categories `0, 5, 6` outside the
  stated 1–4 range → folded into an "other/unknown" bucket.
- `MARRIAGE` contains an undocumented category `0` outside the stated
  1–3 range → folded into "other".
- Both issues are consistent with prior academic reproductions of this
  dataset and are corrected rather than silently ignored.

### Engineered features (`src/features.py::engineer_features`)
On top of the 23 raw fields, we add:
- **Repayment trend features**: `max_delay`, `mean_delay`,
  `num_months_delayed`, `delay_trend` (is delinquency worsening recently?)
- **Credit utilization ratios**: `util_ratio_1..6`, `avg_utilization`,
  `max_utilization` (bill amount ÷ credit limit)
- **Payment-to-bill ratios**: `pay_ratio_1..6`, `avg_pay_ratio` (how much
  of what was billed did they actually pay?)
- **Spending trend**: `bill_trend`, `avg_bill_amt`, `avg_pay_amt`

SHAP analysis (below) confirms these engineered features — not just the
raw fields — carry real, additional predictive signal.

## Explainability

Credit models can't be black boxes — regulators and lenders require
reason codes for adverse decisions. We use SHAP (TreeExplainer) for both
global and per-client explanations.

**Top features by mean |SHAP value|:**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `max_delay` (engineered) | 0.450 |
| 2 | `PAY_0` (most recent repayment status) | 0.247 |
| 3 | `num_months_delayed` (engineered) | 0.128 |
| 4 | `util_ratio_2` (engineered) | 0.123 |
| 5 | `bill_trend` (engineered) | 0.102 |
| 6 | `PAY_AMT1` | 0.099 |
| 7 | `LIMIT_BAL` | 0.089 |

See `reports/figures/shap_summary.png` for the full global summary plot
and `reports/figures/shap_waterfall.png` for a worked example explaining
one individual high-risk prediction.

## Architecture

```
data/raw/UCI_Credit_Card.csv
        │
        ▼
src/features.py  (clean + engineer)
        │
        ▼
src/train.py  ── trains & compares LR / RF / XGBoost
        │         (5-fold CV, class-weighted / scale_pos_weight)
        │         picks best by PR-AUC, tunes decision threshold by F1
        ▼
models/model.joblib, threshold.json, feature_columns.json
        │
        ▼
app/main.py  (FastAPI)  ──►  POST /predict  ──►  probability + risk tier
        │
        ▼
Dockerfile  ──►  containerized service, /health check
        │
        ▼
monitoring/drift_check.py  ──►  PSI-based feature drift report
```

## Repo structure

```
credit-risk-project/
├── data/raw/UCI_Credit_Card.csv     # source data
├── src/
│   ├── features.py                  # cleaning + feature engineering
│   ├── train.py                     # model comparison, training, eval
│   └── explain.py                   # SHAP analysis
├── app/
│   ├── main.py                      # FastAPI service
│   └── schemas.py                   # request/response models
├── demo/
│   └── streamlit_app.py             # interactive demo (live link above)
├── models/                          # trained pipeline + metadata
├── monitoring/drift_check.py        # PSI drift monitoring
├── reports/                         # metrics.json, figures/
├── tests/                           # pytest suite (10 tests)
├── Dockerfile
├── requirements.txt
├── packages.txt                     # system deps for Streamlit Cloud
└── .github/workflows/ci.yml         # test + build on every push
```

## Running it

```bash
pip install -r requirements.txt

# Train (re-runs cleaning, feature engineering, model comparison, saves artifacts)
python src/train.py

# Generate SHAP explainability plots
python src/explain.py

# Run tests
pytest tests/ -v

# Serve the API locally
uvicorn app.main:app --reload

# Or run the interactive Streamlit demo locally
streamlit run demo/streamlit_app.py

# Or via Docker
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 46000,
    "BILL_AMT4": 44000, "BILL_AMT5": 42000, "BILL_AMT6": 40000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000
  }'
```

```json
{
  "default_probability": 0.8714,
  "default_prediction": 1,
  "risk_tier": "very_high",
  "threshold_used": 0.5758,
  "model_version": "credit-risk-xgb-v1"
}
```

## Monitoring

`monitoring/drift_check.py` computes the Population Stability Index (PSI)
per feature against the training reference distribution — the standard
metric banks use for ongoing model-risk governance (SR 11-7 / IFRS 9).
Run with `python monitoring/drift_check.py --incoming path/to/batch.csv`,
or with no arguments to see a self-contained demo against a synthetically
shifted distribution.

## Design decisions and trade-offs

- **Class imbalance**: handled via `class_weight="balanced"` /
  `scale_pos_weight` rather than SMOTE. This avoids synthetic-sample
  leakage across CV folds and keeps the pipeline simpler to reason about
  and audit — an important property for a credit model.
- **Model selection metric**: PR-AUC, not accuracy or plain ROC-AUC,
  because the positive class (default) is the one that actually matters
  and is the minority class.
- **Threshold**: tuned to maximize F1 on the precision-recall curve; in
  a real deployment this would instead be set from a cost matrix (cost of
  a missed default vs. cost of an unnecessary rejection), which is a
  business decision, not a modeling one — the code is written so that
  swapping in a cost-based threshold is a one-line change.

## Next steps (roadmap)

- Model calibration (Platt scaling / isotonic regression) so predicted
  probabilities can be used directly for expected-loss calculations.
- Fairness audit across `SEX`, `AGE`, `EDUCATION` subgroups.
- Swap in the larger Home Credit Default Risk dataset (relational,
  multi-table) as a stretch extension once this pipeline is solid.

## Author

**Swayam** — Computer Engineering student, AI/ML Engineering focus
[GitHub](https://github.com/Swayam13-exe) · [LinkedIn](https://linkedin.com/in/your-linkedin-handle)

## License

This project is licensed under the [MIT License](LICENSE) — free to use,
modify, and learn from.