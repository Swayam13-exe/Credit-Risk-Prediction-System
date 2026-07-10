"""
Feature engineering for the credit default prediction pipeline.

Dataset: UCI "Default of Credit Card Clients" (Yeh & Lien, 2009), Taiwan,
April-September 2005. 30,000 clients, 23 raw features, binary target.

Known data quality issues (documented in prior academic reproductions of
this dataset, e.g. MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys)
that this module explicitly corrects:
  - EDUCATION contains undocumented categories 0, 5, 6 (outside the stated
    1=grad school, 2=university, 3=high school, 4=others). We fold these
    into a single "other/unknown" bucket (4).
  - MARRIAGE contains an undocumented category 0 (outside 1=married,
    2=single, 3=other). We fold it into "other" (3).
  - PAY_0 is actually the September 2005 repayment status; the naming skips
    from PAY_0 straight to PAY_2 (there is no PAY_1). We keep the original
    column names for traceability but document this explicitly.
"""

import numpy as np
import pandas as pd

RAW_TARGET_COL = "default.payment.next.month"
TARGET_COL = "default"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAYAMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={RAW_TARGET_COL: TARGET_COL})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fold undocumented EDUCATION categories (0, 5, 6) into "other" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # Fold undocumented MARRIAGE category (0) into "other" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-informed features on top of the raw/cleaned columns.
    These are the features that typically carry the most signal in
    published analyses of this dataset (repayment-status trend and
    utilization ratios dominate SHAP importance in our experiments too;
    see reports/figures/shap_summary.png).
    """
    df = df.copy()

    # --- Repayment-status derived features ---
    # PAY_x: -2/-1/0 roughly mean "paid duly / no consumption", >=1 means
    # months delinquent. We summarize the 6-month repayment trend.
    df["max_delay"] = df[PAY_COLS].max(axis=1)
    df["mean_delay"] = df[PAY_COLS].mean(axis=1)
    df["num_months_delayed"] = (df[PAY_COLS] > 0).sum(axis=1)
    # Trend: is the client's delinquency getting worse in recent months?
    df["delay_trend"] = df["PAY_0"] - df[["PAY_5", "PAY_6"]].mean(axis=1)

    # --- Credit utilization ---
    limit_safe = df["LIMIT_BAL"].replace(0, np.nan)
    for i, col in enumerate(BILL_COLS, start=1):
        df[f"util_ratio_{i}"] = (df[col] / limit_safe).clip(-2, 5)
    df["avg_utilization"] = df[[f"util_ratio_{i}" for i in range(1, 7)]].mean(axis=1)
    df["max_utilization"] = df[[f"util_ratio_{i}" for i in range(1, 7)]].max(axis=1)

    # --- Payment behavior relative to what was billed ---
    for i in range(1, 7):
        bill = df[BILL_COLS[i - 1]].replace(0, np.nan)
        df[f"pay_ratio_{i}"] = (df[PAYAMT_COLS[i - 1]] / bill).clip(0, 5).fillna(0)
    df["avg_pay_ratio"] = df[[f"pay_ratio_{i}" for i in range(1, 7)]].mean(axis=1)

    # --- Bill amount trend (is spending/balance rising or falling) ---
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)
    df["avg_pay_amt"] = df[PAYAMT_COLS].mean(axis=1)

    df = df.fillna(0)
    return df


def build_dataset(raw_path: str) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)
    return df


CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != TARGET_COL]
