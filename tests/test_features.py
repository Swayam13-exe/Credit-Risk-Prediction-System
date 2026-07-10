import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import clean, engineer_features  # noqa: E402


def make_raw_row(**overrides):
    base = {
        "LIMIT_BAL": 100000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 10000, "BILL_AMT2": 10000, "BILL_AMT3": 10000,
        "BILL_AMT4": 10000, "BILL_AMT5": 10000, "BILL_AMT6": 10000,
        "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
        "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_clean_folds_undocumented_education_categories():
    df = make_raw_row(EDUCATION=0)
    cleaned = clean(df)
    assert cleaned["EDUCATION"].iloc[0] == 4

    df = make_raw_row(EDUCATION=5)
    assert clean(df)["EDUCATION"].iloc[0] == 4

    df = make_raw_row(EDUCATION=6)
    assert clean(df)["EDUCATION"].iloc[0] == 4


def test_clean_folds_undocumented_marriage_category():
    df = make_raw_row(MARRIAGE=0)
    cleaned = clean(df)
    assert cleaned["MARRIAGE"].iloc[0] == 3


def test_engineer_features_handles_zero_limit_bal_without_error():
    df = clean(make_raw_row(LIMIT_BAL=0))
    result = engineer_features(df)
    assert not result.isnull().any().any()


def test_engineer_features_adds_expected_columns():
    df = clean(make_raw_row())
    result = engineer_features(df)
    for col in ["max_delay", "mean_delay", "num_months_delayed",
                "avg_utilization", "avg_pay_ratio", "bill_trend"]:
        assert col in result.columns


def test_max_delay_reflects_worst_repayment_month():
    df = clean(make_raw_row(PAY_0=0, PAY_2=3, PAY_3=1, PAY_4=0, PAY_5=0, PAY_6=0))
    result = engineer_features(df)
    assert result["max_delay"].iloc[0] == 3


def test_num_months_delayed_counts_positive_pay_values():
    df = clean(make_raw_row(PAY_0=1, PAY_2=2, PAY_3=-1, PAY_4=0, PAY_5=0, PAY_6=0))
    result = engineer_features(df)
    assert result["num_months_delayed"].iloc[0] == 2
