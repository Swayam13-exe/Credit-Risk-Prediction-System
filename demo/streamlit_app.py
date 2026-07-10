"""
Interactive demo for the Credit Default Risk model.

Run locally with:  streamlit run demo/streamlit_app.py
Deployed on Hugging Face Spaces (see demo/README.md for the Space config).
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import clean, engineer_features  # noqa: E402

st.set_page_config(page_title="Credit Default Risk Predictor", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "model.joblib")
    import json
    with open(ROOT / "models" / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    with open(ROOT / "models" / "feature_columns.json") as f:
        feature_columns = json.load(f)["columns"]
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    return model, threshold, feature_columns, explainer


model, threshold, feature_columns, explainer = load_artifacts()

st.title("💳 Credit Default Risk Predictor")
st.markdown(
    "Predicts the probability a credit card client will default on their next "
    "payment, using an XGBoost model trained on the "
    "[UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350) "
    "(Yeh & Lien, 2009). Held-out test performance: **ROC-AUC 0.78, PR-AUC 0.56**. "
    "[View the full project on GitHub](https://github.com/Swayam13-exe/Credit-Risk-Prediction-System)."
)

st.divider()
st.subheader("Client profile")

col1, col2 = st.columns(2)
with col1:
    limit_bal = st.number_input("Credit limit (NT$)", min_value=10000, max_value=1000000, value=200000, step=10000)
    age = st.slider("Age", 18, 80, 35)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox(
        "Education", options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[x],
    )
    marriage = st.selectbox(
        "Marital status", options=[1, 2, 3],
        format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x],
    )

with col2:
    st.caption("Repayment status, last 6 months (-1 = paid duly, 0 = revolving, 1+ = months delinquent)")
    pay_0 = st.slider("Most recent month (PAY_0)", -2, 8, 0)
    pay_2 = st.slider("1 month ago (PAY_2)", -2, 8, 0)
    pay_3 = st.slider("2 months ago (PAY_3)", -2, 8, 0)
    pay_4 = st.slider("3 months ago (PAY_4)", -2, 8, 0)
    pay_5 = st.slider("4 months ago (PAY_5)", -2, 8, 0)
    pay_6 = st.slider("5 months ago (PAY_6)", -2, 8, 0)

st.caption("Bill amounts and payments, last 6 months (NT$)")
bill_cols = st.columns(6)
bill_amts = []
for i, c in enumerate(bill_cols, start=1):
    bill_amts.append(c.number_input(f"Bill {i}", min_value=0, value=40000, step=1000, key=f"bill{i}"))

pay_cols = st.columns(6)
pay_amts = []
for i, c in enumerate(pay_cols, start=1):
    pay_amts.append(c.number_input(f"Paid {i}", min_value=0, value=2000, step=500, key=f"pay{i}"))

if st.button("Predict default risk", type="primary", use_container_width=True):
    raw = {
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
        "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
        "BILL_AMT1": bill_amts[0], "BILL_AMT2": bill_amts[1], "BILL_AMT3": bill_amts[2],
        "BILL_AMT4": bill_amts[3], "BILL_AMT5": bill_amts[4], "BILL_AMT6": bill_amts[5],
        "PAY_AMT1": pay_amts[0], "PAY_AMT2": pay_amts[1], "PAY_AMT3": pay_amts[2],
        "PAY_AMT4": pay_amts[3], "PAY_AMT5": pay_amts[4], "PAY_AMT6": pay_amts[5],
    }
    df = pd.DataFrame([raw])
    df = clean(df)
    df = engineer_features(df)
    X = df[feature_columns]

    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= threshold)
    tier = "low" if prob < 0.2 else "medium" if prob < 0.5 else "high" if prob < 0.75 else "very_high"

    st.divider()
    st.subheader("Result")

    tier_colors = {"low": "green", "medium": "orange", "high": "red", "very_high": "red"}
    c1, c2 = st.columns(2)
    c1.metric("Default probability", f"{prob:.1%}")
    c2.metric("Risk tier", tier.replace("_", " ").upper())

    if pred == 1:
        st.error(f"⚠️ Predicted to **default** (threshold: {threshold:.1%})")
    else:
        st.success(f"✅ Predicted to **not default** (threshold: {threshold:.1%})")

    st.divider()
    st.subheader("Why this prediction? (SHAP explanation)")

    pre = model.named_steps["pre"]
    X_transformed = pre.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = pre.get_feature_names_out()

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_transformed[0],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(exp, show=False, max_display=10)
    st.pyplot(fig)
    st.caption(
        "Red bars push the prediction toward default; blue bars push toward no default. "
        "Feature names prefixed `num__`/`cat__` reflect the preprocessing pipeline's internal naming."
    )

st.divider()
st.caption(
    "Portfolio project by Swayam · [GitHub repo](https://github.com/Swayam13-exe/Credit-Risk-Prediction-System) "
    "· Not financial advice, trained on a public academic dataset for demonstration purposes."
)
