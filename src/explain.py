"""
Generate SHAP explainability artifacts for the trained model.

Produces:
    reports/figures/shap_summary.png   - global feature importance / effect direction
    reports/figures/shap_waterfall.png - explanation for one individual prediction
    reports/shap_top_features.json     - top features by mean |SHAP value|
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from features import CATEGORICAL_COLS, TARGET_COL, build_dataset, get_feature_columns

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42


def main():
    df = build_dataset(str(RAW_PATH))
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = joblib.load(MODELS_DIR / "model.joblib")
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Transform a sample of the test set for SHAP (keep it fast)
    sample = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
    X_transformed = pre.transform(sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # some sklearn/xgboost versions return a list
        shap_values = shap_values[1]

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Individual explanation (waterfall) for one defaulted-and-predicted-risky client
    probs = clf.predict_proba(X_transformed)[:, 1]
    idx = int(np.argmax(probs))
    exp = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_transformed[idx],
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_waterfall.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Top features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:15]
    with open(ROOT / "reports" / "shap_top_features.json", "w") as f:
        json.dump([{"feature": f, "mean_abs_shap": float(v)} for f, v in top], f, indent=2)

    print("Top features by mean |SHAP value|:")
    for f, v in top:
        print(f"  {f}: {v:.4f}")
    print("\nSaved SHAP plots to reports/figures/")


if __name__ == "__main__":
    main()
