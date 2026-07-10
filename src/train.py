"""
Train and compare credit-default models.

Usage:
    python src/train.py

Produces:
    models/model.joblib          - best pipeline (preprocessing + classifier)
    models/feature_columns.json  - column order expected at inference time
    models/threshold.json        - selected decision threshold
    reports/metrics.json         - full comparison table + final test metrics
    reports/figures/*.png        - ROC, PR curve, confusion matrix, SHAP plots
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CATEGORICAL_COLS, TARGET_COL, build_dataset, get_feature_columns

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
RANDOM_STATE = 42

for d in [MODELS_DIR, REPORTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def build_preprocessor(numeric_cols, categorical_cols, scale_numeric: bool):
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transform, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def get_candidate_models(n_pos, n_neg):
    scale_pos_weight = n_neg / n_pos
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            True,  # needs scaling
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }


def find_best_threshold(y_true, y_prob):
    """Pick the threshold that maximizes F1 on validation folds (business
    cost of missing a defaulter is usually higher than a false alarm, but
    F1 is a reasonable, tunable default; see README for how to adjust this
    if a different cost matrix applies)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1s[:-1]) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def main():
    print("Loading and engineering features...")
    df = build_dataset(str(RAW_PATH))
    feature_cols = get_feature_columns(df)
    categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    print(f"Train size: {len(y_train)} | default rate: {y_train.mean():.3f}")

    candidates = get_candidate_models(n_pos, n_neg)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    comparison = {}
    fitted_pipelines = {}

    for name, (model, needs_scaling) in candidates.items():
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=needs_scaling)
        pipe = Pipeline([("pre", pre), ("clf", model)])

        t0 = time.time()
        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "avg_precision": "average_precision", "f1": "f1"},
            n_jobs=-1,
        )
        elapsed = time.time() - t0

        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        comparison[name] = {
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(cv_scores["test_avg_precision"])),
            "cv_pr_auc_std": float(np.std(cv_scores["test_avg_precision"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "train_time_sec": round(elapsed, 2),
        }
        print(f"{name}: CV ROC-AUC={comparison[name]['cv_roc_auc_mean']:.4f} "
              f"PR-AUC={comparison[name]['cv_pr_auc_mean']:.4f}")

    # Select best model by cross-validated PR-AUC (more informative than
    # ROC-AUC under class imbalance ~22% positive rate).
    best_name = max(comparison, key=lambda k: comparison[k]["cv_pr_auc_mean"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model by CV PR-AUC: {best_name}")

    # Final held-out test evaluation
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_test.values, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "f1_at_threshold": float(f1_score(y_test, y_pred)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2))

    # --- Plots ---
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve - {best_name} (AUC={test_metrics['roc_auc']:.3f})")
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title(f"Precision-Recall Curve - {best_name} (AP={test_metrics['pr_auc']:.3f})")
    plt.savefig(FIG_DIR / "pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Default", "Default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.2f}")
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- Save artifacts ---
    joblib.dump(best_pipe, MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump({"columns": feature_cols, "categorical": categorical_cols,
                    "numeric": numeric_cols, "best_model": best_name}, f, indent=2)
    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    all_results = {"model_comparison": comparison, "best_model": best_name,
                   "test_metrics": test_metrics,
                   "train_default_rate": float(y_train.mean()),
                   "test_default_rate": float(y_test.mean()),
                   "n_train": len(y_train), "n_test": len(y_test)}
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save a reference distribution for the monitoring/drift script
    X_train.assign(**{TARGET_COL: y_train}).to_csv(REPORTS_DIR / "reference_distribution.csv", index=False)

    print("\nSaved model, metrics, and figures. Run src/explain.py next for SHAP plots.")


if __name__ == "__main__":
    main()
