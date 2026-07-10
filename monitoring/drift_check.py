"""
Simple data drift monitor using Population Stability Index (PSI).

PSI is a standard, widely used metric in credit-risk model monitoring
(banks use it to satisfy model-governance requirements like SR 11-7 /
IFRS 9 ongoing monitoring). Rule of thumb thresholds used industry-wide:
    PSI < 0.1  -> no significant shift
    0.1 - 0.25 -> moderate shift, investigate
    > 0.25     -> major shift, retrain/recalibrate

Usage:
    python monitoring/drift_check.py --incoming path/to/new_batch.csv

If --incoming is omitted, this runs a self-test by synthetically shifting
the reference distribution so the script's output is demonstrable without
requiring a live production feed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "reports" / "reference_distribution.csv"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.linspace(0, 100, bins + 1)
    cuts = np.unique(np.percentile(reference, breakpoints))
    if len(cuts) < 3:
        return 0.0  # not enough variation to bin meaningfully

    ref_counts, _ = np.histogram(reference, bins=cuts)
    cur_counts, _ = np.histogram(current, bins=cuts)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(len(reference), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(len(current), 1))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify(psi_value: float) -> str:
    if psi_value < 0.1:
        return "stable"
    if psi_value < 0.25:
        return "moderate_shift"
    return "major_shift"


def run(incoming_path: str | None):
    reference = pd.read_csv(REFERENCE_PATH)
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "default"]

    if incoming_path:
        current = pd.read_csv(incoming_path)
    else:
        # Self-test: simulate a moderate shift (e.g. utilization creeping up
        # in a recessionary period) so the report is demonstrable end-to-end.
        current = reference.copy()
        current["avg_utilization"] = current["avg_utilization"] * 1.4
        current["max_delay"] = current["max_delay"] + 0.5
        print("No --incoming file given; running self-test with a synthetic shift.\n")

    results = {}
    for col in numeric_cols:
        if col not in current.columns:
            continue
        value = psi(reference[col].values, current[col].values)
        results[col] = {"psi": round(value, 4), "status": classify(value)}

    flagged = {k: v for k, v in results.items() if v["status"] != "stable"}

    print(json.dumps(results, indent=2))
    print(f"\n{len(flagged)}/{len(results)} features show moderate or major drift.")
    if flagged:
        print("Flagged features:", list(flagged.keys()))

    out_path = ROOT / "reports" / "drift_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=str, default=None,
                         help="Path to a CSV of new incoming data (same schema as training data)")
    args = parser.parse_args()
    run(args.incoming)
