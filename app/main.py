import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import clean, engineer_features  # noqa: E402

try:
    from app.schemas import ClientFeatures, PredictionResponse  # noqa: E402
except ImportError:
    from schemas import ClientFeatures, PredictionResponse  # noqa: E402

MODELS_DIR = ROOT / "models"
MODEL_VERSION = "credit-risk-xgb-v1"

_model = None
_threshold = 0.5
_feature_columns = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _threshold, _feature_columns
    _model = joblib.load(MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "threshold.json") as f:
        _threshold = json.load(f)["threshold"]
    with open(MODELS_DIR / "feature_columns.json") as f:
        _feature_columns = json.load(f)["columns"]
    yield


app = FastAPI(
    title="Credit Default Risk API",
    description="Serves default-probability predictions for credit card clients, "
                 "trained on the UCI Default of Credit Card Clients dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


def risk_tier(prob: float) -> str:
    if prob < 0.2:
        return "low"
    if prob < 0.5:
        return "medium"
    if prob < 0.75:
        return "high"
    return "very_high"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_df = pd.DataFrame([client.model_dump()])
    df = clean(raw_df)
    df = engineer_features(df)

    missing = [c for c in _feature_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing engineered columns: {missing}")

    X = df[_feature_columns]
    prob = float(_model.predict_proba(X)[:, 1][0])
    pred = int(prob >= _threshold)

    return PredictionResponse(
        default_probability=round(prob, 4),
        default_prediction=pred,
        risk_tier=risk_tier(prob),
        threshold_used=_threshold,
        model_version=MODEL_VERSION,
    )
