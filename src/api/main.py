"""FastAPI REST endpoint for real-time fraud detection."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import logging
import joblib
import numpy as np

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="XGBoost fraud classifier with real-time predictions and monitoring",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logger = logging.getLogger(__name__)

# Load model on startup
model = None


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("models/fraud_model.pkl")
        logger.info("Fraud detection model loaded successfully")
    except FileNotFoundError:
        logger.warning("Model not found, will use mock predictions")


class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant_category: str
    device_type: str
    hour_of_day: int
    day_of_week: int
    distance_from_home_km: float
    transaction_count_24h: int
    avg_amount_7d: float


class FraudPrediction(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: str  # LOW / MEDIUM / HIGH / CRITICAL
    latency_ms: float


@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionRequest):
    """Real-time fraud prediction endpoint."""
    start = time.time()

    features = np.array([[
        transaction.amount,
        transaction.hour_of_day,
        transaction.day_of_week,
        transaction.distance_from_home_km,
        transaction.transaction_count_24h,
        transaction.avg_amount_7d,
    ]])

    if model is not None:
        fraud_prob = float(model.predict_proba(features)[0][1])
    else:
        # Mock prediction for demo
        fraud_prob = min(1.0, transaction.amount / 10000 * 0.5)

    is_fraud = fraud_prob > 0.5

    if fraud_prob < 0.3:
        risk_score = "LOW"
    elif fraud_prob < 0.5:
        risk_score = "MEDIUM"
    elif fraud_prob < 0.8:
        risk_score = "HIGH"
    else:
        risk_score = "CRITICAL"

    latency_ms = (time.time() - start) * 1000
    logger.info(f"Prediction: {transaction.transaction_id} | fraud={is_fraud} | prob={fraud_prob:.3f} | {latency_ms:.1f}ms")

    return FraudPrediction(
        transaction_id=transaction.transaction_id,
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        risk_score=risk_score,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
async def metrics():
    return {"model_version": "1.0.0", "framework": "XGBoost"}
