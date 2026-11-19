import os
import joblib
import pandas as pd
from typing import Dict, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Pydantic Schemas
class CreditCardPredictionRequest(BaseModel):
    Time: float = Field(..., ge=0, description="Seconds elapsed since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in currency units")
    pca_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dictionary of V1 to V28 PCA features. Defaults to 0.0 if not provided."
    )

class CreditCardPredictionResponse(BaseModel):
    is_fraud: bool
    prediction_label: str
    fraud_probability: float
    risk_level: str

# FastAPI App
app = FastAPI(
    title="Credit Card Fraud Detection Service",
    description="FastAPI service for evaluating credit card transaction fraud probability.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "credit_card_model.joblib")
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "credit_card_fraud"}

@app.post("/predict", response_model=CreditCardPredictionResponse)
def predict(req: CreditCardPredictionRequest):
    try:
        clf = load_model()
        features = {"Time": req.Time}
        provided = req.pca_features or {}
        for i in range(1, 29):
            col = f"V{i}"
            features[col] = float(provided.get(col, 0.0))
        features["Amount"] = req.Amount

        df = pd.DataFrame([features])
        prob = float(clf.predict_proba(df)[0][1])
        pred = int(clf.predict(df)[0])

        is_fraud = bool(pred == 1 or prob >= 0.5)
        risk = "HIGH RISK" if is_fraud else ("MEDIUM" if prob >= 0.25 else "LOW RISK")
        label = "Fraudulent Transaction" if is_fraud else "Legitimate Transaction"

        return CreditCardPredictionResponse(
            is_fraud=is_fraud,
            prediction_label=label,
            fraud_probability=round(prob, 4),
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
