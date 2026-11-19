import os
import joblib
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Pydantic Schemas
class HeartDiseasePredictionRequest(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = Female, 1 = Male")
    cp: Literal[0, 1, 2, 3] = Field(..., description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=50, le=600, description="Serum cholestoral (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: Literal[0, 1, 2] = Field(..., description="Slope of peak exercise ST segment (0-2)")
    ca: Literal[0, 1, 2, 3, 4] = Field(..., description="Number of major vessels (0-4)")
    thal: Literal[0, 1, 2, 3] = Field(..., description="Thal (0-3)")

class HeartDiseasePredictionResponse(BaseModel):
    has_heart_disease: bool
    diagnosis: str
    risk_probability: float
    status_indicator: str

# FastAPI App
app = FastAPI(
    title="Med-Buddy Heart Disease Risk Service",
    description="FastAPI service for evaluating cardiology clinical risk probabilities.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "heart_disease_model.joblib")
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
    return {"status": "healthy", "service": "heart_disease"}

@app.post("/predict", response_model=HeartDiseasePredictionResponse)
def predict(req: HeartDiseasePredictionRequest):
    try:
        clf = load_model()
        data = {
            "age": req.age,
            "sex": req.sex,
            "cp": req.cp,
            "trestbps": req.trestbps,
            "chol": req.chol,
            "fbs": req.fbs,
            "restecg": req.restecg,
            "thalach": req.thalach,
            "exang": req.exang,
            "oldpeak": req.oldpeak,
            "slope": req.slope,
            "ca": req.ca,
            "thal": req.thal,
        }
        df = pd.DataFrame([data])
        pred = int(clf.predict(df)[0])
        prob = float(clf.predict_proba(df)[0][1])

        has_heart_disease = bool(pred == 1 or prob >= 0.5)
        diagnosis = (
            "High risk of heart disease 🔴"
            if has_heart_disease
            else "Low risk of heart disease 🟢"
        )
        indicator = "Critical Attention Recommended" if has_heart_disease else "Normal / Stable"

        return HeartDiseasePredictionResponse(
            has_heart_disease=has_heart_disease,
            diagnosis=diagnosis,
            risk_probability=round(prob, 4),
            status_indicator=indicator
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
