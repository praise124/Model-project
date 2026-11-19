import os
import joblib
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Pydantic Schemas
class HousePricePredictionRequest(BaseModel):
    longitude: float = Field(..., ge=-125.0, le=-114.0, description="Block longitude")
    latitude: float = Field(..., ge=32.0, le=43.0, description="Block latitude")
    housing_median_age: float = Field(..., ge=1.0, le=100.0, description="Median age of house in block")
    total_rooms: float = Field(..., ge=1.0, description="Total rooms in block")
    total_bedrooms: float = Field(..., ge=1.0, description="Total bedrooms in block")
    population: float = Field(..., ge=1.0, description="Total population in block")
    households: float = Field(..., ge=1.0, description="Total households in block")
    median_income: float = Field(..., ge=0.1, le=25.0, description="Median income (in $10k)")
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"] = Field(
        ..., description="Proximity to ocean"
    )

class HousePricePredictionResponse(BaseModel):
    estimated_price: float
    formatted_price: str
    median_income_bracket: str

# FastAPI App
app = FastAPI(
    title="California House Price Prediction Service",
    description="FastAPI service for estimating California district real estate valuations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "house_price_model.joblib")
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
    return {"status": "healthy", "service": "house_price_prediction"}

@app.post("/predict", response_model=HousePricePredictionResponse)
def predict(req: HousePricePredictionRequest):
    try:
        reg = load_model()
        data = {
            "longitude": req.longitude,
            "latitude": req.latitude,
            "housing_median_age": req.housing_median_age,
            "total_rooms": req.total_rooms,
            "total_bedrooms": req.total_bedrooms,
            "population": req.population,
            "households": req.households,
            "median_income": req.median_income,
            "ocean_proximity": req.ocean_proximity,
        }
        df = pd.DataFrame([data])
        price_est = max(10000.0, float(reg.predict(df)[0]))

        bracket = (
            "Upper Income Tier"
            if req.median_income > 6.0
            else ("Middle Income Tier" if req.median_income >= 3.0 else "Modest Income Tier")
        )

        return HousePricePredictionResponse(
            estimated_price=round(price_est, 2),
            formatted_price=f"${price_est:,.2f}",
            median_income_bracket=bracket
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
