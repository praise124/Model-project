# California House Price Prediction

A machine learning application for California real estate pricing using **FastAPI** and **Streamlit**.

## Project Architecture
- `backend/main.py`: FastAPI server running on port `8003`.
- `frontend/app.py`: Streamlit real estate valuation dashboard running on port `8503`.
- `model/house_price_model.joblib`: ColumnTransformer + Random Forest Regressor Pipeline.

## How to Run

```bash
# Terminal 1 - Start Backend API
python -m uvicorn 3_house_price_prediction.backend.main:app --port 8003 --reload

# Terminal 2 - Start Streamlit Frontend
streamlit run 3_house_price_prediction/frontend/app.py --server.port 8503
```

- **Frontend UI**: [http://localhost:8503](http://localhost:8503)
- **API Swagger Docs**: [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs)
