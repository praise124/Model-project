# Heart Disease Risk Prediction (Med-Buddy)

Cardiovascular disease screening machine learning service utilizing **FastAPI** and **Streamlit**.

## Project Architecture
- `backend/main.py`: FastAPI server running on port `8002`.
- `frontend/app.py`: Streamlit patient intake and risk dashboard running on port `8502`.
- `model/heart_disease_model.joblib`: StandardScaler + Random Forest Pipeline.

## How to Run

```bash
# Terminal 1 - Start Backend API
python -m uvicorn 2_heart_disease.backend.main:app --port 8002 --reload

# Terminal 2 - Start Streamlit Frontend
streamlit run 2_heart_disease/frontend/app.py --server.port 8502
```

- **Frontend UI**: [http://localhost:8502](http://localhost:8502)
- **API Swagger Docs**: [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)
