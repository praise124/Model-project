# Machine Learning Applications (3 Independent Projects)

This repository contains 3 standalone, clean machine learning applications, each equipped with its own **FastAPI backend** and **Streamlit frontend**.

---

## 📁 Repository Structure

```text
praise/
├── 1_credit_card_fraud/            # Project 1: Credit Card Fraud Detection
│   ├── backend/main.py             # FastAPI service (port 8001)
│   ├── frontend/app.py             # Streamlit UI (port 8501)
│   ├── model/credit_card_model.joblib
│   ├── credit_card_fraud_detection.ipynb
│   └── README.md
│
├── 2_heart_disease/                # Project 2: Heart Disease Risk Assessment (Med-Buddy)
│   ├── backend/main.py             # FastAPI service (port 8002)
│   ├── frontend/app.py             # Streamlit UI (port 8502)
│   ├── model/heart_disease_model.joblib
│   ├── 17_1_med_buddy_ml_notebook.ipynb
│   └── README.md
│
├── 3_house_price_prediction/       # Project 3: California House Price Prediction
│   ├── backend/main.py             # FastAPI service (port 8003)
│   ├── frontend/app.py             # Streamlit UI (port 8503)
│   ├── model/house_price_model.joblib
│   ├── house_price_prediction.ipynb
│   └── README.md
│
├── Model-project/                  # Original raw exploration notebooks
├── README.md                       # Master documentation
└── requirements.txt                # Unified dependency list
```

---

## 🚀 How to Run Any Project

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Credit Card Fraud Detection
```bash
# Terminal 1 - Backend
python -m uvicorn 1_credit_card_fraud.backend.main:app --port 8001 --reload

# Terminal 2 - Frontend
streamlit run 1_credit_card_fraud/frontend/app.py --server.port 8501
```
- **UI**: [http://localhost:8501](http://localhost:8501)
- **API Docs**: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

### 2. Heart Disease Risk Assessment
```bash
# Terminal 1 - Backend
python -m uvicorn 2_heart_disease.backend.main:app --port 8002 --reload

# Terminal 2 - Frontend
streamlit run 2_heart_disease/frontend/app.py --server.port 8502
```
- **UI**: [http://localhost:8502](http://localhost:8502)
- **API Docs**: [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)

---

### 3. California House Price Prediction
```bash
# Terminal 1 - Backend
python -m uvicorn 3_house_price_prediction.backend.main:app --port 8003 --reload

# Terminal 2 - Frontend
streamlit run 3_house_price_prediction/frontend/app.py --server.port 8503
```
- **UI**: [http://localhost:8503](http://localhost:8503)
- **API Docs**: [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs)
