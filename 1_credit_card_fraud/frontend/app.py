import streamlit as st
import requests

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.markdown("Assess transaction legitimacy via our FastAPI-backed Random Forest Classifier.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("Backend URL", value="http://127.0.0.1:8001")
    st.divider()
    preset = st.radio("Load Preset Sample:", ["Custom", "Typical Normal Transaction", "High Risk / Fraud Pattern"])

default_amount = 45.00
default_time = 14500.0
pca_values = {f"V{i}": 0.0 for i in range(1, 29)}

if preset == "Typical Normal Transaction":
    default_amount = 28.50
    default_time = 3600.0
    pca_values["V1"] = 0.12
    pca_values["V2"] = -0.05
    pca_values["V14"] = 0.25
elif preset == "High Risk / Fraud Pattern":
    default_amount = 899.99
    default_time = 72000.0
    pca_values["V14"] = -6.80
    pca_values["V4"] = 4.50
    pca_values["V12"] = -3.20
    pca_values["V10"] = -2.90

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Parameters")
    amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=default_amount, step=1.0)
    time_sec = st.number_input("Time (Seconds from first transaction)", min_value=0.0, value=default_time, step=100.0)

with col2:
    st.subheader("Key Predictive PCA Components")
    pca_values["V14"] = st.slider("V14 (High inverse sensitivity to fraud)", -10.0, 5.0, float(pca_values.get("V14", 0.0)), step=0.1)
    pca_values["V4"] = st.slider("V4 (Positive correlation to fraud)", -5.0, 10.0, float(pca_values.get("V4", 0.0)), step=0.1)
    pca_values["V12"] = st.slider("V12", -10.0, 5.0, float(pca_values.get("V12", 0.0)), step=0.1)

with st.expander("🛠️ All PCA Components (V1 to V28)"):
    grid_cols = st.columns(4)
    for idx, i in enumerate(range(1, 29)):
        col_name = f"V{i}"
        with grid_cols[idx % 4]:
            if col_name not in ["V14", "V4", "V12"]:
                pca_values[col_name] = st.number_input(
                    col_name,
                    value=float(pca_values.get(col_name, 0.0)),
                    step=0.1,
                    format="%.2f"
                )

st.divider()

if st.button("🚀 Analyze Transaction", type="primary", use_container_width=True):
    payload = {
        "Time": float(time_sec),
        "Amount": float(amount),
        "pca_features": pca_values
    }

    with st.spinner("Calling Fraud Detection API..."):
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                is_fraud = result["is_fraud"]
                prob = result["fraud_probability"]

                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    if is_fraud:
                        st.error(f"### ⚠️ {result['prediction_label']}")
                    else:
                        st.success(f"### ✅ {result['prediction_label']}")
                    st.metric(label="Calculated Risk Level", value=result["risk_level"])

                with res_c2:
                    st.metric(label="Fraud Probability", value=f"{prob * 100:.2f}%")
                    st.progress(min(max(prob, 0.0), 1.0))
            else:
                st.error(f"Backend error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend at `{api_url}`: {e}")
