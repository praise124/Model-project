import streamlit as st
import requests

st.set_page_config(
    page_title="Med-Buddy | Heart Disease Risk",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Med-Buddy: Heart Disease Risk Assessment")
st.markdown("Clinical cardiology screening interface backed by a machine learning FastAPI service.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("Backend URL", value="http://127.0.0.1:8002")
    st.divider()
    preset = st.radio("Patient Profiles:", ["Default", "Healthy / Low Risk Sample", "High Risk Clinical Case"])

vals = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}

if preset == "Healthy / Low Risk Sample":
    vals = {
        "age": 35, "sex": 0, "cp": 1, "trestbps": 115, "chol": 180, "fbs": 0,
        "restecg": 0, "thalach": 175, "exang": 0, "oldpeak": 0.2, "slope": 2, "ca": 0, "thal": 2
    }
elif preset == "High Risk Clinical Case":
    vals = {
        "age": 64, "sex": 1, "cp": 0, "trestbps": 155, "chol": 290, "fbs": 1,
        "restecg": 1, "thalach": 118, "exang": 1, "oldpeak": 3.4, "slope": 1, "ca": 3, "thal": 3
    }

st.subheader("Patient Vitals & Clinical Tests")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (years)", 18, 100, vals["age"])
    sex_label = st.selectbox("Sex", ["Male (1)", "Female (0)"], index=0 if vals["sex"] == 1 else 1)
    sex = 1 if "Male" in sex_label else 0
    cp_opts = {
        0: "0: Typical Angina",
        1: "1: Atypical Angina",
        2: "2: Non-anginal pain",
        3: "3: Asymptomatic"
    }
    cp = st.selectbox("Chest Pain Type (cp)", options=list(cp_opts.keys()), format_func=lambda x: cp_opts[x], index=vals["cp"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, vals["trestbps"])

with col2:
    chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 580, vals["chol"])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (<= 120 mg/dl)", index=vals["fbs"])
    restecg_opts = {0: "0: Normal", 1: "1: ST-T wave abnormality", 2: "2: Left ventricular hypertrophy"}
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], format_func=lambda x: restecg_opts[x], index=vals["restecg"])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 230, vals["thalach"])

with col3:
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=vals["exang"])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, float(vals["oldpeak"]), step=0.1)
    slope_opts = {0: "0: Upsloping", 1: "1: Flat", 2: "2: Downsloping"}
    slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2], format_func=lambda x: slope_opts[x], index=vals["slope"])
    ca = st.selectbox("Major Vessels Colored (ca)", options=[0, 1, 2, 3, 4], index=vals["ca"])
    thal_opts = {0: "0: Normal", 1: "1: Fixed Defect", 2: "2: Reversable Defect", 3: "3: Other"}
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], format_func=lambda x: thal_opts[x], index=vals["thal"])

st.divider()

if st.button("🩺 Run Risk Evaluation", type="primary", use_container_width=True):
    payload = {
        "age": int(age),
        "sex": int(sex),
        "cp": int(cp),
        "trestbps": int(trestbps),
        "chol": int(chol),
        "fbs": int(fbs),
        "restecg": int(restecg),
        "thalach": int(thalach),
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal),
    }

    with st.spinner("Processing clinical evaluation..."):
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                has_disease = data["has_heart_disease"]
                prob = data["risk_probability"]

                out_c1, out_c2 = st.columns(2)
                with out_c1:
                    if has_disease:
                        st.error(f"### {data['diagnosis']}")
                    else:
                        st.success(f"### {data['diagnosis']}")
                    st.write(f"**Clinical Status:** {data['status_indicator']}")

                with out_c2:
                    st.metric("Estimated Risk Probability", f"{prob * 100:.1f}%")
                    st.progress(min(max(prob, 0.0), 1.0))
            else:
                st.error(f"Backend error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend at `{api_url}`: {e}")
