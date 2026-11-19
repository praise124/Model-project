import streamlit as st
import requests

st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 California Real Estate Price Prediction")
st.markdown("Estimate district median property values using Scikit-Learn with a FastAPI backend.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("Backend URL", value="http://127.0.0.1:8003")
    st.divider()
    location_preset = st.selectbox(
        "Choose Region Preset:",
        [
            "Custom Input",
            "San Francisco Bay Area (Near Bay)",
            "Central Valley (Inland)",
            "Coastal Southern California (Near Ocean)",
        ]
    )

defaults = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.32,
    "ocean_proximity": "NEAR BAY"
}

if location_preset == "San Francisco Bay Area (Near Bay)":
    defaults = {
        "longitude": -122.25, "latitude": 37.85, "housing_median_age": 52.0,
        "total_rooms": 2500.0, "total_bedrooms": 420.0, "population": 1100.0,
        "households": 410.0, "median_income": 9.20, "ocean_proximity": "NEAR BAY"
    }
elif location_preset == "Central Valley (Inland)":
    defaults = {
        "longitude": -119.80, "latitude": 36.75, "housing_median_age": 25.0,
        "total_rooms": 1800.0, "total_bedrooms": 350.0, "population": 1200.0,
        "households": 340.0, "median_income": 2.85, "ocean_proximity": "INLAND"
    }
elif location_preset == "Coastal Southern California (Near Ocean)":
    defaults = {
        "longitude": -117.20, "latitude": 32.80, "housing_median_age": 35.0,
        "total_rooms": 3100.0, "total_bedrooms": 580.0, "population": 1600.0,
        "households": 560.0, "median_income": 6.80, "ocean_proximity": "NEAR OCEAN"
    }

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Geolocation & Environment")
    longitude = st.slider("Longitude", -124.3, -114.3, float(defaults["longitude"]), step=0.01)
    latitude = st.slider("Latitude", 32.5, 42.0, float(defaults["latitude"]), step=0.01)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
        index=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"].index(defaults["ocean_proximity"])
    )
    median_income = st.slider(
        "Median Income (tens of thousands $USD)",
        min_value=0.5,
        max_value=15.0,
        value=float(defaults["median_income"]),
        step=0.1,
        help="E.g., 3.5 represents $35,000 median annual income."
    )

with col2:
    st.subheader("🏘️ District Characteristics")
    housing_median_age = st.slider("Housing Median Age (years)", 1.0, 52.0, float(defaults["housing_median_age"]), step=1.0)
    total_rooms = st.number_input("Total Rooms", min_value=10.0, max_value=40000.0, value=float(defaults["total_rooms"]), step=50.0)
    total_bedrooms = st.number_input("Total Bedrooms", min_value=5.0, max_value=8000.0, value=float(defaults["total_bedrooms"]), step=20.0)
    col_sub1, col_sub2 = st.columns(2)
    with col_sub1:
        population = st.number_input("Population", min_value=10.0, max_value=35000.0, value=float(defaults["population"]), step=50.0)
    with col_sub2:
        households = st.number_input("Households", min_value=5.0, max_value=6000.0, value=float(defaults["households"]), step=20.0)

st.divider()

if st.button("💰 Calculate Property Valuation", type="primary", use_container_width=True):
    payload = {
        "longitude": float(longitude),
        "latitude": float(latitude),
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(total_bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": float(median_income),
        "ocean_proximity": ocean_proximity,
    }

    with st.spinner("Valuating district pricing model..."):
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.success("### Estimated Median House Price")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Model Estimate", data["formatted_price"])
                with res_col2:
                    st.info(f"**District Income Tier:** {data['median_income_bracket']}")
            else:
                st.error(f"Backend returned error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend at `{api_url}`: {e}")
