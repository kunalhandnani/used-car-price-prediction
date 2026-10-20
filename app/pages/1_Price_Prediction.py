import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.utils import metadata, read_metrics, rupees
from src.config import REFERENCE_YEAR
from src.predict import predict
st.set_page_config(page_title="Price Prediction", page_icon="🚗", layout="wide"); st.title("Car Price Estimate")
meta = metadata()
if not meta:
    st.warning("No saved pipelines found. From the project root, run `python -m src.train_model` and `python -m src.evaluate_model`."); st.stop()
features = meta["features"]
with st.form("prediction"):
    values = {}
    for feature in features:
        if feature == "car_age": values[feature] = REFERENCE_YEAR - st.number_input("Manufacturing year", min_value=1900, max_value=REFERENCE_YEAR, value=2018)
        elif feature == "km_driven": values[feature] = st.number_input("Kilometres driven", min_value=0, max_value=2_000_000, value=50_000)
        elif feature == "mileage": values[feature] = st.number_input("Mileage (km/l)", min_value=0.0, max_value=100.0, value=18.0)
        elif feature == "engine": values[feature] = st.number_input("Engine capacity (cc)", min_value=0.0, max_value=10_000.0, value=1200.0)
        elif feature == "max_power": values[feature] = st.number_input("Maximum power (bhp)", min_value=0.0, max_value=2_000.0, value=80.0)
        elif feature == "seats": values[feature] = st.number_input("Seats", min_value=1.0, max_value=100.0, value=5.0)
        elif feature == "fuel": values[feature] = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        elif feature == "transmission": values[feature] = st.selectbox("Transmission", ["Manual", "Automatic"])
        elif feature == "seller_type": values[feature] = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
        elif feature == "owner": values[feature] = st.selectbox("Previous owner", ["First Owner", "Second Owner", "Third Owner", "Test Drive Car"])
        else: values[feature] = st.text_input(feature.replace("_", " ").title(), value="Unknown")
    submitted = st.form_submit_button("Predict Price")
if submitted:
    try:
        estimates = {key: predict(values, key)[0] for key in ("linear_regression", "random_forest")}
        cols = st.columns(2); cols[0].metric("Multiple Linear Regression", rupees(estimates["linear_regression"])); cols[1].metric("Random Forest", rupees(estimates["random_forest"]))
        st.caption("This is a model estimate, not a guaranteed market value.")
    except Exception as exc: st.error(f"Could not make a prediction: {exc}")
metrics = read_metrics()
if metrics is not None: st.subheader("Held-out test metrics"); st.dataframe(metrics, use_container_width=True, hide_index=True)
