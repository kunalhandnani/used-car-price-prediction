import streamlit as st
st.set_page_config(page_title="Used Car Price Analysis", page_icon="🚗", layout="wide")
st.title("Used Car Price Analysis and Prediction")
st.write("A local ML-lab dashboard comparing Multiple Linear Regression and Random Forest on the CarDekho dataset.")
st.info("Use the sidebar to open Price Prediction, Data Analysis, or Model Performance. Train first with `python -m src.train_model`, then evaluate with `python -m src.evaluate_model`.")
