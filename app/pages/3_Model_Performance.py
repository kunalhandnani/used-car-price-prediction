import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.utils import image_grid, read_metrics
st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide"); st.title("Model Performance")
metrics = read_metrics()
if metrics is None: st.warning("Evaluation results missing. Run `python -m src.evaluate_model` after training."); st.stop()
st.dataframe(metrics, use_container_width=True, hide_index=True)
best = metrics.sort_values("RMSE_INR").iloc[0]; st.info(f"On this held-out test set, {best['Model']} has the lowest RMSE. This conclusion is dataset-specific; R² is not percentage accuracy.")
image_grid([
    ("actual_vs_predicted_linear_regression.png", "Linear Regression: actual vs predicted"),
    ("residuals_linear_regression.png", "Linear Regression residuals"),
    ("actual_vs_predicted_random_forest.png", "Random Forest: actual vs predicted"),
    ("residuals_random_forest.png", "Random Forest residuals"),
    ("random_forest_feature_importance.png", "Random Forest feature importance - association, not causation"),
])
