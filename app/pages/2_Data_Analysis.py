import sys
from pathlib import Path
import pandas as pd
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.utils import PROCESSED, image_grid, rupees
st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide"); st.title("Dataset Analysis")
path = PROCESSED / "cleaned_cardekho.csv"
if not path.exists(): st.warning("Processed data missing. Run `python -m src.data_preprocessing` first."); st.stop()
df = pd.read_csv(path); c1,c2,c3=st.columns(3); c1.metric("Clean rows", f"{len(df):,}"); c2.metric("Features", len(df.columns)-1); c3.metric("Median selling price", rupees(df.selling_price.median()))
st.subheader("Dataset preview"); st.dataframe(df.head(20), use_container_width=True)
image_grid([
    ("price_distribution.png", "Selling-price distribution"),
    ("age_vs_price.png", "Price and car age"),
    ("km_vs_price.png", "Price and kilometres driven"),
    ("price_by_fuel.png", "Price by fuel type"),
    ("price_by_transmission.png", "Price by transmission"),
    ("correlation_heatmap.png", "Numeric correlation heatmap"),
])
