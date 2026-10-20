from pathlib import Path
import json
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"; PROCESSED = ROOT / "data" / "processed"; FIGURES = ROOT / "reports" / "figures"
def rupees(value: float) -> str: return f"₹{value:,.0f}"
def read_metrics() -> pd.DataFrame | None:
    path = MODELS / "model_metrics.csv"; return pd.read_csv(path) if path.exists() else None
def metadata() -> dict | None:
    path = MODELS / "metadata.json"; return json.loads(path.read_text()) if path.exists() else None
def image(name: str, caption: str) -> None:
    path = FIGURES / name
    if path.exists(): st.image(str(path), caption=caption, width=420)
    else: st.info("This chart is generated after preprocessing/evaluation.")

def image_grid(items: list[tuple[str, str]]) -> None:
    """Show compact chart cards in two columns with readable labels beneath."""
    for start in range(0, len(items), 2):
        columns = st.columns(2, gap="large")
        for column, (name, caption) in zip(columns, items[start:start + 2]):
            with column:
                path = FIGURES / name
                if path.exists():
                    st.image(str(path), width=420)
                    st.markdown(
                        f"<p style='font-size:1.15rem; font-weight:600; margin-top:0.35rem'>{caption}</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("This chart is generated after preprocessing/evaluation.")
