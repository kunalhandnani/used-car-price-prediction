"""Reusable, saved-pipeline prediction API."""
from __future__ import annotations
import json
import joblib
import pandas as pd
from .config import MODELS_DIR

def load_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    if not path.exists(): raise FileNotFoundError("Model artifacts missing. Run: python -m src.train_model")
    return json.loads(path.read_text(encoding="utf-8"))

def predict(data: pd.DataFrame | dict, model_key: str = "random_forest") -> list[float]:
    metadata = load_metadata(); frame = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    features = metadata["features"]; missing = [c for c in features if c not in frame]
    if missing: raise ValueError(f"Prediction input is missing required columns: {missing}")
    model_path = MODELS_DIR / f"{model_key}_pipeline.joblib"
    if not model_path.exists(): raise FileNotFoundError(f"Model artifact missing: {model_path.name}")
    return joblib.load(model_path).predict(frame[features]).tolist()
