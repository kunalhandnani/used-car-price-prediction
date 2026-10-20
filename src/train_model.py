"""Train the two required pipelines on the same held-out split."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .config import MODELS_DIR, RANDOM_STATE, TARGET
from .data_preprocessing import available_features, load_and_clean, save_processed_and_split

def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")

def train_models() -> dict:
    df, audit = load_and_clean()
    train, _ = save_processed_and_split(df, audit)
    numeric, categorical = available_features(train)
    features = numeric + categorical
    if not numeric or not categorical:
        raise ValueError("Dataset lacks sufficient numeric/categorical baseline features after schema mapping.")
    X_train, y_train = train[features], train[TARGET]
    estimators = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"trained_at_utc": datetime.now(timezone.utc).isoformat(), "features": features,
                "numeric_features": numeric, "categorical_features": categorical,
                "target": TARGET, "training_rows": len(train), "cleaning_audit": audit}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(numeric, categorical)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODELS_DIR / f"{name}_pipeline.joblib")
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)
    return metadata

def main() -> None:
    result = train_models(); print(f"Trained required pipelines using {result['training_rows']} training rows.")
if __name__ == "__main__": main()
