"""Loading, schema mapping, audit, cleaning, EDA, and reproducible splitting."""
from __future__ import annotations
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from .config import (COLUMN_ALIASES, FIGURES_DIR, NUMERIC_FEATURES, PROCESSED_DIR,
                     RANDOM_STATE, RAW_DIR, REFERENCE_YEAR, TARGET, TEST_SIZE)
from .feature_engineering import engineer_features

def normalise_header(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")

def find_raw_csv(raw_dir: Path = RAW_DIR) -> Path:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Put the CarDekho CSV in data/raw/.")
    return files[0]

def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    original = {normalise_header(c): c for c in df.columns}
    rename, mapping = {}, {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            source = original.get(normalise_header(alias))
            if source:
                rename[source] = canonical
                mapping[canonical] = source
                break
    result = df.rename(columns=rename)
    required = [TARGET, "km_driven", "fuel", "seller_type", "transmission"]
    missing = [c for c in required if c not in result.columns]
    if missing or ("year" not in result.columns and "car_age" not in result.columns):
        needed = missing + ([] if ("year" in result.columns or "car_age" in result.columns) else ["year or vehicle_age"])
        raise ValueError(f"Unsupported dataset schema. Required fields not mapped: {needed}. Found: {list(df.columns)}")
    return result, mapping

def load_and_clean(csv_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = csv_path or find_raw_csv()
    raw = pd.read_csv(path)
    mapped, mapping = normalise_columns(raw)
    engineered = engineer_features(mapped)
    audit = {"source_file": path.name, "rows_before_cleaning": len(engineered),
             "columns_mapped": mapping, "missing_values_before": engineered.isna().sum().dropna().astype(int).to_dict(),
             "duplicate_rows": int(engineered.duplicated().sum())}
    clean = engineered.drop_duplicates().copy()
    invalid_target = clean[TARGET].isna() | (clean[TARGET] <= 0)
    invalid_km = clean["km_driven"].notna() & (clean["km_driven"] < 0)
    invalid_age = clean["car_age"].notna() & ((clean["car_age"] < 0) | (clean["car_age"] > 126))
    audit["removed_invalid_target"] = int(invalid_target.sum())
    audit["removed_invalid_km"] = int(invalid_km.sum())
    audit["removed_implausible_age"] = int(invalid_age.sum())
    clean = clean.loc[~(invalid_target | invalid_km | invalid_age)].reset_index(drop=True)
    audit["rows_after_cleaning"] = len(clean)
    audit["missing_values_after"] = clean.isna().sum().astype(int).to_dict()
    audit["outlier_policy"] = "No automatic statistical outlier removal; legitimate expensive vehicles are retained."
    return clean, audit

def available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in ["fuel", "seller_type", "transmission", "owner", "brand"] if c in df.columns]
    return numeric, categorical

def save_processed_and_split(df: pd.DataFrame, audit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "cleaned_cardekho.csv", index=False)
    numeric, categorical = available_features(df)
    with open(PROCESSED_DIR / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump({**audit, "reference_year": REFERENCE_YEAR, "numeric_features": numeric, "categorical_features": categorical}, f, indent=2)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return train, test

def generate_eda(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plots = [
        ("price_distribution.png", lambda: sns.histplot(df[TARGET], bins=40, kde=True), "Selling Price Distribution", "Selling Price (INR)", "Count"),
        ("age_vs_price.png", lambda: sns.scatterplot(data=df, x="car_age", y=TARGET, alpha=.45), "Car Age vs Selling Price", "Car Age (years)", "Selling Price (INR)"),
        ("km_vs_price.png", lambda: sns.scatterplot(data=df, x="km_driven", y=TARGET, alpha=.45), "Kilometres Driven vs Selling Price", "Kilometres Driven", "Selling Price (INR)"),
    ]
    for filename, draw, title, xlabel, ylabel in plots:
        plt.figure(figsize=(8, 5)); draw(); plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(FIGURES_DIR / filename, dpi=150); plt.close()
    for category, filename in [("fuel", "price_by_fuel.png"), ("transmission", "price_by_transmission.png")]:
        if category in df:
            plt.figure(figsize=(8, 5)); sns.boxplot(data=df, x=category, y=TARGET); plt.title(f"Selling Price by {category.replace('_', ' ').title()}"); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(FIGURES_DIR / filename, dpi=150); plt.close()
    numeric = [c for c in [TARGET, "car_age", "km_driven", "mileage", "engine", "max_power", "seats"] if c in df]
    plt.figure(figsize=(8, 6)); sns.heatmap(df[numeric].corr(numeric_only=True), cmap="Blues", annot=True, fmt=".2f"); plt.title("Numeric Feature Correlations"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150); plt.close()

def main() -> None:
    df, audit = load_and_clean(); save_processed_and_split(df, audit); generate_eda(df)
    print(json.dumps(audit, indent=2)); print(f"EDA figures saved to {FIGURES_DIR}")

if __name__ == "__main__": main()
