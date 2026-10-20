"""Shared configuration. Paths are anchored at the project root."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TARGET = "selling_price"
REFERENCE_YEAR = 2026
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Baseline is deliberately small and interpretable. It is built dynamically from
# available columns, so documented schema variants remain usable.
NUMERIC_FEATURES = ["car_age", "km_driven", "mileage", "engine", "max_power", "seats"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "owner", "brand"]

COLUMN_ALIASES = {
    "selling_price": ["selling_price", "selling price", "price", "sell_price"],
    "year": ["year", "manufacturing_year", "model_year"],
    "car_age": ["vehicle_age", "car_age", "used_years", "used_years_2021"],
    "km_driven": ["km_driven", "km driven", "driven", "kilometers_driven", "kilometres_driven"],
    "fuel": ["fuel", "fuel_type", "fuel type"],
    "seller_type": ["seller_type", "seller type"],
    "transmission": ["transmission", "transmission_type", "gear_type", "gear type"],
    "owner": ["owner", "owner_type", "nonowners", "previous_owner"],
    "mileage": ["mileage", "kmpl"],
    "engine": ["engine", "engg_capacity", "engine_capacity"],
    "max_power": ["max_power", "max power", "power"],
    "seats": ["seats", "seat_capacity", "seating_capacity"],
    "brand": ["brand", "make"],
    "name": ["name", "car_name", "model"],
}
