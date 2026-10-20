"""Defensive feature parsing for CarDekho schema variants."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .config import REFERENCE_YEAR

def parse_numeric(value: object) -> float:
    """Return the first meaningful numeric token, or NaN for malformed values."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse unit-bearing fields and derive car_age with a fixed reference year."""
    result = df.copy()
    for column in ("selling_price", "km_driven", "mileage", "engine", "max_power", "seats", "car_age", "year"):
        if column in result:
            result[column] = result[column].map(parse_numeric)
    if "year" in result:
        derived_age = REFERENCE_YEAR - result["year"]
        if "car_age" in result:
            result["car_age"] = result["car_age"].fillna(derived_age)
        else:
            result["car_age"] = derived_age
    if "brand" not in result and "name" in result:
        result["brand"] = result["name"].astype("string").str.split().str[0]
    return result
