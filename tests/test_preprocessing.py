import pandas as pd
from src.data_preprocessing import normalise_columns
from src.feature_engineering import engineer_features

def test_schema_mapping_and_feature_engineering():
    raw = pd.DataFrame({"Selling Price": [100000], "Km Driven": [12000], "Fuel Type": ["Petrol"], "Seller Type": ["Individual"], "Transmission Type": ["Manual"], "Vehicle Age": [5], "Mileage": [20.1]})
    mapped, _ = normalise_columns(raw)
    result = engineer_features(mapped)
    assert {"selling_price", "km_driven", "fuel", "seller_type", "transmission", "car_age"}.issubset(result.columns)
    assert result.loc[0, "car_age"] == 5
