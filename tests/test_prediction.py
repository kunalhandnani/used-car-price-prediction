import json
import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from src import predict

def test_saved_model_prediction(monkeypatch, tmp_path):
    pipeline = Pipeline([("model", DummyRegressor(strategy="constant", constant=123456))]).fit(pd.DataFrame({"km_driven": [1]}), [123456])
    joblib.dump(pipeline, tmp_path / "random_forest_pipeline.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"features": ["km_driven"]}))
    monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
    assert predict.predict({"km_driven": 5000}) == [123456.0]
