"""Evaluate saved models; metrics and plots always come from held-out test data."""
from __future__ import annotations
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, TARGET

def evaluate_models() -> pd.DataFrame:
    test_path = PROCESSED_DIR / "test.csv"
    metadata_path = MODELS_DIR / "metadata.json"
    if not test_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Training artifacts are missing. Run: python -m src.train_model")
    test = pd.read_csv(test_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")); features = metadata["features"]
    records, prediction_columns = [], {"actual": test[TARGET]}
    FIGURES_DIR.mkdir(parents=True, exist_ok=True); sns.set_theme(style="whitegrid")
    for key, display in [("linear_regression", "Multiple Linear Regression"), ("random_forest", "Random Forest Regressor")]:
        model = joblib.load(MODELS_DIR / f"{key}_pipeline.joblib")
        prediction = model.predict(test[features]); prediction_columns[key] = prediction
        records.append({"model_key": key, "Model": display, "MAE_INR": mean_absolute_error(test[TARGET], prediction),
                        "RMSE_INR": mean_squared_error(test[TARGET], prediction) ** .5, "R2": r2_score(test[TARGET], prediction)})
        plt.figure(figsize=(6, 5)); plt.scatter(test[TARGET], prediction, alpha=.5); limits = [min(test[TARGET].min(), prediction.min()), max(test[TARGET].max(), prediction.max())]; plt.plot(limits, limits, "r--"); plt.xlabel("Actual Price (INR)"); plt.ylabel("Predicted Price (INR)"); plt.title(f"Actual vs Predicted: {display}"); plt.tight_layout(); plt.savefig(FIGURES_DIR / f"actual_vs_predicted_{key}.png", dpi=150); plt.close()
        residuals = test[TARGET] - prediction; plt.figure(figsize=(6, 5)); plt.scatter(prediction, residuals, alpha=.5); plt.axhline(0, color="r", linestyle="--"); plt.xlabel("Predicted Price (INR)"); plt.ylabel("Residual (Actual - Predicted)"); plt.title(f"Residuals: {display}"); plt.tight_layout(); plt.savefig(FIGURES_DIR / f"residuals_{key}.png", dpi=150); plt.close()
        if key == "random_forest":
            names = model.named_steps["preprocessor"].get_feature_names_out(); importances = model.named_steps["model"].feature_importances_
            importance = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
            importance.to_csv(MODELS_DIR / "random_forest_feature_importance.csv", index=False)
            plt.figure(figsize=(8, 6)); sns.barplot(data=importance.head(15), y="feature", x="importance", color="#2f6f9f"); plt.title("Random Forest Feature Importance (Top 15)"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "random_forest_feature_importance.png", dpi=150); plt.close()
    metrics = pd.DataFrame(records); metrics.to_csv(MODELS_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(prediction_columns).to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    return metrics

def main() -> None: print(evaluate_models().to_string(index=False))
if __name__ == "__main__": main()
