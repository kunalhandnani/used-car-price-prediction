# Used Car Price Analysis and Prediction

Machine Learning lab mini-project comparing Multiple Linear Regression and Random Forest on a public CarDekho dataset. It is a methodology-based adaptation: it does **not** claim to reproduce the numerical results of the referenced research exactly.

## Project setup

From this folder in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
python -m src.data_preprocessing
python -m src.train_model
python -m src.evaluate_model
streamlit run app/streamlit_app.py
```

The supplied file is `data/raw/cardekho_dataset.csv`. Raw data is ignored by Git and never overwritten. If replacing it, put exactly one compatible CSV in `data/raw/` and rerun preprocessing.

## Reproducible experiment

- Target: `selling_price` (INR).
- Split: 80:20, `random_state=42`, shared by both models.
- Car age: sourced from `vehicle_age` when available or derived with fixed reference year **2026**.
- Preprocessing happens inside each fitted pipeline: median numeric imputation plus most-frequent categorical imputation and one-hot encoding. It is fit only on training rows.
- Initial features are car age, kilometres, mileage, engine, power, seats, fuel, seller type, transmission, owner if available, and brand. `selling_price` is never an input.
- Exact duplicates and records with invalid target, negative mileage, or implausible age are removed. Statistical outliers are retained, and the data audit records row counts.

Generated outputs are in `data/processed/`, `models/`, and `reports/figures/`. `models/model_metrics.csv` contains actual held-out MAE, RMSE, and R² once evaluation runs.

## Dashboard

Run `streamlit run app/streamlit_app.py`. The three pages are Price Prediction, Data Analysis, and Model Performance. Prediction loads the saved pipelines directly and shows both real estimates. Pages provide a helpful message when artifacts are missing.

## Research and limitations

The project follows the specification's comparison of Multiple Linear Regression and Random Forest. The attached Longania, Potharaju, and Deore (2021), *Price Prediction for Pre-Owned Cars Using Ensemble Machine Learning Techniques*, provides useful context on a Mumbai dataset and random forest/XGBoost comparison, but is not the Gao (2024) paper named as the primary reference in the specification. It must not be cited as Gao (2024).

Dataset version, used-car market timing, unobserved vehicle condition, and missing fields limit generalisation. Random Forest feature importance indicates association, not causation.
