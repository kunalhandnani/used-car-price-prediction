# Project Report: Used Car Price Analysis and Prediction

## Objective
Estimate CarDekho used-car selling prices and compare Multiple Linear Regression with Random Forest on one fixed held-out split.

## Methodology adaptation
The project follows the specification's methodology-based adaptation of Jiaying Gao (2024), *Second-hand Car Price Prediction Based on Multiple Linear Regression and Random Forest*. The original dataset is not reproduced; this experiment uses the supplied public CarDekho CSV, so numerical findings must not be claimed as an exact replication.

The attached PDF is a different source: Longania, Potharaju, and Deore (2021), *Price Prediction for Pre-Owned Cars Using Ensemble Machine Learning Techniques*, DOI 10.3233/APC210194. It studied 2,454 retained Mumbai records and compared Random Forest and XGBoost; it is supporting context only, not the primary Gao (2024) paper specified for this project.

## Data and preprocessing
The pipeline normalises schema variants, retains `selling_price` only as target, records data quality in `data/processed/data_audit.json`, removes exact duplicates and invalid records, parses unit-bearing numeric values, and derives/uses car age against reference year 2026. Legitimate high-value cars are not automatically removed as outliers.

## Experiment design
An 80:20 split using random state 42 is shared by both models. Numeric fields receive median imputation; categorical fields receive most-frequent imputation and one-hot encoding. All transformers are fit only on training rows and saved together with their estimator.

## Results
The real held-out test set has 3,083 rows. The evaluation generated the following results (INR):

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 270,497.21 | 491,948.78 | 0.6785 |
| Random Forest Regressor | 101,425.77 | 224,771.95 | 0.9329 |

Random Forest has lower MAE/RMSE and higher R² on this dataset and split. This is a dataset-specific comparison, not percentage accuracy and not a claim of causal feature effects. The generated evaluation also includes actual-versus-predicted, residual, and Random Forest importance plots.

## Limitations
Prices depend on market, time, locality, and unrecorded car condition. Dataset schema and preprocessing differ from the research sources; comparisons are qualitative unless conditions and metrics are truly comparable.
