# HERAI — Menstrual Health Prediction System

## 1. Project Overview

HERAI is a machine learning system designed to predict a user's next-cycle overall health score using menstrual-cycle observations, demographic and lifestyle factors, and prior-cycle health data.

The project demonstrates end-to-end data validation, feature engineering, regression model training, model comparison, privacy-preserving inference, and automated testing.

**Note:** HERAI is an educational prototype and is not a medically validated diagnostic or clinical decision-making system.

---

## 2. Objectives

- Validate raw menstrual-cycle logs and user profile schemas before processing.
- Prevent data leakage by isolating users across training and testing splits (`GroupShuffleSplit`).
- Engineer chronological lag features and align next-cycle prediction targets.
- Train and compare regression models (Ridge Regression, Random Forest).
- Provide a privacy-preserving inference engine that predicts next-cycle health using **only** user-owned data and learned population baseline defaults (without borrowing stranger records).
- Deliver automated unit and integration tests with `pytest`.

---

## 3. Technology Stack

- **Language:** Python (3.9+)
- **Core Libraries:** Pandas, NumPy, Scikit-learn, Joblib, Pydantic
- **Testing:** Pytest
- **Data Storage:** Validated CSV files
- **Selected Model:** 37-feature Ridge Regression Pipeline

---

## 4. Project Structure

```text
HERAI/
├── data/
│   ├── raw/
│   │   ├── Period_Log.csv          # 17,976 raw menstrual cycle records
│   │   └── User_Profile.csv        # 2,000 raw user demographic & lifestyle records
│   └── processed/
│       ├── merged_data.csv         # Validated many-to-one merged dataset
│       ├── engineered_data.csv     # Chronological lag features and aligned target
│       └── training_data.csv       # Filtered training dataset (15,976 records)
├── models/
│   ├── herai_ridge_model.pkl       # 42-feature Ridge pipeline
│   └── herai_ridge_37_features.pkl # 37-feature Ridge production model
├── src/
│   ├── herai/                      # Reusable core Python package
│   │   ├── __init__.py
│   │   ├── config.py               # Paths, feature sets, validation bounds
│   │   ├── data.py                 # Schema validation, merging, lag generation
│   │   ├── models.py               # Pipelines, user-grouped splitting, evaluation
│   │   └── inference.py            # Ethical inference engine & input validation
│   ├── 01_inspect_data.py          # Exploratory data analysis & health score formula audit
│   ├── 02_merge_data.py            # Schema validation and dataset merge
│   ├── 03_create_features.py       # Chronological sort and lag feature creation
│   ├── 04_prepare_data.py          # Training dataset preparation
│   ├── 05_train_model.py           # Trains 42-feature Ridge model
│   ├── 06_compare_models.py        # Compares Ridge vs RF vs 37-feature Ridge
│   ├── 06_predict.py               # Programmatic & sample prediction demonstration
│   ├── 07_user_input.py            # Interactive CLI prediction
│   ├── 08_feature_leakage_audit.py # Target leakage verification
│   ├── inspect_features.py         # Feature timing exploration
│   └── src/
│       └── 06_compare_models.py    # Backward-compatibility forwarding stub
├── tests/
│   └── test_herai.py               # Automated pytest suite
├── pyproject.toml                  # Package configuration & pytest settings
├── requirements.txt                # Dependency manifest
└── README.md
```

---

## 5. Dataset and Preprocessing

The project uses two raw datasets:
- `Period_Log.csv` (17,976 records, 21 columns) — menstrual cycle dates, symptom severity, mood, sleep, pain, and health scores.
- `User_Profile.csv` (2,000 records, 14 columns) — demographics, BMI, diet quality, exercise frequency, and baseline stress.

### Preprocessing & Leakage Safeguards
1. **Schema Validation:** Verifies that all required fields are present and confirms `user_id` is unique in `User_Profile.csv` and `(user_id, cycle_number)` is unique in `Period_Log.csv`.
2. **Chronological Ordering:** Sorts each user's records strictly by `(user_id, start_date)` before generating lag features.
3. **Lag Feature Engineering:** Computes previous-cycle metrics (`overall_health_score.shift(1)`, `cycle_length_days.shift(1)`, `pain_level.shift(1)`, etc.).
4. **Target Alignment:** The prediction target `next_health_score` is defined strictly as `overall_health_score.shift(-1)`. For the terminal cycle of each user, `next_health_score` is `NaN` and is excluded from training.
5. **No Data Leakage:** Preprocessing imputers (`SimpleImputer`) and scalers (`StandardScaler`) are encapsulated inside Scikit-learn `Pipeline` objects so that statistics are fitted exclusively on training folds during cross-validation.

---

## 6. Model Comparison

Models evaluated on the same 80/20 user-grouped split (`GroupShuffleSplit`, 1,600 train users, 400 test users):

| Model | MAE | RMSE | R² Score | Description |
|---|---:|---:|---:|---|
| Ridge Regression (42 features) | 0.6452 | 0.8029 | 0.4616 | Includes all features (some lab/inaccessible at home) |
| Random Forest (42 features) | 0.6550 | 0.8130 | 0.4480 | Non-linear tree ensemble |
| **Ridge Without Audited Features (37 features)** | **0.6448** | **0.8025** | **0.4620** | **Selected production candidate** (excludes lab/timing features) |

---

## 7. How to Run the Project

### 7.1 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 7.2 Run the Pipeline

Execute the pipeline in sequential order:

```bash
# 1. Inspect raw data schemas and correlations
python src/01_inspect_data.py

# 2. Validate schemas and merge raw datasets
python src/02_merge_data.py

# 3. Chronologically sort and engineer lag features
python src/03_create_features.py

# 4. Prepare training dataset
python src/04_prepare_data.py

# 5. Train Ridge baseline model
python src/05_train_model.py

# 6. Verify zero target leakage
python src/08_feature_leakage_audit.py

# 7. Compare models and save selected 37-feature Ridge model
python src/06_compare_models.py

# 8. Run prediction on sample data
python src/06_predict.py

# 9. Run interactive personalized user prediction
python src/07_user_input.py
```

### 7.3 Run Automated Tests

Execute the full test suite with:

```bash
pytest
```

---

## 8. Ethical Inference & Privacy Safeguards

Prior prototypes relied on copying an arbitrary stranger's data row from the training set to fill missing features. **This defect has been completely resolved.**

HERAI's inference pipeline:
- **Never copies or borrows other users' records.**
- Validates all user inputs against domain boundaries (e.g. age: 12–100, BMI: 10–70, pain/stress/mood: 0–10).
- If a user provides historical cycle records, features are calculated strictly from their own data.
- For first-time users (Cycle 1 / no historical cycle logs), the pipeline applies documented cohort baseline defaults learned by the preprocessing pipeline from the training set, clearly labeling this in the prediction output.

---

## 9. Limitations & Medical Disclaimer

- **Non-Clinical Prototype:** HERAI is an educational demonstration and has not been clinically validated.
- **Medical Decisions:** The outputs must never be used for diagnosis, medical treatment, or contraception planning.
