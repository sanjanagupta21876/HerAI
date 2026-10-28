# HERAI — Menstrual Health Prediction System

## 1. Project Overview

HERAI is a machine learning project designed to predict a user's next-cycle overall health score using menstrual-cycle information, lifestyle factors, and previous-cycle health data.

The project demonstrates data preprocessing, feature engineering, regression model training, model evaluation, and prediction through user input.

**Note:** HERAI is an educational prototype and is not a medically validated diagnostic or prediction system.

## 2. Objectives

- Analyze menstrual-cycle and user-profile data.
- Create features using current and previous-cycle information.
- Train and compare machine learning regression models.
- Predict the next-cycle overall health score.
- Evaluate model performance using standard regression metrics.

## 3. Technology Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Development Environment:** Visual Studio Code
- **Data Storage:** CSV files
- **Model Type:** Ridge Regression

## 4. Project Structure

```text
HERAI/
├── data/
│   ├── raw/
│   │   ├── Period_Log.csv
│   │   └── User_Profile.csv
│   └── processed/
│       ├── merged_data.csv
│       ├── engineered_data.csv
│       └── training_data.csv
├── models/
│   ├── herai_ridge_model.pkl
│   └── herai_ridge_37_features.pkl
├── src/
│   ├── 01_inspect_data.py
│   ├── 02_merge_data.py
│   ├── 03_create_features.py
│   ├── 04_prepare_data.py
│   ├── 05_train_model.py
│   ├── 06_predict.py
│   ├── 07_user_input.py
│   ├── 08_feature_leakage_audit.py
│   └── src/
│       └── 06_compare_models.py
└── README.md
```

## 5. Dataset and Preprocessing

The project uses two raw datasets:

- `Period_Log.csv` — menstrual-cycle records and related health information.
- `User_Profile.csv` — user-level demographic and lifestyle information.

The datasets are merged using `user_id`. Feature engineering creates previous-cycle features and aligns the target with the following cycle's overall health score.

The merged dataset contains 17,976 records. After removing records without a next-cycle target, 15,976 records remain for model evaluation and training.

The original dataset's source and generation method should be verified before making claims about its representativeness.

## 6. Machine Learning Workflow

1. Inspect the raw datasets.
2. Merge cycle records with user profiles.
3. Engineer current-cycle and previous-cycle features.
4. Prepare the data for model training.
5. Split data by user to keep users separate between training and testing.
6. Train and compare regression models.
7. Save the selected model.
8. Accept user inputs and generate a prototype prediction.

## 7. Model Comparison

Three regression models were evaluated using the same user-grouped train/test split.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Ridge Regression | 0.6452 | 0.8029 | 0.4616 |
| Random Forest | 0.6536 | 0.8116 | 0.4498 |
| Ridge Regression (37 features) | 0.6448 | 0.8025 | 0.4620 |

The 37-feature Ridge Regression model achieved the best results among these candidates on this test split.

- **MAE:** Mean Absolute Error; lower is better.
- **RMSE:** Root Mean Squared Error; lower is better.
- **R²:** Measures the proportion of target variance explained by the model; higher is generally better.

These results are preliminary and do not establish clinical reliability or performance on new real-world populations.

## 8. How to Run the Project

Open a terminal in the HERAI project root directory.

Install the required libraries if needed:

```bash
pip install pandas numpy scikit-learn joblib
```

Run the scripts in the appropriate order when rebuilding the pipeline:

```bash
python src/01_inspect_data.py
python src/02_merge_data.py
python src/03_create_features.py
python src/04_prepare_data.py
python src/05_train_model.py
python src/08_feature_leakage_audit.py
python src/src/06_compare_models.py
python src/07_user_input.py
```

The model comparison script saves `models/herai_ridge_37_features.pkl`, which is loaded by the user-input prediction script.

Run the pipeline from the project root so relative paths resolve correctly. The comparison script should be rerun after retraining if you want to refresh the selected 37-feature model.

## 9. Limitations

- The model has not been clinically validated.
- The source and generation process of the dataset need verification.
- Some user-input prediction features still come from a sample dataset row, so the current prediction is not fully personalized.
- The evaluation uses a user-grouped random split rather than a separate future-time evaluation.
- The output must not be used for diagnosis, treatment, or medical decisions.

## 10. Future Improvements

- Collect and validate suitable real-world data with appropriate consent and privacy safeguards.
- Build a prediction pipeline that uses only information genuinely available at prediction time.
- Replace sample-row features with complete user-specific inputs or a principled missing-feature strategy.
- Add temporal validation and further model evaluation.
- Develop a user-friendly interface after the prediction pipeline is improved.

## 11. Conclusion

HERAI demonstrates an end-to-end machine learning workflow for a menstrual-health score prediction prototype, including data merging, feature engineering, model comparison, evaluation, and user-input inference. Further validation and improvements are required before it can support reliable personalized predictions.