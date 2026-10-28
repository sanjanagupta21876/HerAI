
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the engineered dataset
df = pd.read_csv("data/processed/engineered_data.csv")

# 2. Remove rows where the target is missing
df = df.dropna(subset=["next_health_score"]).copy()

# 3. Separate features and target
target_column = "next_health_score"

excluded_columns = [
    "user_id",
    "start_date",
    "next_health_score",
    "cycle_number"
]

suspicious_features = [
    "estrogen_pgml",
    "progesterone_ngml",
    "ovulation_result",
    "log_consistency_score",
    "prepared_before_period"
]

X = df.drop(columns=excluded_columns)
X_without_suspicious = X.drop(
    columns=suspicious_features,
    errors="ignore"
)

y = df[target_column]
groups = df["user_id"]

# 4. Split by user so the same user cannot appear in both sets
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

X_train_without = X_without_suspicious.iloc[train_idx]
X_test_without = X_without_suspicious.iloc[test_idx]

print("Original feature count:", X_train.shape[1])
print("Feature count without audited features:", X_train_without.shape[1])

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training users:", groups.iloc[train_idx].nunique())
print("Testing users:", groups.iloc[test_idx].nunique())

# 5. Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# 6. Preprocessing for Ridge Regression
ridge_preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# 7. Preprocessing for Random Forest
forest_preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# 8. Define both models
models = {
    "Ridge Regression": Pipeline([
        ("preprocessor", ridge_preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", forest_preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        ))
    ]),
    "Ridge Without Audited Features": Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), X_train_without.select_dtypes(
                include=["number"]
            ).columns.tolist()),

            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), X_train_without.select_dtypes(
                exclude=["number"]
            ).columns.tolist())
        ])),
        ("regressor", Ridge(alpha=1.0))
    ])
}


# 9. Train, predict, and evaluate
results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Use reduced features only for the third model
    if model_name == "Ridge Without Audited Features":
        features_train = X_train_without
        features_test = X_test_without
    else:
        features_train = X_train
        features_test = X_test

    model.fit(features_train, y_train)
    predictions = model.predict(features_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    })


# 10. Display comparison
results_df = pd.DataFrame(results)
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# Save the Ridge model trained without audited features
best_model = models["Ridge Without Audited Features"]

model_path = Path("models/herai_ridge_37_features.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_path)

print("\n37-feature Ridge model saved to:", model_path)
