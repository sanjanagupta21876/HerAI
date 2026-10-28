
import pandas as pd

# Load the raw datasets
period = pd.read_csv("data/raw/Period_Log.csv")
profile = pd.read_csv("data/raw/User_Profile.csv")

# Display basic information
print("========== DATA LOADED SUCCESSFULLY ==========")

print("\nPeriod Log shape:")
print(period.shape)

print("\nUser Profile shape:")
print(profile.shape)

print("\nPeriod Log columns:")
print(period.columns.tolist())

print("\nUser Profile columns:")
print(profile.columns.tolist())


print("\n========== CHRONOLOGICAL SORT CHECK ==========")

# Convert start_date into a proper date format
period["start_date"] = pd.to_datetime(
    period["start_date"],
    errors="coerce"
)

# Sort each user's cycles by date
period = period.sort_values(
    ["user_id", "start_date"]
).reset_index(drop=True)

# Display the first 15 rows
print(
    period[
        ["user_id", "cycle_number", "start_date"]
    ].head(15)
)

# Check whether dates are in chronological order within each user
date_differences = (
    period.groupby("user_id")["start_date"].diff()
)

print("\nNegative date differences:")
print((date_differences < pd.Timedelta(0)).sum())


print("\n========== CREATE NEXT HEALTH TARGET ==========")

period["next_health_score"] = (
    period.groupby("user_id")["overall_health_score"]
    .shift(-1)
)

print(
    period[
        [
            "user_id",
            "cycle_number",
            "overall_health_score",
            "next_health_score"
        ]
    ].head(15)
)

print("\nAvailable targets:")
print(period["next_health_score"].notna().sum())

print("\nMissing targets:")
print(period["next_health_score"].isna().sum())


print("\n========== CREATE HISTORICAL FEATURES ==========")

# Previous cycle's health score
period["previous_health_score"] = (
    period.groupby("user_id")["overall_health_score"]
    .shift(1)
)

# Previous cycle's length
period["previous_cycle_length_feature"] = (
    period.groupby("user_id")["cycle_length_days"]
    .shift(1)
)

print(
    period[
        [
            "user_id",
            "cycle_number",
            "overall_health_score",
            "previous_health_score",
            "cycle_length_days",
            "previous_cycle_length_feature",
            "next_health_score"
        ]
    ].head(15)
)


# Previous-cycle numerical features
previous_columns = {
    "previous_pain_level": "pain_level",
    "previous_stress_score": "stress_score_cycle",
    "previous_sleep_hours": "sleep_hours_cycle",
    "previous_mood_score": "mood_score",
    "previous_energy_level": "energy_level",
    "previous_concentration_score": "concentration_score",
    "previous_work_hours_lost": "work_hours_lost"
}

for new_column, original_column in previous_columns.items():
    period[new_column] = (
        period.groupby("user_id")[original_column].shift(1)
    )

print("\n========== PREVIOUS NUMERICAL FEATURES ==========")

print(
    period[
        [
            "user_id",
            "cycle_number",
            "previous_pain_level",
            "previous_stress_score",
            "previous_sleep_hours",
            "previous_mood_score",
            "previous_energy_level",
            "previous_concentration_score",
            "previous_work_hours_lost"
        ]
    ].head(15)
)


# Previous-cycle categorical features
period["previous_flow_level"] = (
    period.groupby("user_id")["flow_level"].shift(1)
)

period["previous_pms_symptoms"] = (
    period.groupby("user_id")["pms_symptoms"].shift(1)
)

print("\n========== PREVIOUS CATEGORICAL FEATURES ==========")

print(
    period[
        [
            "user_id",
            "cycle_number",
            "flow_level",
            "previous_flow_level",
            "pms_symptoms",
            "previous_pms_symptoms"
        ]
    ].head(15)
)


print("\n========== MERGE USER PROFILE ==========")

# Merge each user's profile with their cycle records
period = period.merge(
    profile,
    on="user_id",
    how="left",
    validate="many_to_one"
)

print("\nMerged dataset shape:")
print(period.shape)

print("\nMissing user IDs after merge:")
print(period["user_id"].isna().sum())

print("\nMissing profile values after merge:")
print(
    period[
        [
            "age",
            "bmi",
            "diet_quality",
            "exercise_frequency",
            "stress_score_baseline"
        ]
    ].isna().sum()
)


print("\n========== EXERCISE FREQUENCY INVESTIGATION ==========")

print(
    "Missing in original profile:",
    profile["exercise_frequency"].isna().sum()
)

print(
    "Users with missing exercise frequency:",
    profile.loc[
        profile["exercise_frequency"].isna(), "user_id"
    ].nunique()
)

print(
    "Missing in merged dataset:",
    period["exercise_frequency"].isna().sum()
)

print(
    "Unique users missing exercise frequency after merge:",
    period.loc[
        period["exercise_frequency"].isna(), "user_id"
    ].nunique()
)


print("\n========== FINAL MISSING VALUE CHECK ==========")

missing_counts = period.isna().sum()

print(
    missing_counts[missing_counts > 0].sort_values(
        ascending=False
    )
)


print("\n========== TARGET VALIDATION ==========")

print("Total rows:", len(period))

print(
    "Rows with a next-cycle target:",
    period["next_health_score"].notna().sum()
)

print(
    "Rows without a next-cycle target:",
    period["next_health_score"].isna().sum()
)

print(
    "Unique users without a next-cycle target:",
    period.loc[
        period["next_health_score"].isna(), "user_id"
    ].nunique()
)


print("\n========== CREATE TRAINING DATA COPY ==========")

# Keep only rows where the next-cycle target exists
training_data = period.dropna(
    subset=["next_health_score"]
).copy()

print("Original rows:", len(period))
print("Training-data rows:", len(training_data))
print(
    "Missing target values in training data:",
    training_data["next_health_score"].isna().sum()
)

print(
    "Unique users in training data:",
    training_data["user_id"].nunique()
)


print("\n========== TRAINING DATA MISSING VALUES ==========")

missing_training = training_data.isna().sum()

print(
    missing_training[missing_training > 0].sort_values(
        ascending=False
    )
)


print("\n========== EXERCISE FREQUENCY IN TRAINING DATA ==========")

print(
    "Missing exercise-frequency rows:",
    training_data["exercise_frequency"].isna().sum()
)

print(
    "Unique users with missing exercise frequency:",
    training_data.loc[
        training_data["exercise_frequency"].isna(),
        "user_id"
    ].nunique()
)

print(
    "Unique users in training data:",
    training_data["user_id"].nunique()
)


print("\n========== EXERCISE FREQUENCY CATEGORIES ==========")

print(
    profile["exercise_frequency"].value_counts(
        dropna=False
    )
)


print("\n========== FIRST CYCLE VALIDATION ==========")

historical_columns = [
    "previous_health_score",
    "previous_cycle_length_feature",
    "previous_pain_level",
    "previous_stress_score",
    "previous_sleep_hours",
    "previous_mood_score",
    "previous_energy_level",
    "previous_concentration_score",
    "previous_work_hours_lost",
    "previous_flow_level",
    "previous_pms_symptoms"
]

first_cycles = training_data[
    training_data["cycle_number"] == 1
]

print("First-cycle rows:", len(first_cycles))

print("\nMissing values in first cycles:")
print(first_cycles[historical_columns].isna().sum())

print("\nMissing values in later cycles:")
print(
    training_data.loc[
        training_data["cycle_number"] > 1,
        historical_columns
    ].isna().sum()
)


print("\n========== FIRST CYCLE + EXERCISE MISSING CHECK ==========")

print(
    "First-cycle rows:",
    (training_data["cycle_number"] == 1).sum()
)

print(
    "First-cycle rows with missing exercise frequency:",
    training_data.loc[
        training_data["cycle_number"] == 1,
        "exercise_frequency"
    ].isna().sum()
)

print(
    "Later-cycle rows with missing exercise frequency:",
    training_data.loc[
        training_data["cycle_number"] > 1,
        "exercise_frequency"
    ].isna().sum()
)


print("\n========== TARGET SCORE DISTRIBUTION ==========")

print("\nTarget summary:")
print(training_data["next_health_score"].describe())

print("\nTarget missing values:")
print(training_data["next_health_score"].isna().sum())

print("\nTarget score range:")
print("Minimum:", training_data["next_health_score"].min())
print("Maximum:", training_data["next_health_score"].max())


print("\n========== TRAINING DATA DUPLICATE CHECK ==========")

print(
    "Duplicate user-cycle records:",
    training_data.duplicated(
        subset=["user_id", "cycle_number"]
    ).sum()
)

print(
    "Duplicate complete rows:",
    training_data.duplicated().sum()
)


print("\n========== TRAINING DATA TYPES ==========")

print(training_data.dtypes.to_string())


print("\n========== SEPARATE FEATURES AND TARGET ==========")

# Target: next-cycle health score
y = training_data["next_health_score"].copy()

# Exclude identifiers, target, and date from model inputs
X = training_data.drop(
    columns=[
        "user_id",
        "next_health_score",
        "start_date"
    ]
).copy()

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget column:")
print(y.name)

print("\nUser ID excluded from features:", "user_id" not in X.columns)
print("\nTarget excluded from features:", "next_health_score" not in X.columns)


print("\n========== INPUT FEATURE REVIEW ==========")

for i, column in enumerate(X.columns, start=1):
    print(f"{i}. {column} ({X[column].dtype})")

print("\nTotal input features:", X.shape[1])

print(
    "\nCurrent health score included:",
    "overall_health_score" in X.columns
)

print(
    "Next health target included:",
    "next_health_score" in X.columns
)


print("\n========== TARGET LEAKAGE REVIEW ==========")

possible_leakage_columns = [
    "next_health_score",
    "overall_health_score",
    "health_score",
    "next_cycle_health_score"
]

for column in possible_leakage_columns:
    print(f"{column}: {column in X.columns}")

print("\nCurrent health score correlation with target:")
print(
    training_data[
        ["overall_health_score", "next_health_score"]
    ].corr().iloc[0, 1]
)


print("\n========== HEALTH SCORE FORMULA CHECK ==========")

formula_columns = [
    "overall_health_score",
    "pain_level",
    "stress_score_cycle",
    "work_hours_lost",
    "concentration_score",
    "energy_level",
    "mood_score",
    "sleep_hours_cycle"
]

print(
    training_data[formula_columns].corr()[
        "overall_health_score"
    ].sort_values()
)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("\n========== HEALTH SCORE RECONSTRUCTION CHECK ==========")

formula_features = [
    "pain_level",
    "work_hours_lost",
    "stress_score_cycle",
    "concentration_score",
    "energy_level",
    "mood_score",
    "sleep_hours_cycle"
]

formula_data = training_data[
    formula_features + ["overall_health_score"]
].dropna()

X_formula = formula_data[formula_features]
y_formula = formula_data["overall_health_score"]

formula_model = LinearRegression()
formula_model.fit(X_formula, y_formula)

formula_predictions = formula_model.predict(X_formula)

print("Rows checked:", len(formula_data))
print("R2 score:", r2_score(y_formula, formula_predictions))

print("\nCoefficients:")
for name, coefficient in zip(
    formula_features, formula_model.coef_
):
    print(f"{name}: {coefficient:.4f}")

print("Intercept:", round(formula_model.intercept_, 4))


print("\n========== HISTORICAL MODEL FEATURE SET ==========")

excluded_current_cycle = [
    "overall_health_score",
    "pain_level",
    "stress_score_cycle",
    "sleep_hours_cycle",
    "mood_score",
    "energy_level",
    "concentration_score",
    "work_hours_lost",
    "flow_level",
    "pms_symptoms",
    "estrogen_pgml",
    "progesterone_ngml",
    "ovulation_result",
    "log_consistency_score",
    "prepared_before_period"
]

X_historical = X.drop(
    columns=excluded_current_cycle,
    errors="ignore"
).copy()

print("Original feature count:", X.shape[1])
print("Historical model feature count:", X_historical.shape[1])

print("\nFeatures retained:")
print(X_historical.columns.tolist())


print("\n========== DUPLICATE FEATURE CHECK ==========")

print(
    "prev_cycle_length equals previous_cycle_length_feature:",
    X_historical["prev_cycle_length"].equals(
        X_historical["previous_cycle_length_feature"]
    )
)

print("\nFirst 10 values:")

print(
    X_historical[
        [
            "prev_cycle_length",
            "previous_cycle_length_feature"
        ]
    ].head(10)
)


print("\n========== REMOVE DUPLICATE FEATURE ==========")

X_historical = X_historical.drop(
    columns=["prev_cycle_length"]
)

print("Updated feature count:", X_historical.shape[1])

print(
    "prev_cycle_length removed:",
    "prev_cycle_length" not in X_historical.columns
)

print(
    "previous_cycle_length_feature retained:",
    "previous_cycle_length_feature" in X_historical.columns
)


print("\n========== IDENTICAL FEATURE CHECK ==========")

duplicate_feature_pairs = []

columns = X_historical.columns.tolist()

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        if X_historical[columns[i]].equals(
            X_historical[columns[j]]
        ):
            duplicate_feature_pairs.append(
                (columns[i], columns[j])
            )

if duplicate_feature_pairs:
    print("Identical feature pairs:")
    for pair in duplicate_feature_pairs:
        print(pair)
else:
    print("No identical feature pairs found.")

print("Total features checked:", X_historical.shape[1])


print("\n========== NUMERICAL VS CATEGORICAL FEATURES ==========")

numerical_features = X_historical.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_historical.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)
print("Count:", len(numerical_features))

print("\nCategorical features:")
print(categorical_features)
print("Count:", len(categorical_features))

print("\nTotal features:", len(numerical_features) + len(categorical_features))


print("\n========== SELECTED FEATURE MISSING VALUES ==========")

missing_features = X_historical.isna().sum()
missing_features = missing_features[missing_features > 0]

if missing_features.empty:
    print("No missing values found.")
else:
    print(missing_features.sort_values(ascending=False))

print("\nTotal rows:", len(X_historical))
print("Total selected features:", X_historical.shape[1])


from sklearn.model_selection import GroupShuffleSplit

print("\n========== USER-WISE TRAIN TEST SPLIT ==========")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X_historical,
        y,
        groups=training_data["user_id"]
    )
)

X_train = X_historical.iloc[train_idx].copy()
X_test = X_historical.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

train_users = set(training_data.iloc[train_idx]["user_id"])
test_users = set(training_data.iloc[test_idx]["user_id"])

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training users:", len(train_users))
print("Testing users:", len(test_users))
print("Users shared between sets:", len(train_users & test_users))


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("\n========== CREATE PREPROCESSING PIPELINE ==========")

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))
print("Preprocessing pipeline created successfully.")


print("\n========== FIT AND TRANSFORM DATA ==========")

# Learn preprocessing rules from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same rules to test data
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print(
    "Missing values in processed training data:",
    __import__("numpy").isnan(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed).sum()
)

print(
    "Missing values in processed testing data:",
    __import__("numpy").isnan(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed).sum()
)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n========== TRAIN BASELINE MODEL ==========")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

train_predictions = model.predict(X_train_processed)
test_predictions = model.predict(X_test_processed)

print("\nTraining Performance:")
print("MAE:", round(mean_absolute_error(y_train, train_predictions), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_train, train_predictions)), 4))
print("R2:", round(r2_score(y_train, train_predictions), 4))

print("\nTesting Performance:")
print("MAE:", round(mean_absolute_error(y_test, test_predictions), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, test_predictions)), 4))
print("R2:", round(r2_score(y_test, test_predictions), 4))


print("\n========== TOP 15 FEATURE IMPORTANCES ==========")

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print(importance_df.head(15).to_string(index=False))


from sklearn.dummy import DummyRegressor

print("\n========== DUMMY BASELINE COMPARISON ==========")

dummy_model = DummyRegressor(strategy="mean")

dummy_model.fit(X_train_processed, y_train)

dummy_predictions = dummy_model.predict(X_test_processed)

print(
    "Dummy MAE:",
    round(mean_absolute_error(y_test, dummy_predictions), 4)
)

print(
    "Dummy RMSE:",
    round(np.sqrt(mean_squared_error(y_test, dummy_predictions)), 4)
)

print(
    "Dummy R2:",
    round(r2_score(y_test, dummy_predictions), 4)
)

print("\nRandom Forest Test Performance:")
print("MAE:", round(mean_absolute_error(y_test, test_predictions), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, test_predictions)), 4))
print("R2:", round(r2_score(y_test, test_predictions), 4))


print("\n========== TUNED RANDOM FOREST ==========")

tuned_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

tuned_model.fit(X_train_processed, y_train)

tuned_train_predictions = tuned_model.predict(X_train_processed)
tuned_test_predictions = tuned_model.predict(X_test_processed)

print("\nTraining Performance:")
print("MAE:", round(mean_absolute_error(y_train, tuned_train_predictions), 4))
print("R2:", round(r2_score(y_train, tuned_train_predictions), 4))

print("\nTesting Performance:")
print("MAE:", round(mean_absolute_error(y_test, tuned_test_predictions), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, tuned_test_predictions)), 4))
print("R2:", round(r2_score(y_test, tuned_test_predictions), 4))


print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame({
    "Model": ["Original Random Forest", "Tuned Random Forest"],
    "Test MAE": [
        mean_absolute_error(y_test, test_predictions),
        mean_absolute_error(y_test, tuned_test_predictions)
    ],
    "Test RMSE": [
        np.sqrt(mean_squared_error(y_test, test_predictions)),
        np.sqrt(mean_squared_error(y_test, tuned_test_predictions))
    ],
    "Test R2": [
        r2_score(y_test, test_predictions),
        r2_score(y_test, tuned_test_predictions)
    ]
})

print(comparison.round(4).to_string(index=False))


from sklearn.linear_model import Ridge

print("\n========== RIDGE REGRESSION ==========")

ridge_model = Ridge(alpha=10.0)

ridge_model.fit(X_train_processed, y_train)

ridge_train_predictions = ridge_model.predict(X_train_processed)
ridge_test_predictions = ridge_model.predict(X_test_processed)

print("\nTraining Performance:")
print("MAE:", round(mean_absolute_error(y_train, ridge_train_predictions), 4))
print("R2:", round(r2_score(y_train, ridge_train_predictions), 4))

print("\nTesting Performance:")
print("MAE:", round(mean_absolute_error(y_test, ridge_test_predictions), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, ridge_test_predictions)), 4))
print("R2:", round(r2_score(y_test, ridge_test_predictions), 4))


print("\n========== FINAL MODEL COMPARISON ==========")

comparison = pd.DataFrame({
    "Model": [
        "Original Random Forest",
        "Tuned Random Forest",
        "Ridge Regression"
    ],
    "Test MAE": [
        mean_absolute_error(y_test, test_predictions),
        mean_absolute_error(y_test, tuned_test_predictions),
        mean_absolute_error(y_test, ridge_test_predictions)
    ],
    "Test RMSE": [
        np.sqrt(mean_squared_error(y_test, test_predictions)),
        np.sqrt(mean_squared_error(y_test, tuned_test_predictions)),
        np.sqrt(mean_squared_error(y_test, ridge_test_predictions))
    ],
    "Test R2": [
        r2_score(y_test, test_predictions),
        r2_score(y_test, tuned_test_predictions),
        r2_score(y_test, ridge_test_predictions)
    ]
})

print(comparison.round(4).to_string(index=False))


from sklearn.model_selection import GroupKFold, cross_validate

# print("\n========== RIDGE GROUP CROSS-VALIDATION ==========")

# ridge_cv_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", Ridge(alpha=10.0))
# ])

# groups = training_data.iloc[train_idx]["user_id"]

# cv = GroupKFold(n_splits=5)

# cv_results = cross_validate(
#     ridge_cv_pipeline,
#     X_historical.iloc[train_idx],
#     y_train,
#     groups=groups,
#     cv=cv,
#     scoring={
#         "mae": "neg_mean_absolute_error",
#         "r2": "r2"
#     },
#     n_jobs=-1
# )

# print(
#     "Fold MAE:",
#     (-cv_results["test_mae"]).round(4)
# )

# print(
#     "Mean CV MAE:",
#     round((-cv_results["test_mae"]).mean(), 4)
# )

# print(
#     "Mean CV R2:",
#     round(cv_results["test_r2"].mean(), 4)
# )


import joblib
import os

print("\n========== SAVE HERAI MODEL ==========")

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Fit the final pipeline on the training split
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=10.0))
])

final_model.fit(X_train, y_train)

# Save preprocessing and model together
joblib.dump(
    final_model,
    "models/herai_ridge_model.pkl"
)

print("Model saved successfully!")
print("Path: models/herai_ridge_model.pkl")


print("\n========== LOAD AND TEST SAVED MODEL ==========")

# Load the saved pipeline
loaded_model = joblib.load(
    "models/herai_ridge_model.pkl"
)


