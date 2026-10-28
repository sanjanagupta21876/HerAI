
import pandas as pd
from pathlib import Path

print("\n========== HERAI MODEL TRAINING ==========")

# Locate the project directory
ROOT = Path(__file__).resolve().parent.parent

# Load the engineered dataset
data_path = ROOT / "data" / "processed" / "engineered_data.csv"
df = pd.read_csv(data_path)

# Remove rows without the prediction target
df = df.dropna(subset=["next_health_score"]).copy()

print("Dataset shape:", df.shape)
print("Number of unique users:", df["user_id"].nunique())

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nTarget variable:")
print("next_health_score")

print("\nTarget summary:")
print(df["next_health_score"].describe())


print("\n========== SELECT INPUT FEATURES ==========")

target_column = "next_health_score"

excluded_columns = [
    "user_id",
    "start_date",
    "next_health_score",
    "cycle_number"
]

X = df.drop(columns=excluded_columns)
y = df[target_column]

print("Number of input features:", X.shape[1])
print("Number of target values:", y.shape[0])

print("\nInput feature columns:")
print(X.columns.tolist())


print("\n========== CHECK POSSIBLE TARGET LEAKAGE ==========")

print("Target:", target_column)
print("Current health score included:", "overall_health_score" in X.columns)
print("Next health score included:", "next_health_score" in X.columns)

print("\nPotentially time-sensitive features:")
for col in [
    "estrogen_pgml",
    "progesterone_ngml",
    "ovulation_result",
    "log_consistency_score",
    "prepared_before_period"
]:
    if col in X.columns:
        print("-", col)

print("\nTarget shape:", y.shape)



from sklearn.model_selection import GroupShuffleSplit

print("\n========== SPLIT DATA ==========")

# Keep all cycles of a user in the same split
groups = df["user_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("Unique users in training:", groups.iloc[train_indices].nunique())
print("Unique users in testing:", groups.iloc[test_indices].nunique())

# Verify that no user occurs in both sets
train_users = set(groups.iloc[train_indices])
test_users = set(groups.iloc[test_indices])

print("Users appearing in both sets:", len(train_users & test_users))



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

print("\n========== BUILD PREPROCESSING PIPELINE ==========")

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("Number of numerical features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))

# Process numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Process categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Combine preprocessing and Ridge Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

print("Preprocessing pipeline created successfully!")
print("Model: Ridge Regression")



print("\n========== TRAIN HERAI MODEL ==========")

# Train the complete pipeline using training data
model.fit(X_train, y_train)

print("Model training completed successfully!")

# Generate predictions for unseen users
y_pred = model.predict(X_test)

print("\nPredictions generated successfully!")
print("Number of predictions:", len(y_pred))

# Compare actual and predicted values
results = pd.DataFrame({
    "Actual_Next_Health_Score": y_test.values,
    "Predicted_Next_Health_Score": y_pred
})

print("\nSample predictions:")
print(results.head(10).round(2))



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n========== EVALUATE HERAI MODEL ==========")

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")



print("\n========== BASELINE COMPARISON ==========")

# Predict the average health score from the training set
baseline_prediction = np.full(
    shape=len(y_test),
    fill_value=y_train.mean()
)

baseline_mae = mean_absolute_error(y_test, baseline_prediction)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_prediction)
)
baseline_r2 = r2_score(y_test, baseline_prediction)

print(f"Baseline MAE:  {baseline_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline R2:   {baseline_r2:.4f}")

print("\n========== PERFORMANCE COMPARISON ==========")
print(f"Ridge MAE:     {mae:.4f}")
print(f"Baseline MAE:  {baseline_mae:.4f}")
print(f"Ridge RMSE:    {rmse:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Ridge R2:      {r2:.4f}")
print(f"Baseline R2:   {baseline_r2:.4f}")



import joblib

print("\n========== SAVE HERAI MODEL ==========")

model_dir = ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "herai_ridge_model.pkl"

joblib.dump(model, model_path)

print("Model saved successfully!")
print("Saved location:", model_path)


print("\n========== TOP MODEL COEFFICIENTS ==========")

# Get the preprocessing component and trained Ridge model
preprocessor_fitted = model.named_steps["preprocessor"]
ridge_model = model.named_steps["regressor"]

# Get feature names after preprocessing
feature_names = preprocessor_fitted.get_feature_names_out()

# Create a table of coefficients
coefficient_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": ridge_model.coef_
})

# Sort by absolute coefficient magnitude
coefficient_table["Absolute_Coefficient"] = (
    coefficient_table["Coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\nTop 15 features by absolute coefficient:")
print(
    coefficient_table[
        ["Feature", "Coefficient"]
    ].head(15).to_string(index=False)
)
