
import pandas as pd
import os

print("\n========== HERAI FEATURE CREATION ==========")

# Load the merged dataset
data_path = "data/processed/merged_data.csv"

period = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", period.shape)
print("Columns:")
print(period.columns.tolist())



print("\n========== SORT CYCLE RECORDS ==========")

# Convert start_date to datetime format
period["start_date"] = pd.to_datetime(
    period["start_date"],
    errors="coerce"
)

# Sort each user's cycles chronologically
period = period.sort_values(
    by=["user_id", "start_date"]
).reset_index(drop=True)

print("Cycle records sorted successfully!")

print(
    period[
        ["user_id", "cycle_number", "start_date"]
    ].head(15)
)


print("Invalid dates:", period["start_date"].isna().sum())
print("Duplicate user-date rows:", period.duplicated(
    subset=["user_id", "start_date"]
).sum())



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

# Previous cycle's numerical features
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
        period.groupby("user_id")[original_column]
        .shift(1)
    )

# Previous cycle's categorical features
previous_categorical = {
    "previous_flow_level": "flow_level",
    "previous_pms_symptoms": "pms_symptoms"
}

for new_column, original_column in previous_categorical.items():
    period[new_column] = (
        period.groupby("user_id")[original_column]
        .shift(1)
    )

print("Historical features created successfully!")

print(
    period[
        [
            "user_id",
            "cycle_number",
            "overall_health_score",
            "previous_health_score",
            "cycle_length_days",
            "previous_cycle_length_feature",
            "stress_score_cycle",
            "previous_stress_score"
        ]
    ].head(15)
)


print("\n========== VERIFY USER PROFILE FEATURES ==========")

profile_columns = [
    "age",
    "bmi",
    "diet_quality",
    "exercise_frequency",
    "sleep_hours",
    "caffeine_intake",
    "water_intake_liters",
    "alcohol_consumption",
    "smoking_status",
    "birth_control_use",
    "pcos_diagnosed",
    "stress_score_baseline",
    "state"
]

# Check which profile columns exist
missing_columns = [
    col for col in profile_columns
    if col not in period.columns
]

if missing_columns:
    print("Missing profile columns:", missing_columns)
else:
    print("All expected profile columns are present!")

# Check missing values in profile columns
print("\nMissing profile values:")
print(period[profile_columns].isnull().sum())

# Display a few profile records
print("\nProfile feature sample:")
print(
    period[
        ["user_id", "age", "bmi", "diet_quality",
         "exercise_frequency", "stress_score_baseline"]
    ].head(10)
)



print("\n========== CREATE PREDICTION TARGET ==========")

# Target = next cycle's health score for the same user
period["next_health_score"] = (
    period.groupby("user_id")["overall_health_score"]
    .shift(-1)
)

print("Next-cycle target created successfully!")

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

print(
    "\nMissing next-cycle targets:",
    period["next_health_score"].isna().sum()
)


print("\n========== VERIFY MISSING TARGETS ==========")

print("Total users:", period["user_id"].nunique())

print(
    "Users whose last cycle has a missing target:",
    period.groupby("user_id")["next_health_score"]
    .apply(lambda x: x.isna().sum())
    .sum()
)

print(
    "Rows with missing targets:",
    period["next_health_score"].isna().sum()
)

print("\n========== PREPARE TRAINING DATA ==========")

training_data = period.dropna(
    subset=["next_health_score"]
).copy()

print("Original dataset shape:", period.shape)
print("Training dataset shape:", training_data.shape)

print(
    "Missing targets remaining:",
    training_data["next_health_score"].isna().sum()
)

print("\n========== CHECK MISSING FEATURE VALUES ==========")

missing_values = training_data.isnull().sum()

print(
    missing_values[missing_values > 0]
    .sort_values(ascending=False)
)

print(
    "\nTotal columns with missing values:",
    (missing_values > 0).sum()
)

print("\n========== HANDLE MISSING PROFILE VALUES ==========")

exercise_mode = training_data["exercise_frequency"].mode()[0]

training_data["exercise_frequency"] = (
    training_data["exercise_frequency"].fillna(exercise_mode)
)

print("Filled missing exercise_frequency values with:", exercise_mode)

print(
    "Remaining missing exercise_frequency values:",
    training_data["exercise_frequency"].isna().sum()
)

print("\n========== HANDLE MISSING HISTORICAL FEATURES ==========")

historical_numeric = [
    "prev_cycle_length",
    "previous_health_score",
    "previous_cycle_length_feature",
    "previous_pain_level",
    "previous_stress_score",
    "previous_sleep_hours",
    "previous_mood_score",
    "previous_energy_level",
    "previous_concentration_score",
    "previous_work_hours_lost"
]

for column in historical_numeric:
    median_value = training_data[column].median()
    training_data[column] = training_data[column].fillna(median_value)

print("Missing historical numerical values filled.")

print(
    training_data[historical_numeric]
    .isnull()
    .sum()
)


print("\n========== HANDLE MISSING CATEGORICAL FEATURES ==========")

historical_categorical = [
    "previous_flow_level",
    "previous_pms_symptoms"
]

for column in historical_categorical:
    training_data[column] = training_data[column].fillna("Unknown")

print("Missing categorical historical values filled.")

print(training_data[historical_categorical].isnull().sum())

print("\n========== FINAL MISSING VALUE CHECK ==========")

missing_values = training_data.isnull().sum()

print("Total missing values:", missing_values.sum())

if missing_values.sum() == 0:
    print("All missing values have been handled!")
else:
    print(missing_values[missing_values > 0])


    print("\n========== SAVE ENGINEERED DATA ==========")

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "engineered_data.csv"
)

period.to_csv(output_path, index=False)

print("Engineered dataset saved to:", output_path)