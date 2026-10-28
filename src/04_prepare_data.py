
import pandas as pd
from pathlib import Path

print("\n========== HERAI DATA PREPARATION ==========")

# Get the project root directory
ROOT = Path(__file__).resolve().parent.parent

# Load engineered dataset
input_path = ROOT / "data" / "processed" / "engineered_data.csv"
period = pd.read_csv(input_path)

print("Loaded dataset shape:", period.shape)

# Remove rows without a prediction target
training_data = period.dropna(
    subset=["next_health_score"]
).copy()

print("After removing missing targets:", training_data.shape)

# Handle missing numerical historical features
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

# Handle missing categorical features
historical_categorical = [
    "previous_flow_level",
    "previous_pms_symptoms"
]

for column in historical_categorical:
    training_data[column] = training_data[column].fillna("Unknown")

# Handle missing exercise frequency
exercise_mode = training_data["exercise_frequency"].mode()[0]

training_data["exercise_frequency"] = (
    training_data["exercise_frequency"].fillna(exercise_mode)
)

# Verify missing values
print("\nTotal missing values:", training_data.isnull().sum().sum())

# Save prepared dataset
output_path = ROOT / "data" / "processed" / "training_data.csv"
training_data.to_csv(output_path, index=False)

print("Prepared dataset saved to:", output_path)
print("Final training dataset shape:", training_data.shape)
