
import pandas as pd
import joblib
from pathlib import Path

print("\n========== HERAI HEALTH SCORE PREDICTION ==========")

# Locate the project folder
ROOT = Path(__file__).resolve().parent.parent

# Load the saved model
model_path = ROOT / "models" / "herai_ridge_model.pkl"
model = joblib.load(model_path)

print("HERAI model loaded successfully!")

# Load one existing record as an example input
data_path = ROOT / "data" / "processed" / "engineered_data.csv"
df = pd.read_csv(data_path)

# Select a record that has a previous cycle
sample = df[df["previous_health_score"].notna()].iloc[[0]].copy()

# Remove columns that were not used as model inputs
excluded_columns = [
    "user_id",
    "start_date",
    "next_health_score",
    "cycle_number"
]

X_new = sample.drop(columns=excluded_columns)

# Predict the next-cycle score
prediction = model.predict(X_new)[0]

print("\n========== PREDICTION RESULT ==========")
print("Current cycle health score:",
      sample["overall_health_score"].iloc[0])
print("Predicted next-cycle health score:", round(prediction, 2))



print("\n========== INPUT FEATURES REQUIRED ==========")

expected_features = model.named_steps[
    "preprocessor"
].feature_names_in_

for index, feature in enumerate(expected_features, start=1):
    print(f"{index}. {feature}")

print("\nTotal required input features:", len(expected_features))



print("\n========== HERAI USER INPUT ==========")

# Collect basic profile information
age = float(input("Enter your age: "))
bmi = float(input("Enter your BMI: "))
sleep_hours = float(input("Average daily sleep hours: "))
stress_score_baseline = float(
    input("Baseline stress score: ")
)

# Collect current-cycle information
cycle_length_days = float(
    input("Current cycle length in days: ")
)
pain_level = float(input("Current pain level: "))
mood_score = float(input("Current mood score: "))
stress_score_cycle = float(
    input("Current cycle stress score: ")
)
sleep_hours_cycle = float(
    input("Sleep hours during current cycle: ")
)
energy_level = float(input("Current energy level: "))
concentration_score = float(
    input("Current concentration score: ")
)
overall_health_score = float(
    input("Current overall health score: ")
)

print("\n========== INFORMATION RECEIVED ==========")
print("Age:", age)
print("BMI:", bmi)
print("Average sleep:", sleep_hours)
print("Baseline stress:", stress_score_baseline)
print("Cycle length:", cycle_length_days)
print("Pain level:", pain_level)
print("Mood score:", mood_score)
print("Cycle stress:", stress_score_cycle)
print("Cycle sleep:", sleep_hours_cycle)
print("Energy level:", energy_level)
print("Concentration score:", concentration_score)
print("Current health score:", overall_health_score)

