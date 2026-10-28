
import pandas as pd
import joblib
from pathlib import Path

print("\n========== HERAI USER INPUT ==========")


def get_number(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))

            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")

# Collect user inputs
# Collect user inputs with basic validation
age = get_number("Enter your age: ", 1, 120)
bmi = get_number("Enter your BMI: ", 1, 100)
sleep_hours = get_number("Average daily sleep hours (0-24): ", 0, 24)
stress_score_baseline = get_number("Baseline stress score (0-10): ", 0, 10)

cycle_length_days = get_number("Current cycle length in days (15-60): ", 15, 60)
pain_level = get_number("Current pain level (0-10): ", 0, 10)
mood_score = get_number("Current mood score (0-10): ", 0, 10)
stress_score_cycle = get_number("Current cycle stress score (0-10): ", 0, 10)
sleep_hours_cycle = get_number("Sleep hours during current cycle (0-24): ", 0, 24)
energy_level = get_number("Current energy level (0-10): ", 0, 10)
concentration_score = get_number("Current concentration score (0-10): ", 0, 10)
overall_health_score = get_number("Current overall health score (0-10): ", 0, 10)

ROOT = Path(__file__).resolve().parent.parent

# Load the 37-feature model
model_path = ROOT / "models" / "herai_ridge_37_features.pkl"
model = joblib.load(model_path)

# Load a template row for features not collected yet
data_path = ROOT / "data" / "processed" / "engineered_data.csv"
df = pd.read_csv(data_path)

valid_rows = df[df["previous_health_score"].notna()]
if valid_rows.empty:
    raise ValueError("No suitable template row found in engineered data.")

sample = valid_rows.iloc[[0]].copy()

# Replace the features entered by the user
user_values = {
    "age": age,
    "bmi": bmi,
    "sleep_hours": sleep_hours,
    "stress_score_baseline": stress_score_baseline,
    "cycle_length_days": cycle_length_days,
    "pain_level": pain_level,
    "mood_score": mood_score,
    "stress_score_cycle": stress_score_cycle,
    "sleep_hours_cycle": sleep_hours_cycle,
    "energy_level": energy_level,
    "concentration_score": concentration_score,
    "overall_health_score": overall_health_score
}

for column, value in user_values.items():
    sample.loc[:, column] = value

# Select the exact 37 features used by the reduced model
excluded_columns = [
    "user_id",
    "start_date",
    "next_health_score",
    "cycle_number",
    "estrogen_pgml",
    "progesterone_ngml",
    "ovulation_result",
    "log_consistency_score",
    "prepared_before_period"
]

X_new = sample.drop(columns=excluded_columns, errors="ignore")

# Predict the next-cycle score
prediction = model.predict(X_new)[0]

print("\n========== HERAI PREDICTION ==========")
print("Predicted next-cycle health score:", round(prediction, 2))

print("\nIMPORTANT:")
print("This is a prototype demonstration, not a validated")
print("personalized health prediction.")
print("Some features still come from a sample dataset row.")
print("Do not use this score for medical decisions.")
21
