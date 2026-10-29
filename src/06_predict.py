"""HERAI Health Score Prediction Script.

Demonstrates model prediction on a test record and provides a programmatic
inference entry point using the trained 37-feature Ridge pipeline.
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herai.config import (
    MODEL_RIDGE_37_PATH,
    ENGINEERED_DATA_PATH,
    FEATURES_37,
)
from herai.models import load_model
from herai.inference import (
    HerAIPredictor,
    UserProfileInput,
    CurrentCycleInput,
    HistoricalCycleInput,
)


def run_sample_prediction() -> None:
    """Load a sample cycle from engineered data and predict next health score."""
    print("\n========== HERAI HEALTH SCORE PREDICTION ==========")
    print(f"Loading model from: {MODEL_RIDGE_37_PATH}")

    model = load_model(MODEL_RIDGE_37_PATH)
    print("HERAI 37-feature model loaded successfully!")

    # Load one verified record that has historical cycle data
    if not ENGINEERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_DATA_PATH}. Run src/03_create_features.py first."
        )

    df = pd.read_csv(ENGINEERED_DATA_PATH)
    valid_samples = df[df["previous_health_score"].notna()]
    if valid_samples.empty:
        raise ValueError("No records with historical lag features found in engineered_data.csv.")

    sample = valid_samples.iloc[[0]].copy()
    user_id = sample["user_id"].iloc[0]
    cycle_num = sample["cycle_number"].iloc[0]
    actual_current = sample["overall_health_score"].iloc[0]
    actual_next = sample["next_health_score"].iloc[0]

    # Predict using the 37 features
    X_sample = sample[FEATURES_37]
    prediction = float(model.predict(X_sample)[0])

    print("\n========== SAMPLE DATA PREDICTION ==========")
    print(f"User ID: {user_id} (Cycle #{cycle_num})")
    print(f"Current cycle health score: {actual_current:.2f}")
    print(f"Predicted next-cycle health score: {prediction:.2f}")
    if pd.notna(actual_next):
        print(f"Actual next-cycle health score: {actual_next:.2f}")
        print(f"Absolute error: {abs(prediction - actual_next):.2f}")


def run_interactive_prediction() -> None:
    """Prompt user for clinical inputs and compute prediction without stranger data."""
    print("\n========== INTERACTIVE USER PREDICTION ==========")
    predictor = HerAIPredictor()

    def ask(prompt: str, min_val: float, max_val: float, default: float) -> float:
        try:
            val_str = input(f"{prompt} [default: {default}]: ").strip()
            if not val_str:
                return default
            val = float(val_str)
            if val < min_val or val > max_val:
                print(f"Out of range [{min_val}, {max_val}]. Using default {default}.")
                return default
            return val
        except (ValueError, EOFError):
            return default

    # Demographic profile
    age = ask("Enter your age (12-100)", 12, 100, 28)
    bmi = ask("Enter your BMI (10-70)", 10, 70, 22.5)
    sleep_hours = ask("Average baseline sleep hours (0-24)", 0, 24, 7.5)
    stress_base = ask("Baseline stress score (0-10)", 0, 10, 4.0)

    # Current cycle
    cycle_length = ask("Current cycle length in days (15-60)", 15, 60, 28)
    pain = ask("Current cycle pain level (0-10)", 0, 10, 3.0)
    mood = ask("Current cycle mood score (0-10)", 0, 10, 7.0)
    stress_cycle = ask("Current cycle stress score (0-10)", 0, 10, 4.0)
    sleep_cycle = ask("Current cycle sleep hours (0-24)", 0, 24, 7.5)
    energy = ask("Current cycle energy level (0-10)", 0, 10, 7.0)
    concentration = ask("Current cycle concentration score (0-10)", 0, 10, 7.0)
    current_health = ask("Current cycle overall health score (0-10)", 0, 10, 8.5)

    profile = UserProfileInput(
        age=age,
        bmi=bmi,
        sleep_hours=sleep_hours,
        stress_score_baseline=stress_base,
    )
    current_cycle = CurrentCycleInput(
        cycle_length_days=cycle_length,
        pain_level=pain,
        mood_score=mood,
        stress_score_cycle=stress_cycle,
        sleep_hours_cycle=sleep_cycle,
        energy_level=energy,
        concentration_score=concentration,
        overall_health_score=current_health,
    )

    result = predictor.predict(profile, current_cycle)

    print("\n========== PREDICTION RESULT ==========")
    print(f"Predicted Next-Cycle Health Score: {result.predicted_next_health_score} / 10.0")
    print(f"Category: {result.score_category}")
    print(f"Summary: {result.summary}")
    for warning in result.warnings:
        print(f"Note: {warning}")
    print(f"\nDisclaimer: {result.disclaimer}")


if __name__ == "__main__":
    # If run in non-interactive environment, execute sample prediction
    run_sample_prediction()
    if sys.stdin.isatty():
        try:
            choice = input("\nWould you like to enter your own inputs? (y/N): ").strip().lower()
            if choice == "y":
                run_interactive_prediction()
        except EOFError:
            pass
