"""HERAI Interactive User Input & Prediction.

Collects user-specific demographic and cycle measurements, validates them against
physiological ranges, and generates personalized next-cycle health predictions
WITHOUT borrowing or copying data from other users.
"""

from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herai.inference import (
    HerAIPredictor,
    UserProfileInput,
    CurrentCycleInput,
    HistoricalCycleInput,
    InputValidationError,
)


def get_validated_number(
    prompt: str,
    min_value: float,
    max_value: float,
    default_value: Optional[float] = None,
) -> float:
    """Prompt the user for a numerical input and validate against bounds.
    
    In non-interactive environments, returns default_value or midpoint.
    """
    if not sys.stdin.isatty() and default_value is not None:
        return default_value

    while True:
        default_str = f" [default: {default_value}]" if default_value is not None else ""
        try:
            val_str = input(f"{prompt}{default_str}: ").strip()
            if not val_str and default_value is not None:
                return default_value
            val = float(val_str)
            if val < min_value:
                print(f"Value must be at least {min_value}.")
                continue
            if val > max_value:
                print(f"Value must be at most {max_value}.")
                continue
            return val
        except (ValueError, EOFError):
            if default_value is not None:
                print(f"Using default value: {default_value}")
                return default_value
            print("Please enter a valid number.")


def run_user_input_prediction() -> None:
    """Run interactive prediction session."""
    print("\n========== HERAI USER INPUT & PREDICTION ==========")
    print("Please enter your personal health and cycle observations.")
    print("HERAI uses only your own provided data and population reference baselines.")

    # 1. Profile information
    print("\n--- 1. Demographic & Lifestyle Profile ---")
    age = get_validated_number("Enter your age (12-100)", 12, 100, 28.0)
    bmi = get_validated_number("Enter your BMI (10-70)", 10.0, 70.0, 22.5)
    sleep_hours = get_validated_number("Average daily sleep hours (0-24)", 0.0, 24.0, 7.5)
    stress_base = get_validated_number("Baseline stress score (0-10)", 0.0, 10.0, 4.0)

    # 2. Current cycle information
    print("\n--- 2. Current Menstrual Cycle Observations ---")
    cycle_length = get_validated_number("Current cycle length in days (15-60)", 15, 60, 28.0)
    pain = get_validated_number("Current pain level (0-10)", 0.0, 10.0, 3.0)
    mood = get_validated_number("Current mood score (0-10)", 0.0, 10.0, 7.0)
    stress_cycle = get_validated_number("Current cycle stress score (0-10)", 0.0, 10.0, 4.0)
    sleep_cycle = get_validated_number("Sleep hours during current cycle (0-24)", 0.0, 24.0, 7.5)
    energy = get_validated_number("Current energy level (0-10)", 0.0, 10.0, 7.0)
    concentration = get_validated_number("Current concentration score (0-10)", 0.0, 10.0, 7.0)
    overall_health = get_validated_number("Current overall health score (0-10)", 0.0, 10.0, 8.5)

    # 3. Optional previous cycle information
    hist_cycle: Optional[HistoricalCycleInput] = None
    has_history = False
    if sys.stdin.isatty():
        try:
            has_hist_str = input("\nDo you have data logged from your previous cycle? (y/N): ").strip().lower()
            has_history = has_hist_str == "y"
        except EOFError:
            has_history = False

    if has_history:
        print("\n--- 3. Previous Cycle Observations (Belonging to You) ---")
        prev_health = get_validated_number("Previous cycle overall health score (0-10)", 0.0, 10.0, overall_health)
        prev_cycle_len = get_validated_number("Previous cycle length in days (15-60)", 15, 60, cycle_length)
        prev_pain = get_validated_number("Previous cycle pain level (0-10)", 0.0, 10.0, pain)
        prev_stress = get_validated_number("Previous cycle stress score (0-10)", 0.0, 10.0, stress_cycle)
        prev_sleep = get_validated_number("Previous cycle sleep hours (0-24)", 0.0, 24.0, sleep_cycle)
        prev_mood = get_validated_number("Previous cycle mood score (0-10)", 0.0, 10.0, mood)
        prev_energy = get_validated_number("Previous cycle energy level (0-10)", 0.0, 10.0, energy)
        prev_concentration = get_validated_number("Previous cycle concentration score (0-10)", 0.0, 10.0, concentration)

        hist_cycle = HistoricalCycleInput(
            previous_health_score=prev_health,
            previous_cycle_length_feature=prev_cycle_len,
            previous_pain_level=prev_pain,
            previous_stress_score=prev_stress,
            previous_sleep_hours=prev_sleep,
            previous_mood_score=prev_mood,
            previous_energy_level=prev_energy,
            previous_concentration_score=prev_concentration,
            previous_work_hours_lost=0.0,
            previous_flow_level="Medium",
            previous_pms_symptoms="Mild",
        )

    # Initialize predictor
    predictor = HerAIPredictor()

    profile_obj = UserProfileInput(
        age=age,
        bmi=bmi,
        sleep_hours=sleep_hours,
        stress_score_baseline=stress_base,
    )
    cycle_obj = CurrentCycleInput(
        cycle_length_days=cycle_length,
        pain_level=pain,
        mood_score=mood,
        stress_score_cycle=stress_cycle,
        sleep_hours_cycle=sleep_cycle,
        energy_level=energy,
        concentration_score=concentration,
        overall_health_score=overall_health,
    )

    result = predictor.predict(profile_obj, cycle_obj, historical_cycle=hist_cycle)

    print("\n========== HERAI PREDICTION RESULT ==========")
    print(f"Predicted Next-Cycle Health Score: {result.predicted_next_health_score:.2f} / 10.0")
    print(f"Health Status Category: {result.score_category}")
    print(f"Summary: {result.summary}")

    if result.warnings:
        print("\nPipeline Notes:")
        for w in result.warnings:
            print(f"- {w}")

    print(f"\nMedical Disclaimer:\n{result.disclaimer}")


if __name__ == "__main__":
    run_user_input_prediction()
