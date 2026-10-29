"""HERAI Data Preparation Pipeline.

Filters records where the next-cycle target is known and prepares training data.
Saves the prepared training dataset to data/processed/training_data.csv.
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herai.config import (
    ENGINEERED_DATA_PATH,
    TRAINING_DATA_PATH,
    TARGET_COLUMN,
)
from herai.data import prepare_training_dataset


def run_data_preparation() -> pd.DataFrame:
    """Prepare the training dataset by filtering records with known next-cycle targets.
    
    Imputation policy:
    1. Historical lag numerical features missing due to cycle 1 are imputed with training medians.
    2. Categorical lag features missing due to cycle 1 are marked as 'Unknown'.
    3. Missing exercise_frequency values in user profiles are filled with the mode.
    
    Note: For model training pipelines, the scikit-learn Pipeline (SimpleImputer) is
    used to fit imputations within training folds to guarantee zero data leakage.
    
    Returns:
        pd.DataFrame: Cleaned training dataset.
    """
    print("\n========== HERAI DATA PREPARATION ==========")
    print(f"Loading engineered dataset from: {ENGINEERED_DATA_PATH}")

    if not ENGINEERED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {ENGINEERED_DATA_PATH}. Run src/03_create_features.py first."
        )

    period = pd.read_csv(ENGINEERED_DATA_PATH)
    print("Loaded dataset shape:", period.shape)

    # Filter records with available next_health_score
    training_data = prepare_training_dataset(period)
    print("After removing missing targets:", training_data.shape)

    # Impute missing numerical historical features
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
        "previous_work_hours_lost",
    ]

    for column in historical_numeric:
        if column in training_data.columns:
            median_val = training_data[column].median()
            training_data[column] = training_data[column].fillna(median_val)

    # Impute missing categorical historical features
    historical_categorical = [
        "previous_flow_level",
        "previous_pms_symptoms",
    ]
    for column in historical_categorical:
        if column in training_data.columns:
            training_data[column] = training_data[column].fillna("Unknown")

    # Impute exercise frequency if present and missing
    if "exercise_frequency" in training_data.columns:
        modes = training_data["exercise_frequency"].mode()
        exercise_mode = modes[0] if not modes.empty else "3–4 days/week"
        training_data["exercise_frequency"] = training_data["exercise_frequency"].fillna(exercise_mode)

    print("\nTotal missing values in prepared training data:", training_data.isnull().sum().sum())

    # Save prepared training dataset
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(TRAINING_DATA_PATH, index=False)

    print("Prepared dataset saved to:", TRAINING_DATA_PATH)
    print("Final training dataset shape:", training_data.shape)
    return training_data


if __name__ == "__main__":
    run_data_preparation()
