"""HERAI Data Merging Pipeline.

Merges raw Period_Log.csv and User_Profile.csv safely into merged_data.csv
after validating schemas, unique keys, and join integrity.
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herai.config import (
    RAW_PERIOD_LOG_PATH,
    RAW_USER_PROFILE_PATH,
    MERGED_DATA_PATH,
)
from herai.data import validate_schemas, merge_datasets


def run_merge() -> pd.DataFrame:
    """Execute validation and merging of raw datasets.
    
    Returns:
        pd.DataFrame: Merged dataset containing cycle logs and user profiles.
    """
    print("\n========== HERAI DATA MERGING ==========")
    print(f"Loading period log from: {RAW_PERIOD_LOG_PATH}")
    print(f"Loading user profile from: {RAW_USER_PROFILE_PATH}")

    period = pd.read_csv(RAW_PERIOD_LOG_PATH)
    profile = pd.read_csv(RAW_USER_PROFILE_PATH)

    print("\nOriginal dataset shapes:")
    print("Period log:", period.shape)
    print("User profile:", profile.shape)

    # Validate schema and unique constraints
    validate_schemas(period, profile)
    print("Schema validation successful: all required columns present and keys verified.")

    # Safe many-to-one merge
    merged_data = merge_datasets(period, profile)

    print("\n========== MERGE RESULTS ==========")
    print("Merged dataset shape:", merged_data.shape)
    print("Missing user IDs after merge:", merged_data["user_id"].isna().sum())

    # Profile completeness check
    profile_columns = [
        "age",
        "bmi",
        "diet_quality",
        "exercise_frequency",
        "stress_score_baseline",
    ]
    available_columns = [col for col in profile_columns if col in merged_data.columns]
    print("\nMissing profile values after merge:")
    print(merged_data[available_columns].isna().sum())

    # Save to data/processed/merged_data.csv
    MERGED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_data.to_csv(MERGED_DATA_PATH, index=False)

    print("\nMerged dataset saved successfully!")
    print("Path:", MERGED_DATA_PATH)
    return merged_data


if __name__ == "__main__":
    run_merge()
