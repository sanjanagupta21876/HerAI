"""HERAI Feature Creation Pipeline.

Generates chronological lag features and aligns the next-cycle prediction target.
Saves the engineered dataset to data/processed/engineered_data.csv.
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from herai.config import (
    MERGED_DATA_PATH,
    ENGINEERED_DATA_PATH,
    TARGET_COLUMN,
    HISTORICAL_LAG_NUMERIC_MAPPING,
    HISTORICAL_LAG_CATEGORICAL_MAPPING,
)
from herai.data import sort_and_create_features


def run_feature_creation() -> pd.DataFrame:
    """Load merged data, sort chronologically, create lag features and target.
    
    Returns:
        pd.DataFrame: Engineered dataset with previous-cycle features and next_health_score.
    """
    print("\n========== HERAI FEATURE CREATION ==========")
    print(f"Loading merged dataset from: {MERGED_DATA_PATH}")

    if not MERGED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Merged dataset not found at {MERGED_DATA_PATH}. Run src/02_merge_data.py first."
        )

    period = pd.read_csv(MERGED_DATA_PATH)
    print("Dataset loaded successfully! Shape:", period.shape)

    # Sort chronologically and engineer previous-cycle features + next-cycle target
    engineered_df = sort_and_create_features(period)

    print("\n========== SORT & LAG FEATURES VERIFICATION ==========")
    print("Invalid dates:", engineered_df["start_date"].isna().sum())
    print("Total records:", len(engineered_df))
    print("Unique users:", engineered_df["user_id"].nunique())

    sample_cols = [
        "user_id", "cycle_number", "overall_health_score",
        "previous_health_score", TARGET_COLUMN
    ]
    print("\nSample records (User 1):")
    print(engineered_df[sample_cols].head(10).to_string(index=False))

    missing_targets = engineered_df[TARGET_COLUMN].isna().sum()
    print(f"\nRecords with available next-cycle target: {len(engineered_df) - missing_targets}")
    print(f"Records without next-cycle target (terminal user cycle): {missing_targets}")

    # Save the engineered dataset
    ENGINEERED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(ENGINEERED_DATA_PATH, index=False)

    print(f"\nEngineered dataset saved to: {ENGINEERED_DATA_PATH}")
    return engineered_df


if __name__ == "__main__":
    run_feature_creation()
