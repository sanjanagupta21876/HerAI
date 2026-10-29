"""Data loading, schema validation, and feature engineering for HERAI."""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from herai.config import (
    RAW_PERIOD_LOG_PATH,
    RAW_USER_PROFILE_PATH,
    PERIOD_LOG_REQUIRED_COLUMNS,
    USER_PROFILE_REQUIRED_COLUMNS,
    HISTORICAL_LAG_NUMERIC_MAPPING,
    HISTORICAL_LAG_CATEGORICAL_MAPPING,
    TARGET_COLUMN,
)


def validate_schemas(period_df: pd.DataFrame, profile_df: pd.DataFrame) -> None:
    """Validate that raw datasets contain required columns and valid keys.
    
    Raises:
        ValueError: If columns are missing or if duplicate keys are detected.
    """
    missing_period_cols = [c for c in PERIOD_LOG_REQUIRED_COLUMNS if c not in period_df.columns]
    if missing_period_cols:
        raise ValueError(f"Period log missing required columns: {missing_period_cols}")

    missing_profile_cols = [c for c in USER_PROFILE_REQUIRED_COLUMNS if c not in profile_df.columns]
    if missing_profile_cols:
        raise ValueError(f"User profile missing required columns: {missing_profile_cols}")

    # Duplicate check on user profile primary key
    if profile_df["user_id"].duplicated().any():
        num_dup = profile_df["user_id"].duplicated().sum()
        raise ValueError(f"Duplicate user_id detected in User_Profile: {num_dup} duplicate entries.")

    # Duplicate check on period log user + cycle
    if period_df.duplicated(subset=["user_id", "cycle_number"]).any():
        num_dup = period_df.duplicated(subset=["user_id", "cycle_number"]).sum()
        raise ValueError(f"Duplicate (user_id, cycle_number) detected in Period_Log: {num_dup} duplicates.")


def load_raw_data(
    period_path: Optional[str] = None,
    profile_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Period Log and User Profile datasets with validation."""
    p_path = period_path or RAW_PERIOD_LOG_PATH
    u_path = profile_path or RAW_USER_PROFILE_PATH

    period_df = pd.read_csv(p_path)
    profile_df = pd.read_csv(u_path)

    validate_schemas(period_df, profile_df)
    return period_df, profile_df


def merge_datasets(period_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """Safely merge period records with user profiles using a many-to-one join.
    
    Returns:
        Merged DataFrame with user profile demographic and lifestyle attributes.
    """
    merged = period_df.merge(
        profile_df,
        on="user_id",
        how="left",
        validate="many_to_one"
    )
    if merged["user_id"].isna().any():
        raise ValueError("Merge generated null user_ids.")
    return merged


def sort_and_create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Chronologically sort cycles and generate lag features and next-cycle target.
    
    Assumptions & Logic:
    - start_date is parsed to datetime; invalid dates are flagged.
    - Each user's cycles are ordered strictly by (user_id, start_date).
    - Previous cycle features are computed using .shift(1) grouped by user_id.
    - For cycle 1, previous cycle features are naturally NaN (first cycle recorded).
    - Prediction target next_health_score is computed using .shift(-1) grouped by user_id.
    - For the last recorded cycle of each user, next_health_score is NaN (future target unknown).
    """
    df_sorted = df.copy()
    df_sorted["start_date"] = pd.to_datetime(df_sorted["start_date"], errors="coerce")

    if df_sorted["start_date"].isna().any():
        raise ValueError("One or more records contain unparseable start_date timestamps.")

    df_sorted = df_sorted.sort_values(by=["user_id", "start_date"]).reset_index(drop=True)

    # 1. Historical Lag Features
    for lag_col, curr_col in HISTORICAL_LAG_NUMERIC_MAPPING.items():
        df_sorted[lag_col] = df_sorted.groupby("user_id")[curr_col].shift(1)

    for lag_col, curr_col in HISTORICAL_LAG_CATEGORICAL_MAPPING.items():
        df_sorted[lag_col] = df_sorted.groupby("user_id")[curr_col].shift(1)

    # 2. Prediction Target: next-cycle health score
    df_sorted[TARGET_COLUMN] = df_sorted.groupby("user_id")["overall_health_score"].shift(-1)

    return df_sorted


def prepare_training_dataset(engineered_df: pd.DataFrame) -> pd.DataFrame:
    """Filter engineered data to records with verified next-cycle targets.
    
    Note on imputation:
    Imputation is deliberately kept within the scikit-learn Pipeline (SimpleImputer)
    during model training and cross-validation to strictly prevent test-fold data leakage.
    However, for downstream users expecting `training_data.csv` with fallback imputations,
    this function produces a clean training table where rows without target are removed.
    """
    valid_data = engineered_df.dropna(subset=[TARGET_COLUMN]).copy()
    return valid_data

