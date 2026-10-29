"""Comprehensive test suite for the HERAI project.

Covers schema validation, chronological sorting, lag feature generation,
user isolation across train/test splits, and privacy-preserving inference.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from herai.config import (
    RAW_PERIOD_LOG_PATH,
    RAW_USER_PROFILE_PATH,
    BOUNDS,
)
from herai.data import (
    load_raw_data,
    validate_schemas,
    merge_datasets,
    sort_and_create_features,
)
from herai.models import split_by_user
from herai.inference import (
    UserProfileInput,
    CurrentCycleInput,
    HistoricalCycleInput,
    HerAIPredictor,
    InputValidationError,
)


def test_raw_data_loading_and_schemas():
    """Ensure raw datasets load cleanly and pass strict schema validation."""
    period_df, profile_df = load_raw_data()
    assert not period_df.empty, "Period_Log.csv should not be empty."
    assert not profile_df.empty, "User_Profile.csv should not be empty."
    assert period_df["user_id"].nunique() == profile_df["user_id"].nunique()


def test_schema_validation_duplicate_key_detection():
    """Test that duplicate primary keys in User Profile are caught."""
    dummy_profile = pd.DataFrame({
        "user_id": ["U01", "U01"],
        "state": ["CA", "NY"],
        "age": [25, 30],
        "bmi": [22.0, 24.0],
        "diet_quality": ["Good", "Fair"],
        "exercise_frequency": ["Daily", "Never"],
        "sleep_hours": [8.0, 7.0],
        "caffeine_intake": [1.0, 2.0],
        "water_intake_liters": [2.0, 2.0],
        "alcohol_consumption": ["None", "None"],
        "smoking_status": ["Non-smoker", "Non-smoker"],
        "birth_control_use": [0, 0],
        "pcos_diagnosed": [0, 0],
        "stress_score_baseline": [3.0, 4.0],
    })
    period_df, _ = load_raw_data()
    with pytest.raises(ValueError, match="Duplicate user_id detected"):
        validate_schemas(period_df, dummy_profile)


def test_chronological_ordering_and_lag_features():
    """Verify that lag features match cycle T-1 and target matches cycle T+1."""
    toy_df = pd.DataFrame({
        "user_id": ["U1", "U1", "U1"],
        "cycle_number": [2, 1, 3],
        "start_date": ["2024-02-01", "2024-01-01", "2024-03-01"],
        "cycle_length_days": [28, 30, 27],
        "pain_level": [3, 2, 4],
        "stress_score_cycle": [4.0, 5.0, 3.5],
        "sleep_hours_cycle": [7.0, 8.0, 6.5],
        "mood_score": [7, 6, 8],
        "energy_level": [6, 7, 5],
        "concentration_score": [8, 7, 8],
        "work_hours_lost": [0.0, 0.0, 2.0],
        "flow_level": ["Medium", "Light", "Heavy"],
        "pms_symptoms": ["Mild", "None", "Moderate"],
        "overall_health_score": [8.5, 9.0, 7.8],
    })

    engineered = sort_and_create_features(toy_df)

    # Chronological sort check
    assert list(engineered["cycle_number"]) == [1, 2, 3]

    # Target alignment check (shift -1)
    assert engineered.loc[0, "next_health_score"] == 8.5
    assert engineered.loc[1, "next_health_score"] == 7.8
    assert pd.isna(engineered.loc[2, "next_health_score"])

    # Historical lag check (shift 1)
    assert pd.isna(engineered.loc[0, "previous_health_score"])
    assert engineered.loc[1, "previous_health_score"] == 9.0
    assert engineered.loc[2, "previous_health_score"] == 8.5
    assert engineered.loc[1, "previous_cycle_length_feature"] == 30


def test_user_grouped_split_isolation():
    """Ensure training and test sets have 0 overlapping users."""
    period_df, profile_df = load_raw_data()
    merged = merge_datasets(period_df, profile_df)
    engineered = sort_and_create_features(merged)

    X_train, X_test, y_train, y_test, g_train, g_test = split_by_user(
        engineered, test_size=0.2, random_state=42
    )

    train_users = set(g_train)
    test_users = set(g_test)
    assert len(train_users.intersection(test_users)) == 0, "Data leakage: Users overlap across splits!"


def test_inference_input_validation_bounds():
    """Test that physiological boundary checks prevent invalid inputs."""
    # Invalid negative age
    profile = UserProfileInput(
        age=-5,
        bmi=22.0,
        sleep_hours=8.0,
        stress_score_baseline=4.0,
    )
    with pytest.raises(InputValidationError, match="age"):
        profile.validate()

    # Invalid BMI > 70
    profile2 = UserProfileInput(
        age=25,
        bmi=120.0,
        sleep_hours=8.0,
        stress_score_baseline=4.0,
    )
    with pytest.raises(InputValidationError, match="bmi"):
        profile2.validate()

    # Invalid pain level > 10
    cycle = CurrentCycleInput(
        cycle_length_days=28,
        pain_level=15,
        mood_score=6,
        stress_score_cycle=5.0,
        sleep_hours_cycle=7.0,
        energy_level=6,
        concentration_score=7,
    )
    with pytest.raises(InputValidationError, match="pain_level"):
        cycle.validate()


def test_inference_without_stranger_row():
    """Test inference without historical cycle (Cycle 1 user) and with user-owned historical cycle."""
    predictor = HerAIPredictor()

    profile = UserProfileInput(
        age=28,
        bmi=23.5,
        sleep_hours=7.5,
        stress_score_baseline=4.5,
        diet_quality="Good",
        exercise_frequency="3–4 days/week",
    )

    current_cycle = CurrentCycleInput(
        cycle_length_days=29,
        pain_level=3,
        mood_score=7,
        stress_score_cycle=4.0,
        sleep_hours_cycle=7.5,
        energy_level=7,
        concentration_score=8,
        work_hours_lost=0.0,
        overall_health_score=8.8,
    )

    # 1. Prediction without history
    result_no_hist = predictor.predict(profile, current_cycle)
    assert 1.0 <= result_no_hist.predicted_next_health_score <= 10.0
    assert result_no_hist.has_historical_cycle is False
    assert len(result_no_hist.warnings) > 0

    # 2. Prediction with user's own history
    hist = HistoricalCycleInput(
        previous_health_score=8.5,
        previous_cycle_length_feature=28,
        previous_pain_level=4,
        previous_stress_score=4.5,
        previous_sleep_hours=7.0,
        previous_mood_score=6,
        previous_energy_level=6,
        previous_concentration_score=7,
        previous_work_hours_lost=0.0,
        previous_flow_level="Medium",
        previous_pms_symptoms="Mild",
    )

    result_with_hist = predictor.predict(profile, current_cycle, historical_cycle=hist)
    assert 1.0 <= result_with_hist.predicted_next_health_score <= 10.0
    assert result_with_hist.has_historical_cycle is True
    assert result_with_hist.features_used["previous_health_score"] == 8.5

