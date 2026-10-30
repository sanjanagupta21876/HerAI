"""Configuration and schema definitions for the HERAI project."""

import os
from pathlib import Path
from typing import List, Dict, Any

# Root project paths
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Persistent database path
DATABASE_PATH = Path(os.environ.get("HERAI_DB_PATH", str(DATA_DIR / "herai.db")))

# Dataset file paths
RAW_PERIOD_LOG_PATH = RAW_DATA_DIR / "Period_Log.csv"
RAW_USER_PROFILE_PATH = RAW_DATA_DIR / "User_Profile.csv"
MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_data.csv"
ENGINEERED_DATA_PATH = PROCESSED_DATA_DIR / "engineered_data.csv"
TRAINING_DATA_PATH = PROCESSED_DATA_DIR / "training_data.csv"

# Model artifact paths
MODEL_RIDGE_PATH = MODELS_DIR / "herai_ridge_model.pkl"
MODEL_RIDGE_37_PATH = MODELS_DIR / "herai_ridge_37_features.pkl"

# Target variable
TARGET_COLUMN = "next_health_score"

# Identifiers and non-feature columns
ID_COLUMNS = ["user_id", "cycle_number", "start_date"]
EXCLUDED_FROM_TRAINING = ["user_id", "cycle_number", "start_date", TARGET_COLUMN]

# Features excluded from the 37-feature candidate model due to lookahead/inaccessibility
SUSPICIOUS_OR_LAB_FEATURES = [
    "estrogen_pgml",
    "progesterone_ngml",
    "ovulation_result",
    "log_consistency_score",
    "prepared_before_period",
]

# Raw schemas
PERIOD_LOG_REQUIRED_COLUMNS = [
    "user_id",
    "cycle_number",
    "start_date",
    "cycle_length_days",
    "prev_cycle_length",
    "cycle_phase",
    "flow_level",
    "pain_level",
    "pms_symptoms",
    "mood_score",
    "stress_score_cycle",
    "sleep_hours_cycle",
    "energy_level",
    "concentration_score",
    "work_hours_lost",
    "estrogen_pgml",
    "progesterone_ngml",
    "ovulation_result",
    "overall_health_score",
    "log_consistency_score",
    "prepared_before_period",
]

USER_PROFILE_REQUIRED_COLUMNS = [
    "user_id",
    "state",
    "age",
    "bmi",
    "diet_quality",
    "exercise_frequency",
    "sleep_hours",
    "caffeine_intake",
    "water_intake_liters",
    "alcohol_consumption",
    "smoking_status",
    "birth_control_use",
    "pcos_diagnosed",
    "stress_score_baseline",
]

# Lag features mapping: lag_feature_name -> current_feature_name
HISTORICAL_LAG_NUMERIC_MAPPING: Dict[str, str] = {
    "previous_health_score": "overall_health_score",
    "previous_cycle_length_feature": "cycle_length_days",
    "previous_pain_level": "pain_level",
    "previous_stress_score": "stress_score_cycle",
    "previous_sleep_hours": "sleep_hours_cycle",
    "previous_mood_score": "mood_score",
    "previous_energy_level": "energy_level",
    "previous_concentration_score": "concentration_score",
    "previous_work_hours_lost": "work_hours_lost",
}

HISTORICAL_LAG_CATEGORICAL_MAPPING: Dict[str, str] = {
    "previous_flow_level": "flow_level",
    "previous_pms_symptoms": "pms_symptoms",
}

# The 37 production features
FEATURES_37 = [
    "cycle_length_days",
    "prev_cycle_length",
    "cycle_phase",
    "flow_level",
    "pain_level",
    "pms_symptoms",
    "mood_score",
    "stress_score_cycle",
    "sleep_hours_cycle",
    "energy_level",
    "concentration_score",
    "work_hours_lost",
    "overall_health_score",
    "state",
    "age",
    "bmi",
    "diet_quality",
    "exercise_frequency",
    "sleep_hours",
    "caffeine_intake",
    "water_intake_liters",
    "alcohol_consumption",
    "smoking_status",
    "birth_control_use",
    "pcos_diagnosed",
    "stress_score_baseline",
    "previous_health_score",
    "previous_cycle_length_feature",
    "previous_pain_level",
    "previous_stress_score",
    "previous_sleep_hours",
    "previous_mood_score",
    "previous_energy_level",
    "previous_concentration_score",
    "previous_work_hours_lost",
    "previous_flow_level",
    "previous_pms_symptoms",
]

# Validation limits for inputs
BOUNDS = {
    "age": (12, 100),
    "bmi": (10.0, 70.0),
    "sleep_hours": (0.0, 24.0),
    "caffeine_intake": (0.0, 10.0),
    "water_intake_liters": (0.0, 10.0),
    "stress_score_baseline": (0.0, 10.0),
    "cycle_length_days": (15, 60),
    "pain_level": (0, 10),
    "mood_score": (0, 10),
    "stress_score_cycle": (0.0, 10.0),
    "sleep_hours_cycle": (0.0, 24.0),
    "energy_level": (0, 10),
    "concentration_score": (0, 10),
    "work_hours_lost": (0.0, 80.0),
    "overall_health_score": (0.0, 10.0),
}

