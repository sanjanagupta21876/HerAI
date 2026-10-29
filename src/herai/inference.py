"""Privacy-preserving, validated inference pipeline for HERAI."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from herai.config import (
    MODEL_RIDGE_37_PATH,
    FEATURES_37,
    BOUNDS,
)
from herai.models import load_model


class InputValidationError(ValueError):
    """Raised when user input fails range or type constraints."""
    pass


@dataclass
class UserProfileInput:
    """Demographic and lifestyle features for an individual user."""
    age: float
    bmi: float
    sleep_hours: float
    stress_score_baseline: float
    state: str = "Unknown"
    diet_quality: str = "Good"
    exercise_frequency: str = "3–4 days/week"
    caffeine_intake: float = 1.0
    water_intake_liters: float = 2.0
    alcohol_consumption: str = "None"
    smoking_status: str = "Non-smoker"
    birth_control_use: int = 0
    pcos_diagnosed: int = 0

    def validate(self) -> None:
        """Validate value ranges against physiological and domain bounds."""
        for field, (low, high) in BOUNDS.items():
            if hasattr(self, field):
                val = getattr(self, field)
                if val is not None and (val < low or val > high):
                    raise InputValidationError(
                        f"Field '{field}' with value {val} is outside allowed range [{low}, {high}]."
                    )
        if self.birth_control_use not in (0, 1):
            raise InputValidationError("birth_control_use must be 0 (No) or 1 (Yes).")
        if self.pcos_diagnosed not in (0, 1):
            raise InputValidationError("pcos_diagnosed must be 0 (No) or 1 (Yes).")


@dataclass
class CurrentCycleInput:
    """Current menstrual cycle and symptom metrics."""
    cycle_length_days: float
    pain_level: float
    mood_score: float
    stress_score_cycle: float
    sleep_hours_cycle: float
    energy_level: float
    concentration_score: float
    work_hours_lost: float = 0.0
    overall_health_score: Optional[float] = None
    cycle_phase: str = "Menstrual"
    flow_level: str = "Medium"
    pms_symptoms: str = "Mild"

    def validate(self) -> None:
        """Validate cycle metrics."""
        for field, (low, high) in BOUNDS.items():
            if hasattr(self, field):
                val = getattr(self, field)
                if val is not None and (val < low or val > high):
                    raise InputValidationError(
                        f"Field '{field}' with value {val} is outside allowed range [{low}, {high}]."
                    )
        # If overall_health_score is not provided, estimate using empirical formula from 01_inspect_data
        if self.overall_health_score is None:
            # Empirical regression formula:
            # Score ~ 10 - 0.2*pain - 0.15*stress - 0.1*work_hours_lost + 0.1*energy + 0.1*mood + 0.1*concentration
            est = 10.0 - (0.2 * self.pain_level) - (0.15 * self.stress_score_cycle) \
                  - (0.05 * min(self.work_hours_lost, 40)) + (0.1 * self.energy_level) \
                  + (0.1 * self.mood_score) + (0.1 * self.concentration_score)
            self.overall_health_score = float(np.clip(est, 1.0, 10.0))


@dataclass
class HistoricalCycleInput:
    """Optional historical metrics belonging strictly to the same user."""
    previous_health_score: float
    previous_cycle_length_feature: float
    previous_pain_level: float
    previous_stress_score: float
    previous_sleep_hours: float
    previous_mood_score: float
    previous_energy_level: float
    previous_concentration_score: float
    previous_work_hours_lost: float
    previous_flow_level: str = "Medium"
    previous_pms_symptoms: str = "Mild"
    prev_cycle_length: Optional[float] = None

    def validate(self) -> None:
        if self.prev_cycle_length is None:
            self.prev_cycle_length = self.previous_cycle_length_feature


@dataclass
class PredictionOutput:
    """Structured and reproducible prediction output."""
    predicted_next_health_score: float
    score_category: str
    has_historical_cycle: bool
    summary: str
    warnings: List[str]
    features_used: Dict[str, Any]
    disclaimer: str = (
        "Educational prototype only. Not medically validated for diagnosis, "
        "treatment, or clinical decision-making."
    )


class HerAIPredictor:
    """Inference engine using strictly user-owned records and trained pipelines."""

    def __init__(self, model_pipeline: Optional[Pipeline] = None):
        if model_pipeline is None:
            self.model = load_model(MODEL_RIDGE_37_PATH)
        else:
            self.model = model_pipeline

    def predict(
        self,
        profile: UserProfileInput,
        current_cycle: CurrentCycleInput,
        historical_cycle: Optional[HistoricalCycleInput] = None,
    ) -> PredictionOutput:
        """Run ethical prediction without borrowing stranger data."""
        # Validate inputs
        profile.validate()
        current_cycle.validate()
        if historical_cycle:
            historical_cycle.validate()

        warnings: List[str] = []
        has_history = historical_cycle is not None

        # Build feature row exactly matching FEATURES_37 schema
        feature_dict: Dict[str, Any] = {
            # User profile
            "state": profile.state,
            "age": profile.age,
            "bmi": profile.bmi,
            "diet_quality": profile.diet_quality,
            "exercise_frequency": profile.exercise_frequency,
            "sleep_hours": profile.sleep_hours,
            "caffeine_intake": profile.caffeine_intake,
            "water_intake_liters": profile.water_intake_liters,
            "alcohol_consumption": profile.alcohol_consumption,
            "smoking_status": profile.smoking_status,
            "birth_control_use": profile.birth_control_use,
            "pcos_diagnosed": profile.pcos_diagnosed,
            "stress_score_baseline": profile.stress_score_baseline,

            # Current cycle
            "cycle_length_days": current_cycle.cycle_length_days,
            "prev_cycle_length": (
                historical_cycle.prev_cycle_length
                if has_history and historical_cycle.prev_cycle_length is not None
                else np.nan
            ),
            "cycle_phase": current_cycle.cycle_phase,
            "flow_level": current_cycle.flow_level,
            "pain_level": current_cycle.pain_level,
            "pms_symptoms": current_cycle.pms_symptoms,
            "mood_score": current_cycle.mood_score,
            "stress_score_cycle": current_cycle.stress_score_cycle,
            "sleep_hours_cycle": current_cycle.sleep_hours_cycle,
            "energy_level": current_cycle.energy_level,
            "concentration_score": current_cycle.concentration_score,
            "work_hours_lost": current_cycle.work_hours_lost,
            "overall_health_score": current_cycle.overall_health_score,

            # Previous cycle lag features
            "previous_health_score": (
                historical_cycle.previous_health_score if has_history else np.nan
            ),
            "previous_cycle_length_feature": (
                historical_cycle.previous_cycle_length_feature if has_history else np.nan
            ),
            "previous_pain_level": (
                historical_cycle.previous_pain_level if has_history else np.nan
            ),
            "previous_stress_score": (
                historical_cycle.previous_stress_score if has_history else np.nan
            ),
            "previous_sleep_hours": (
                historical_cycle.previous_sleep_hours if has_history else np.nan
            ),
            "previous_mood_score": (
                historical_cycle.previous_mood_score if has_history else np.nan
            ),
            "previous_energy_level": (
                historical_cycle.previous_energy_level if has_history else np.nan
            ),
            "previous_concentration_score": (
                historical_cycle.previous_concentration_score if has_history else np.nan
            ),
            "previous_work_hours_lost": (
                historical_cycle.previous_work_hours_lost if has_history else np.nan
            ),
            "previous_flow_level": (
                historical_cycle.previous_flow_level if has_history else "Unknown"
            ),
            "previous_pms_symptoms": (
                historical_cycle.previous_pms_symptoms if has_history else "Unknown"
            ),
        }

        if not has_history:
            warnings.append(
                "No previous cycle history was provided. The trained pipeline's learned "
                "cohort medians/modes were used as principled baseline defaults for lag features."
            )

        # Assemble DataFrame in exact feature order
        input_df = pd.DataFrame([feature_dict])[FEATURES_37]

        # Predict
        raw_pred = float(self.model.predict(input_df)[0])
        # Clinically bounded between 0 and 10
        bounded_pred = round(float(np.clip(raw_pred, 1.0, 10.0)), 2)

        if bounded_pred >= 8.0:
            category = "Optimal"
            summary = "Predicted next-cycle health indicator is favorable."
        elif bounded_pred >= 6.0:
            category = "Moderate"
            summary = "Predicted next-cycle health indicator shows moderate wellbeing."
        else:
            category = "Sub-optimal"
            summary = "Predicted next-cycle health indicator suggests elevated stress or symptoms."

        return PredictionOutput(
            predicted_next_health_score=bounded_pred,
            score_category=category,
            has_historical_cycle=has_history,
            summary=summary,
            warnings=warnings,
            features_used={k: (None if pd.isna(v) else v) for k, v in feature_dict.items()},
        )

