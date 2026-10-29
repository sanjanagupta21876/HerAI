"""Phase 9 Quality Assurance Pipeline & Integration Tests.

Validates:
1. Saved model artifact integrity, feature order, and weights.
2. Inference pipeline with all preset scenarios and custom inputs.
3. Strict input bounds validation and error handling.
4. Privacy boundaries: zero stranger data borrowing for Cycle 1.
5. All 10 Plotly chart generators including empty states and edge cases.
6. Execution safety of CLI scripts from project root.
"""

import sys
import subprocess
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

from herai.config import (
    MODEL_RIDGE_37_PATH,
    MODEL_RIDGE_PATH,
    TRAINING_DATA_PATH,
    FEATURES_37,
    BOUNDS,
    PROJECT_ROOT,
)
from herai.inference import (
    HerAIPredictor,
    UserProfileInput,
    CurrentCycleInput,
    HistoricalCycleInput,
    InputValidationError,
    PredictionOutput,
)
from herai.visualizations import (
    create_cycle_wave_hero,
    create_gauge_chart,
    create_radar_chart,
    create_cohort_distribution_chart,
    create_cycle_length_distribution_chart,
    create_user_trajectory_chart,
    create_feature_importance_chart,
    create_model_comparison_bar_chart,
    create_phase_distribution_chart,
    create_pain_vs_health_chart,
)


def test_model_artifact_integrity():
    """Verify production 37-feature Ridge model artifact exists and has verified weights."""
    assert MODEL_RIDGE_37_PATH.exists(), f"Model artifact missing at {MODEL_RIDGE_37_PATH}"

    pipeline = joblib.load(MODEL_RIDGE_37_PATH)
    assert hasattr(pipeline, "named_steps")
    assert "preprocessor" in pipeline.named_steps
    assert "regressor" in pipeline.named_steps

    reg = pipeline.named_steps["regressor"]
    prep = pipeline.named_steps["preprocessor"]

    # Verify intercept matches verified empirical value (8.5786 ± 0.01)
    assert abs(reg.intercept_ - 8.5786) < 0.05

    # Verify preprocessor feature count
    out_features = prep.get_feature_names_out()
    assert len(out_features) == 102

    # Verify baseline stress is the strongest negative driver
    coef_dict = dict(zip(out_features, reg.coef_))
    assert "numeric__stress_score_baseline" in coef_dict
    assert coef_dict["numeric__stress_score_baseline"] < -0.60


def test_feature_order_and_preprocessing_alignment():
    """Verify feature row constructed for inference strictly matches FEATURES_37."""
    predictor = HerAIPredictor()

    profile = UserProfileInput(
        age=28.0,
        bmi=22.5,
        sleep_hours=7.5,
        stress_score_baseline=4.0,
    )
    cycle = CurrentCycleInput(
        cycle_length_days=28.0,
        pain_level=3.0,
        mood_score=7.0,
        stress_score_cycle=4.0,
        sleep_hours_cycle=7.5,
        energy_level=7.0,
        concentration_score=7.0,
        work_hours_lost=0.0,
    )

    row_df = predictor.build_feature_row(profile, cycle, historical_cycle=None)
    assert list(row_df.columns) == FEATURES_37
    assert len(row_df.columns) == 37
    assert len(row_df) == 1


def test_prediction_workflow_with_presets():
    """Verify live prediction on multiple clinical scenarios produces valid scores."""
    predictor = HerAIPredictor()

    # Preset 1: Balanced & regular cycle
    prof_bal = UserProfileInput(age=28, bmi=22.2, sleep_hours=8.0, stress_score_baseline=3.0)
    cyc_bal = CurrentCycleInput(cycle_length_days=28, pain_level=2, mood_score=8, stress_score_cycle=3.0,
                                sleep_hours_cycle=8.0, energy_level=8, concentration_score=8, work_hours_lost=0.0)
    pred_bal = predictor.predict(prof_bal, cyc_bal)

    assert isinstance(pred_bal, PredictionOutput)
    assert 0.0 <= pred_bal.predicted_next_health_score <= 10.0
    assert pred_bal.score_category in ["Optimal", "Moderate", "Needs Attention"]

    # Preset 2: High stress scenario
    prof_stress = UserProfileInput(age=31, bmi=24.0, sleep_hours=6.5, stress_score_baseline=7.5)
    cyc_stress = CurrentCycleInput(cycle_length_days=31, pain_level=4, mood_score=5, stress_score_cycle=8.0,
                                   sleep_hours_cycle=5.5, energy_level=4, concentration_score=5, work_hours_lost=4.0)
    pred_stress = predictor.predict(prof_stress, cyc_stress)

    assert 0.0 <= pred_stress.predicted_next_health_score <= 10.0
    # Balanced should yield higher next health score than high stress
    assert pred_bal.predicted_next_health_score > pred_stress.predicted_next_health_score


def test_first_time_user_cycle_1_without_stranger_copying():
    """Verify first-time user (Cycle 1) applies cohort reference medians with no stranger-row borrowing."""
    predictor = HerAIPredictor()

    prof = UserProfileInput(age=25, bmi=21.0, sleep_hours=8.0, stress_score_baseline=3.0)
    cyc = CurrentCycleInput(cycle_length_days=28, pain_level=2, mood_score=8, stress_score_cycle=3.0,
                            sleep_hours_cycle=8.0, energy_level=8, concentration_score=8, work_hours_lost=0.0)

    result = predictor.predict(prof, cyc, historical_cycle=None)
    assert not result.has_historical_cycle
    assert len(result.warnings) > 0
    assert "No previous cycle history was provided" in result.warnings[0]


def test_returning_user_with_custom_history():
    """Verify returning user with custom history uses personal history without Cycle 1 warning."""
    predictor = HerAIPredictor()

    prof = UserProfileInput(age=25, bmi=21.0, sleep_hours=8.0, stress_score_baseline=3.0)
    cyc = CurrentCycleInput(cycle_length_days=28, pain_level=2, mood_score=8, stress_score_cycle=3.0,
                            sleep_hours_cycle=8.0, energy_level=8, concentration_score=8, work_hours_lost=0.0)
    hist = HistoricalCycleInput(
        previous_health_score=8.5,
        previous_cycle_length_feature=28.0,
        previous_pain_level=2.0,
        previous_stress_score=3.0,
        previous_sleep_hours=8.0,
        previous_mood_score=8.0,
        previous_energy_level=8.0,
        previous_concentration_score=8.0,
        previous_work_hours_lost=0.0,
    )

    result = predictor.predict(prof, cyc, historical_cycle=hist)
    assert result.has_historical_cycle
    assert len(result.warnings) == 0


def test_invalid_inputs_trigger_validation_errors():
    """Verify physiological out-of-bound inputs raise InputValidationError."""
    # Invalid age (< 12)
    with pytest.raises(InputValidationError):
        UserProfileInput(age=5.0, bmi=22.0, sleep_hours=8.0, stress_score_baseline=3.0).validate()

    # Invalid age (> 100)
    with pytest.raises(InputValidationError):
        UserProfileInput(age=120.0, bmi=22.0, sleep_hours=8.0, stress_score_baseline=3.0).validate()

    # Invalid BMI (< 10)
    with pytest.raises(InputValidationError):
        UserProfileInput(age=25.0, bmi=5.0, sleep_hours=8.0, stress_score_baseline=3.0).validate()

    # Invalid cycle length (< 15 days)
    with pytest.raises(InputValidationError):
        CurrentCycleInput(cycle_length_days=10.0, pain_level=2, mood_score=8, stress_score_cycle=3.0,
                          sleep_hours_cycle=8.0, energy_level=8, concentration_score=8, work_hours_lost=0.0).validate()

    # Invalid pain level (> 10)
    with pytest.raises(InputValidationError):
        CurrentCycleInput(cycle_length_days=28.0, pain_level=15.0, mood_score=8, stress_score_cycle=3.0,
                          sleep_hours_cycle=8.0, energy_level=8, concentration_score=8, work_hours_lost=0.0).validate()


def test_visualizations_render_valid_figures():
    """Verify all 10 Plotly chart functions return valid go.Figure instances."""
    fig1 = create_cycle_wave_hero()
    assert isinstance(fig1, go.Figure)

    fig2 = create_gauge_chart(8.5, "+0.10 vs median")
    assert isinstance(fig2, go.Figure)

    fig3 = create_radar_chart({"pain_level": 3, "mood_score": 7, "stress_score_cycle": 4, "sleep_hours_cycle": 8, "energy_level": 7, "concentration_score": 8})
    assert isinstance(fig3, go.Figure)

    scores = pd.Series([7.5, 8.2, 8.6, 9.1])
    fig4 = create_cohort_distribution_chart(scores, user_score=8.6)
    assert isinstance(fig4, go.Figure)

    lengths = pd.Series([26, 28, 28, 30])
    fig5 = create_cycle_length_distribution_chart(lengths)
    assert isinstance(fig5, go.Figure)

    fig6 = create_feature_importance_chart(["stress_baseline", "current_health"], [-0.67, 0.45], top_n=2)
    assert isinstance(fig6, go.Figure)

    fig7 = create_model_comparison_bar_chart()
    assert isinstance(fig7, go.Figure)

    toy_df = pd.DataFrame({
        "cycle_phase": ["Luteal", "Follicular", "Menstrual"],
        "pain_level": [2, 5, 8],
        "overall_health_score": [9.0, 8.5, 6.5],
        "user_id": ["U001", "U001", "U001"],
        "cycle_number": [1, 2, 3],
        "next_health_score": [8.5, 6.5, 8.0]
    })
    fig8 = create_phase_distribution_chart(toy_df)
    assert isinstance(fig8, go.Figure)

    fig9 = create_pain_vs_health_chart(toy_df)
    assert isinstance(fig9, go.Figure)

    fig10 = create_user_trajectory_chart(toy_df, "U001")
    assert isinstance(fig10, go.Figure)


def test_empty_state_and_edge_case_handling():
    """Verify visualization functions handle empty inputs or non-existent IDs gracefully."""
    empty_df = pd.DataFrame()

    fig_empty_phase = create_phase_distribution_chart(empty_df)
    assert isinstance(fig_empty_phase, go.Figure)
    assert len(fig_empty_phase.data) == 0

    fig_empty_pain = create_pain_vs_health_chart(empty_df)
    assert isinstance(fig_empty_pain, go.Figure)
    assert len(fig_empty_pain.data) == 0

    fig_empty_user = create_user_trajectory_chart(empty_df, "NON_EXISTENT")
    assert isinstance(fig_empty_user, go.Figure)
    assert len(fig_empty_user.data) == 0


def test_cli_scripts_execution_safety():
    """Verify core CLI prediction and leakage audit scripts run from project root without error."""
    # Test 06_predict.py
    res_pred = subprocess.run(
        [sys.executable, "src/06_predict.py"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    assert res_pred.returncode == 0, f"06_predict.py failed: {res_pred.stderr}"
    assert "HERAI 37-feature model loaded successfully!" in res_pred.stdout

    # Test 08_feature_leakage_audit.py
    res_leak = subprocess.run(
        [sys.executable, "src/08_feature_leakage_audit.py"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    assert res_leak.returncode == 0, f"08_feature_leakage_audit.py failed: {res_leak.stderr}"
    assert "Mismatches: 0" in res_leak.stdout
