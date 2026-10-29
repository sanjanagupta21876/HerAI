"""Unit tests for HERAI UI visualizations and application components."""

import pandas as pd
import pytest
import plotly.graph_objects as go

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


def test_cycle_wave_hero():
    """Verify cycle wave hero chart generates a clean Plotly figure."""
    fig = create_cycle_wave_hero()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # curve line + milestone markers


def test_gauge_chart_generation():
    """Verify gauge chart generates on continuous 0-10 scale with cohort median."""
    fig_opt = create_gauge_chart(8.8, "+0.20 relative to median")
    assert isinstance(fig_opt, go.Figure)

    fig_mod = create_gauge_chart(7.2, "-1.40 relative to median")
    assert isinstance(fig_mod, go.Figure)


def test_radar_chart_generation():
    """Verify radar chart generates with current cycle and with/without previous cycle."""
    curr = {
        "pain_level": 3,
        "mood_score": 7,
        "stress_score_cycle": 4.0,
        "sleep_hours_cycle": 7.5,
        "energy_level": 7,
        "concentration_score": 8,
    }
    prev = {
        "previous_pain_level": 4,
        "previous_mood_score": 6,
        "previous_stress_score": 5.0,
        "previous_sleep_hours": 7.0,
        "previous_energy_level": 6,
        "previous_concentration_score": 7,
    }

    fig_single = create_radar_chart(curr)
    assert isinstance(fig_single, go.Figure)
    assert len(fig_single.data) == 1

    fig_dual = create_radar_chart(curr, prev)
    assert isinstance(fig_dual, go.Figure)
    assert len(fig_dual.data) == 2


def test_cohort_and_cycle_length_distribution():
    """Verify cohort distribution and cycle length histogram charts."""
    sample_scores = pd.Series([7.5, 8.2, 9.1, 6.4, 8.8, 9.5])
    fig_cohort = create_cohort_distribution_chart(sample_scores, user_score=8.5)
    assert isinstance(fig_cohort, go.Figure)

    sample_lengths = pd.Series([26, 27, 28, 28, 29, 30, 31])
    fig_len = create_cycle_length_distribution_chart(sample_lengths)
    assert isinstance(fig_len, go.Figure)


def test_user_trajectory_chart():
    """Verify longitudinal trajectory chart."""
    toy_period_df = pd.DataFrame({
        "user_id": ["U001", "U001", "U001"],
        "cycle_number": [1, 2, 3],
        "overall_health_score": [8.0, 8.5, 9.0],
        "next_health_score": [8.5, 9.0, 8.7]
    })
    fig_traj = create_user_trajectory_chart(toy_period_df, "U001")
    assert isinstance(fig_traj, go.Figure)
    assert len(fig_traj.data) == 2


def test_feature_importance_chart():
    """Verify feature importance bar chart generation."""
    names = ["numeric__pain_level", "numeric__stress_score_baseline", "categorical__state_CA"]
    coefs = [-0.35, -0.65, 0.12]
    fig = create_feature_importance_chart(names, coefs, top_n=3)
    assert isinstance(fig, go.Figure)


def test_model_comparison_bar_chart():
    """Verify model benchmark bar chart generates with 2 error metrics."""
    fig = create_model_comparison_bar_chart()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # MAE and RMSE traces


def test_phase_distribution_chart():
    """Verify observed cycle phase bar chart."""
    toy_df = pd.DataFrame({
        "cycle_phase": ["Luteal", "Follicular", "Luteal", "Menstrual", "Follicular"]
    })
    fig = create_phase_distribution_chart(toy_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_pain_vs_health_chart():
    """Verify pain level vs mean health score relationship chart."""
    toy_df = pd.DataFrame({
        "pain_level": [1, 2, 3, 5, 8],
        "overall_health_score": [9.5, 9.0, 8.5, 7.5, 6.0]
    })
    fig = create_pain_vs_health_chart(toy_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
