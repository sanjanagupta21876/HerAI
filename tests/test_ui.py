"""Unit tests for HERAI UI visualizations and application components."""

import pandas as pd
import pytest
import plotly.graph_objects as go

from herai.visualizations import (
    create_gauge_chart,
    create_radar_chart,
    create_cohort_distribution_chart,
    create_user_trajectory_chart,
    create_feature_importance_chart,
)


def test_gauge_chart_generation():
    """Verify gauge chart generates correctly across optimal, moderate, and sub-optimal scores."""
    fig_opt = create_gauge_chart(8.8, "Optimal")
    assert isinstance(fig_opt, go.Figure)

    fig_mod = create_gauge_chart(7.2, "Moderate")
    assert isinstance(fig_mod, go.Figure)

    fig_sub = create_gauge_chart(5.1, "Sub-optimal")
    assert isinstance(fig_sub, go.Figure)


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


def test_cohort_and_trajectory_charts():
    """Verify cohort distribution and multi-cycle patient trajectory charts."""
    sample_scores = pd.Series([7.5, 8.2, 9.1, 6.4, 8.8, 9.5])
    fig_cohort = create_cohort_distribution_chart(sample_scores, user_score=8.5)
    assert isinstance(fig_cohort, go.Figure)

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

