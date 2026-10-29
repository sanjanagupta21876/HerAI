"""Interactive Plotly visualization components for the HERAI Web Application."""

from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Brand color palette
PALETTE = {
    "berry": "#882D52",
    "rose": "#D47A94",
    "blush": "#F5E6EC",
    "cream": "#FDFBF9",
    "dark": "#2B1A24",
    "sage": "#388E3C",
    "amber": "#F57C00",
    "coral": "#D32F2F",
    "slate": "#5A6578",
    "lavender": "#8E24AA",
}


def create_gauge_chart(score: float, category: str) -> go.Figure:
    """Create an elegant semi-circular gauge chart for predicted health score."""
    if score >= 8.0:
        bar_color = PALETTE["sage"]
    elif score >= 6.0:
        bar_color = PALETTE["amber"]
    else:
        bar_color = PALETTE["coral"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Predicted Score<br><span style='font-size:16px;color:{bar_color}'><b>{category}</b></span>", 'font': {'size': 20, 'family': 'sans-serif'}},
        number={'suffix': " / 10", 'font': {'size': 38, 'color': PALETTE["dark"]}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': PALETTE["slate"]},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E0D7DD",
            'steps': [
                {'range': [0, 6], 'color': "#FFEBEE"},
                {'range': [6, 8], 'color': "#FFF8E1"},
                {'range': [8, 10], 'color': "#E8F5E9"},
            ],
            'threshold': {
                'line': {'color': PALETTE["berry"], 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=260,
        margin=dict(l=25, r=25, t=40, b=10),
    )
    return fig


def create_radar_chart(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]] = None
) -> go.Figure:
    """Create a radar chart comparing dimensions of menstrual wellbeing."""
    categories = [
        "Pain Control",
        "Mood Balance",
        "Stress Resilience",
        "Sleep Quality",
        "Vitality / Energy",
        "Focus / Concentration"
    ]

    # Invert pain & stress so 10 is always 'best' for radar chart intuition
    curr_values = [
        10.0 - current.get("pain_level", 3.0),
        current.get("mood_score", 7.0),
        10.0 - current.get("stress_score_cycle", 4.0),
        min(10.0, (current.get("sleep_hours_cycle", 7.5) / 9.0) * 10.0),
        current.get("energy_level", 7.0),
        current.get("concentration_score", 7.0),
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=curr_values + [curr_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(136, 45, 82, 0.25)',
        line=dict(color=PALETTE["berry"], width=2.5),
        name="Current Cycle"
    ))

    if previous:
        prev_values = [
            10.0 - previous.get("previous_pain_level", 3.0),
            previous.get("previous_mood_score", 7.0),
            10.0 - previous.get("previous_stress_score", 4.0),
            min(10.0, (previous.get("previous_sleep_hours", 7.5) / 9.0) * 10.0),
            previous.get("previous_energy_level", 7.0),
            previous.get("previous_concentration_score", 7.0),
        ]
        fig.add_trace(go.Scatterpolar(
            r=prev_values + [prev_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(212, 122, 148, 0.15)',
            line=dict(color=PALETTE["rose"], width=1.5, dash='dot'),
            name="Previous Cycle"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=9, color=PALETTE["slate"])
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=PALETTE["dark"], family="sans-serif")
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=320,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    return fig


def create_cohort_distribution_chart(scores_series: pd.Series, user_score: Optional[float] = None) -> go.Figure:
    """Show where the user's score falls along the dataset distribution."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores_series,
        nbinsx=30,
        marker=dict(color='rgba(136, 45, 82, 0.65)', line=dict(color=PALETTE["berry"], width=1)),
        name="Population Cohort"
    ))

    if user_score is not None:
        fig.add_vline(
            x=user_score,
            line_width=3,
            line_dash="dash",
            line_color=PALETTE["coral"],
            annotation_text=f"Your Score: {user_score:.2f}",
            annotation_position="top right"
        )

    fig.update_layout(
        title="Distribution of Next-Cycle Health Scores in Dataset (N=15,976)",
        xaxis_title="Next-Cycle Health Score (0 - 10)",
        yaxis_title="Cycle Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250, 245, 248, 0.5)',
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor='#EAE2E6'),
        yaxis=dict(gridcolor='#EAE2E6'),
    )
    return fig


def create_user_trajectory_chart(period_df: pd.DataFrame, user_id: str) -> go.Figure:
    """Plot multi-cycle trajectory for a specific user."""
    user_records = period_df[period_df["user_id"] == user_id].sort_values("cycle_number").copy()
    if user_records.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=user_records["cycle_number"],
        y=user_records["overall_health_score"],
        mode="lines+markers",
        name="Current Cycle Health",
        line=dict(color=PALETTE["berry"], width=2.5),
        marker=dict(size=8, color=PALETTE["berry"])
    ))

    if "next_health_score" in user_records.columns:
        fig.add_trace(go.Scatter(
            x=user_records["cycle_number"],
            y=user_records["next_health_score"],
            mode="lines+markers",
            name="Next Cycle Target",
            line=dict(color=PALETTE["sage"], width=2, dash="dash"),
            marker=dict(size=6, color=PALETTE["sage"])
        ))

    fig.update_layout(
        title=f"Longitudinal Health Score Trajectory — User {user_id}",
        xaxis_title="Cycle Number",
        yaxis_title="Health Score",
        yaxis=dict(range=[3, 11], gridcolor='#EAE2E6'),
        xaxis=dict(tickmode='linear', dtick=1, gridcolor='#EAE2E6'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250, 245, 248, 0.5)',
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def create_feature_importance_chart(feature_names: List[str], coefficients: List[float], top_n: int = 12) -> go.Figure:
    """Bar chart of top standardized Ridge regression coefficients."""
    df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })
    df["Abs"] = df["Coefficient"].abs()
    df = df.sort_values("Abs", ascending=False).head(top_n).sort_values("Coefficient", ascending=True)

    colors = [PALETTE["sage"] if c > 0 else PALETTE["coral"] for c in df["Coefficient"]]

    # Clean display names
    display_names = [f.replace("numeric__", "").replace("categorical__", "").replace("_", " ").title() for f in df["Feature"]]

    fig = go.Figure(go.Bar(
        x=df["Coefficient"],
        y=display_names,
        orientation='h',
        marker=dict(color=colors)
    ))

    fig.update_layout(
        title=f"Top {top_n} Ridge Model Drivers (Standardized Effect on Next-Cycle Score)",
        xaxis_title="Coefficient Value (Positive = Enhances Health, Negative = Diminishes Health)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(250, 245, 248, 0.5)',
        height=380,
        margin=dict(l=140, r=20, t=50, b=40),
        xaxis=dict(gridcolor='#EAE2E6')
    )
    return fig

