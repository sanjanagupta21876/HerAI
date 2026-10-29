"""Interactive Plotly visualization components for the HERAI Web Application.

Adheres strictly to the HERAI Phase 3 Visual Design System:
- Warm ivory background: #FFF9F5
- Light peach: #F8E5DB
- Dusty rose: #D99AA5
- Muted mauve: #B97B8A
- Deep cocoa text: #59433F
- Sage green: #A8B8A0
- Pale lavender: #EAE5F4
- White cards: #FFFFFF
- Restrained amber: #C97A3E
"""

from typing import Dict, Any, Optional, List
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Cohesive Brand Color Palette
PALETTE = {
    "ivory": "#FFF9F5",
    "peach": "#F8E5DB",
    "dusty_rose": "#D99AA5",
    "mauve": "#B97B8A",
    "cocoa": "#59433F",
    "sage": "#A8B8A0",
    "lavender": "#EAE5F4",
    "white": "#FFFFFF",
    "amber": "#C97A3E",
    "soft_red": "#BD606E",
    "card_border": "#F0E2DA",
    "grid_line": "#F4EBE6",
}

FONT_SERIF = "DM Serif Display, Georgia, serif"
FONT_SANS = "Inter, DM Sans, -apple-system, BlinkMacSystemFont, sans-serif"


def create_cycle_wave_hero() -> go.Figure:
    """Create a subtle, elegant 28-day cycle wave visualization for the hero section."""
    days = np.linspace(1, 28, 120)
    # Combined gentle hormone/temperature curve representing cycle rhythm
    wave = 8.0 + 1.2 * np.sin((days - 5) / 28.0 * 2 * np.pi) + 0.3 * np.cos((days - 14) / 14.0 * 2 * np.pi)

    fig = go.Figure()

    # Smooth curve
    fig.add_trace(go.Scatter(
        x=days,
        y=wave,
        mode="lines",
        line=dict(color=PALETTE["mauve"], width=2.8, shape="spline"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Phase milestone marker nodes
    milestone_days = [1, 7, 14, 21, 28]
    milestone_labels = ["Day 1: Menstrual", "Day 7: Follicular", "Day 14: Ovulatory", "Day 21: Luteal", "Day 28: Cycle Close"]
    milestone_y = [8.0 + 1.2 * np.sin((d - 5) / 28.0 * 2 * np.pi) + 0.3 * np.cos((d - 14) / 14.0 * 2 * np.pi) for d in milestone_days]

    fig.add_trace(go.Scatter(
        x=milestone_days,
        y=milestone_y,
        mode="markers+text",
        marker=dict(size=9, color=PALETTE["cocoa"], line=dict(color=PALETTE["peach"], width=2)),
        text=["Day 1", "Day 7", "Day 14", "Day 21", "Day 28"],
        textposition="top center",
        textfont=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"]),
        hovertext=milestone_labels,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=160,
        margin=dict(l=20, r=20, t=25, b=25),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 29]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[6.2, 10.0]
        ),
    )
    return fig


def create_gauge_chart(score: float, relative_note: str = "") -> go.Figure:
    """Create a calm, elegant semi-circular gauge chart for predicted health score (0-10 scale).
    
    Adheres strictly to Phase 4 constraints:
    - True 0.0 to 10.0 scale supported by dataset.
    - No invented percentages, probabilities, or clinical risk categories.
    - Displays cohort median marker (8.60).
    """
    bar_color = PALETTE["sage"] if score >= 8.60 else PALETTE["amber"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<span style='font-family:{FONT_SERIF};font-size:20px;color:{PALETTE['cocoa']}'>Predicted Next-Cycle Health Score</span><br><span style='font-family:{FONT_SANS};font-size:12px;color:#735D58;'>{relative_note}</span>",
            'font': {'color': PALETTE["cocoa"]}
        },
        number={
            'suffix': " / 10.0",
            'font': {'size': 36, 'color': PALETTE["cocoa"], 'family': FONT_SERIF}
        },
        gauge={
            'axis': {
                'range': [0, 10],
                'tickwidth': 1,
                'tickcolor': PALETTE["mauve"],
                'tickfont': {'family': FONT_SANS, 'size': 10, 'color': PALETTE["cocoa"]},
                'dtick': 2
            },
            'bar': {'color': bar_color, 'thickness': 0.32},
            'bgcolor': PALETTE["white"],
            'borderwidth': 1,
            'bordercolor': PALETTE["card_border"],
            'steps': [
                {'range': [0, 6.0], 'color': "#FAF0F2"},
                {'range': [6.0, 8.6], 'color': PALETTE["peach"]},
                {'range': [8.6, 10.0], 'color': "#EFF4EE"},
            ],
            'threshold': {
                'line': {'color': PALETTE["cocoa"], 'width': 3},
                'thickness': 0.8,
                'value': 8.60  # Observed cohort median
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=240,
        margin=dict(l=25, r=25, t=45, b=10),
    )
    return fig


def create_radar_chart(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]] = None
) -> go.Figure:
    """Create a balanced radar chart comparing 6 core dimensions of cycle observations (0-10 scale)."""
    categories = [
        "Comfort (Inverted Pain)",
        "Emotional Wellbeing",
        "Stress Resilience",
        "Sleep Sufficiency",
        "Vitality / Energy",
        "Cognitive Focus"
    ]

    curr_values = [
        max(0.0, 10.0 - current.get("pain_level", 3.0)),
        current.get("mood_score", 7.0),
        max(0.0, 10.0 - current.get("stress_score_cycle", 4.0)),
        min(10.0, (current.get("sleep_hours_cycle", 7.5) / 9.0) * 10.0),
        current.get("energy_level", 7.0),
        current.get("concentration_score", 7.0),
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=curr_values + [curr_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(217, 154, 165, 0.30)',
        line=dict(color=PALETTE["mauve"], width=2.2),
        name="Current Cycle Observations"
    ))

    if previous:
        prev_values = [
            max(0.0, 10.0 - previous.get("previous_pain_level", 3.0)),
            previous.get("previous_mood_score", 7.0),
            max(0.0, 10.0 - previous.get("previous_stress_score", 4.0)),
            min(10.0, (previous.get("previous_sleep_hours", 7.5) / 9.0) * 10.0),
            previous.get("previous_energy_level", 7.0),
            previous.get("previous_concentration_score", 7.0),
        ]
        fig.add_trace(go.Scatterpolar(
            r=prev_values + [prev_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(168, 184, 160, 0.22)',
            line=dict(color=PALETTE["sage"], width=1.8, dash='dot'),
            name="Previous Cycle (Baseline)"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=8, color=PALETTE["cocoa"], family=FONT_SANS),
                gridcolor=PALETTE["grid_line"],
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color=PALETTE["cocoa"], family=FONT_SANS),
                gridcolor=PALETTE["grid_line"],
            ),
            bgcolor=PALETTE["white"],
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])
        ),
        height=310,
        margin=dict(l=40, r=40, t=20, b=40)
    )
    return fig


def create_cohort_distribution_chart(scores_series: pd.Series, user_score: Optional[float] = None) -> go.Figure:
    """Show where a score sits relative to the empirical dataset distribution (N=15,976)."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores_series,
        nbinsx=32,
        marker=dict(
            color=PALETTE["dusty_rose"],
            line=dict(color=PALETTE["mauve"], width=1)
        ),
        name="Dataset Cohort (N=15,976)"
    ))

    # Cohort median reference line
    fig.add_vline(
        x=8.60,
        line_width=2,
        line_dash="dot",
        line_color=PALETTE["cocoa"],
        annotation_text="Cohort Median: 8.60",
        annotation_position="top left",
        annotation_font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])
    )

    if user_score is not None:
        fig.add_vline(
            x=user_score,
            line_width=2.5,
            line_dash="dash",
            line_color=PALETTE["amber"],
            annotation_text=f"Your Forecast: {user_score:.2f}",
            annotation_position="top right",
            annotation_font=dict(family=FONT_SANS, size=11, color=PALETTE["amber"])
        )

    fig.update_layout(
        title=dict(
            text="Distribution of Next-Cycle Health Scores in Dataset",
            font=dict(family=FONT_SERIF, size=17, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Next-Cycle Health Score (Continuous 0.0 - 10.0 Scale)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Observed Cycle Count", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=290,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        yaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
    )
    return fig


def create_cycle_length_distribution_chart(cycle_lengths: pd.Series) -> go.Figure:
    """Plot the distribution of recorded cycle lengths across all observed cycles."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=cycle_lengths,
        nbinsx=24,
        marker=dict(
            color=PALETTE["peach"],
            line=dict(color=PALETTE["dusty_rose"], width=1)
        ),
        name="Observed Cycle Lengths"
    ))

    fig.add_vline(
        x=28.0,
        line_width=2,
        line_dash="dash",
        line_color=PALETTE["mauve"],
        annotation_text="Median: 28.0 Days",
        annotation_position="top right",
        annotation_font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])
    )

    fig.update_layout(
        title=dict(
            text="Observed Cycle Length Distribution (Mean = 27.8 ± 2.4 Days)",
            font=dict(family=FONT_SERIF, size=17, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Cycle Length (Days)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Frequency (Cycles)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=290,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        yaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
    )
    return fig


def create_user_trajectory_chart(period_df: pd.DataFrame, user_id: str) -> go.Figure:
    """Plot multi-cycle trajectory for a specific patient cohort record."""
    user_records = period_df[period_df["user_id"] == user_id].sort_values("cycle_number").copy()
    if user_records.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=user_records["cycle_number"],
        y=user_records["overall_health_score"],
        mode="lines+markers",
        name="Observed Health Score",
        line=dict(color=PALETTE["mauve"], width=2.4),
        marker=dict(size=7, color=PALETTE["mauve"])
    ))

    if "next_health_score" in user_records.columns:
        fig.add_trace(go.Scatter(
            x=user_records["cycle_number"],
            y=user_records["next_health_score"],
            mode="lines+markers",
            name="Next-Cycle Actual Target",
            line=dict(color=PALETTE["sage"], width=2, dash="dash"),
            marker=dict(size=6, color=PALETTE["sage"])
        ))

    fig.update_layout(
        title=dict(
            text=f"Longitudinal Health Score Trajectory — User {user_id}",
            font=dict(family=FONT_SERIF, size=17, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Cycle Sequence Number", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Health Score (0.0 - 10.0)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis=dict(range=[3.5, 10.5], gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        xaxis=dict(tickmode='linear', dtick=1, gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=310,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])
        )
    )
    return fig


def create_feature_importance_chart(feature_names: List[str], coefficients: List[float], top_n: int = 10) -> go.Figure:
    """Bar chart of top standardized Ridge regression model coefficients."""
    df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })
    df["Abs"] = df["Coefficient"].abs()
    df = df.sort_values("Abs", ascending=False).head(top_n).sort_values("Coefficient", ascending=True)

    colors = [PALETTE["sage"] if c > 0 else PALETTE["soft_red"] for c in df["Coefficient"]]
    display_names = [f.replace("numeric__", "").replace("categorical__", "").replace("_", " ").title() for f in df["Feature"]]

    fig = go.Figure(go.Bar(
        x=df["Coefficient"],
        y=display_names,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0))
    ))

    fig.update_layout(
        title=dict(
            text="Top Standardized Ridge Regression Coefficients",
            font=dict(family=FONT_SERIF, size=17, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Standardized Effect Size (Positive = Elevates Score, Negative = Dampens Score)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=350,
        margin=dict(l=150, r=20, t=50, b=40),
        xaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        yaxis=dict(tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"], size=11)),
    )
    return fig
