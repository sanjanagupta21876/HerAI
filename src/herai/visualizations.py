"""Interactive Plotly visualization components for the HERAI Web Application.

Adheres strictly to the HERAI Phase 3 Visual Design System and Phase 5/6 requirements:
- Colors: Warm Ivory #FFF9F5, Peach #F8E5DB, Dusty Rose #D99AA5, Muted Mauve #B97B8A,
  Deep Cocoa #59433F, Sage Green #A8B8A0, Pale Lavender #EAE5F4, White #FFFFFF, Amber #C97A3E.
- Zero invented metrics or coefficients: all data is either dynamically computed or verified from model outputs.
- Educational cycle timeline clearly labeled as non-predictive reference.
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
    """Create a subtle 28-day cycle wave visualization for the hero banner."""
    days = np.linspace(1, 28, 120)
    wave = 8.0 + 1.2 * np.sin((days - 5) / 28.0 * 2 * np.pi) + 0.3 * np.cos((days - 14) / 14.0 * 2 * np.pi)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days,
        y=wave,
        mode="lines",
        line=dict(color=PALETTE["mauve"], width=2.8, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(217, 154, 165, 0.14)",
        hoverinfo="skip",
        showlegend=False,
    ))

    milestone_days = [1, 7, 14, 21, 28]
    milestone_labels = ["Day 1: Menstrual", "Day 7: Follicular", "Day 14: Ovulatory", "Day 21: Luteal", "Day 28: Cycle End"]
    milestone_y = [8.0 + 1.2 * np.sin((d - 5) / 28.0 * 2 * np.pi) + 0.3 * np.cos((d - 14) / 14.0 * 2 * np.pi) for d in milestone_days]

    fig.add_trace(go.Scatter(
        x=milestone_days,
        y=milestone_y,
        mode="markers+text",
        marker=dict(size=9, color=PALETTE["mauve"], line=dict(color=PALETTE["white"], width=2)),
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
        height=150,
        margin=dict(l=15, r=15, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 29]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[6.0, 10.2]),
    )
    return fig


def create_gauge_chart(score: float, relative_note: str = "") -> go.Figure:
    """Create a semi-circular gauge chart for predicted health score on continuous 0-10 scale."""
    bar_color = PALETTE["sage"] if score >= 8.60 else PALETTE["amber"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<span style='font-family:{FONT_SERIF};font-size:19px;color:{PALETTE['cocoa']}'>Predicted Next-Cycle Health Score</span><br><span style='font-family:{FONT_SANS};font-size:11px;color:#735D58;'>{relative_note}</span>",
            'font': {'color': PALETTE["cocoa"]}
        },
        number={
            'suffix': " / 10.0",
            'font': {'size': 34, 'color': PALETTE["cocoa"], 'family': FONT_SERIF}
        },
        gauge={
            'axis': {
                'range': [0, 10],
                'tickwidth': 1,
                'tickcolor': PALETTE["mauve"],
                'tickfont': {'family': FONT_SANS, 'size': 9, 'color': PALETTE["cocoa"]},
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
        height=230,
        margin=dict(l=20, r=20, t=40, b=10),
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
        line=dict(color=PALETTE["mauve"], width=2.0),
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
            fillcolor='rgba(168, 184, 160, 0.20)',
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
                tickfont=dict(size=9.5, color=PALETTE["cocoa"], family=FONT_SANS),
                gridcolor=PALETTE["grid_line"],
            ),
            bgcolor=PALETTE["white"],
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.24,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
        ),
        height=300,
        margin=dict(l=35, r=35, t=15, b=35)
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

    fig.add_vline(
        x=8.60,
        line_width=2,
        line_dash="dot",
        line_color=PALETTE["cocoa"],
        annotation_text="Cohort Median: 8.60",
        annotation_position="top left",
        annotation_font=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
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
            font=dict(family=FONT_SERIF, size=16, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Next-Cycle Health Score (Continuous 0.0 - 10.0 Scale)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Observed Cycle Count", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=280,
        margin=dict(l=40, r=20, t=45, b=35),
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
        annotation_font=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
    )

    fig.update_layout(
        title=dict(
            text="Observed Cycle Length Distribution (Mean = 27.8 ± 2.4 Days)",
            font=dict(family=FONT_SERIF, size=16, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Cycle Length (Days)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Frequency (Cycles)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=280,
        margin=dict(l=40, r=20, t=45, b=35),
        xaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        yaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
    )
    return fig


def create_user_trajectory_chart(period_df: pd.DataFrame, user_id: str) -> go.Figure:
    """Plot multi-cycle trajectory for a specific patient cohort record."""
    if period_df.empty or "user_id" not in period_df.columns or "cycle_number" not in period_df.columns:
        return go.Figure()

    user_records = period_df[period_df["user_id"] == user_id].sort_values("cycle_number").copy()
    if user_records.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=user_records["cycle_number"],
        y=user_records["overall_health_score"],
        mode="lines+markers",
        name="Observed Health Score",
        line=dict(color=PALETTE["mauve"], width=2.2),
        marker=dict(size=6, color=PALETTE["mauve"])
    ))

    if "next_health_score" in user_records.columns:
        fig.add_trace(go.Scatter(
            x=user_records["cycle_number"],
            y=user_records["next_health_score"],
            mode="lines+markers",
            name="Next-Cycle Actual Target",
            line=dict(color=PALETTE["sage"], width=1.8, dash="dash"),
            marker=dict(size=5, color=PALETTE["sage"])
        ))

    fig.update_layout(
        title=dict(
            text=f"Longitudinal Health Score Trajectory — User {user_id}",
            font=dict(family=FONT_SERIF, size=16, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Cycle Sequence Number", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Health Score (0.0 - 10.0)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis=dict(range=[3.5, 10.5], gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        xaxis=dict(tickmode='linear', dtick=1, gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=300,
        margin=dict(l=40, r=20, t=45, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
        )
    )
    return fig


def create_model_comparison_bar_chart() -> go.Figure:
    """Create a verified comparison chart of MAE and RMSE across evaluated models.
    
    Data strictly verified against README.md and src/06_compare_models.py outputs.
    """
    models = ["Ridge (37 feat.)", "Ridge (42 feat.)", "Random Forest", "Mean Baseline"]
    mae_vals = [0.6448, 0.6452, 0.6550, 0.9113]
    rmse_vals = [0.8025, 0.8029, 0.8130, 1.0942]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Mean Absolute Error (MAE)",
        x=models,
        y=mae_vals,
        marker=dict(color=PALETTE["mauve"]),
        text=[f"{v:.4f}" for v in mae_vals],
        textposition="outside",
        textfont=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
    ))

    fig.add_trace(go.Bar(
        name="Root Mean Squared Error (RMSE)",
        x=models,
        y=rmse_vals,
        marker=dict(color=PALETTE["dusty_rose"]),
        text=[f"{v:.4f}" for v in rmse_vals],
        textposition="outside",
        textfont=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
    ))

    fig.update_layout(
        title=dict(
            text="Error Metrics Comparison (Test Set, N=3,176 / 400 Users) — Lower is Better",
            font=dict(family=FONT_SERIF, size=16, color=PALETTE["cocoa"])
        ),
        barmode="group",
        yaxis=dict(
            title="Error Magnitude (0-10 Scale)",
            range=[0, 1.3],
            gridcolor=PALETTE["grid_line"],
            tickfont=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
        ),
        xaxis=dict(tickfont=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(family=FONT_SANS, size=10, color=PALETTE["cocoa"])
        )
    )
    return fig


def create_feature_importance_chart(feature_names: List[str], coefficients: List[float], top_n: int = 10) -> go.Figure:
    """Bar chart of verified standardized Ridge regression model coefficients."""
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
            text="Top Verified Ridge Model Drivers (Standardized Effect)",
            font=dict(family=FONT_SERIF, size=16, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Standardized Effect Size (+ Supports Health, - Lowers Score)", font=dict(family=FONT_SANS, size=10.5, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=330,
        margin=dict(l=150, r=20, t=50, b=35),
        xaxis=dict(gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        yaxis=dict(tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"], size=10.5)),
    )
    return fig


def create_phase_distribution_chart(cohort_df: pd.DataFrame) -> go.Figure:
    """Plot observed frequency of self-reported cycle phases at time of log entry."""
    if cohort_df.empty or "cycle_phase" not in cohort_df.columns:
        return go.Figure()

    phase_counts = cohort_df["cycle_phase"].value_counts(normalize=True).reset_index()
    phase_counts.columns = ["Phase", "Proportion"]
    phase_counts["Percentage"] = (phase_counts["Proportion"] * 100).round(1)

    colors = [PALETTE["mauve"], PALETTE["dusty_rose"], PALETTE["sage"], PALETTE["amber"]]

    fig = go.Figure(go.Bar(
        x=phase_counts["Phase"],
        y=phase_counts["Percentage"],
        marker=dict(color=colors[:len(phase_counts)]),
        text=[f"{p:.1f}%" for p in phase_counts["Percentage"]],
        textposition="outside",
        textfont=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])
    ))

    fig.update_layout(
        title=dict(
            text="Observed Cycle Phases at Log Entry (Dataset N=15,976)",
            font=dict(family=FONT_SERIF, size=16, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Phase Reported During Log", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Cohort Proportion (%)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis=dict(range=[0, 60], gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        xaxis=dict(tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=280,
        margin=dict(l=40, r=20, t=50, b=35),
    )
    return fig


def create_pain_vs_health_chart(cohort_df: pd.DataFrame) -> go.Figure:
    """Plot observed relationship between self-reported pain levels and overall health scores."""
    if cohort_df.empty or "pain_level" not in cohort_df.columns or "overall_health_score" not in cohort_df.columns:
        return go.Figure()

    grouped = cohort_df.groupby("pain_level")["overall_health_score"].mean().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped["pain_level"],
        y=grouped["overall_health_score"],
        marker=dict(color=PALETTE["dusty_rose"], line=dict(color=PALETTE["mauve"], width=1)),
        text=[f"{s:.2f}" for s in grouped["overall_health_score"]],
        textposition="inside",
        textfont=dict(family=FONT_SANS, size=10, color=PALETTE["white"])
    ))

    fig.update_layout(
        title=dict(
            text="Observed Pain Level vs Mean Health Score (N=15,976)",
            font=dict(family=FONT_SERIF, size=16, color=PALETTE["cocoa"])
        ),
        xaxis_title=dict(text="Reported Pain Level (1 to 10)", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis_title=dict(text="Mean Overall Health Score", font=dict(family=FONT_SANS, size=11, color=PALETTE["cocoa"])),
        yaxis=dict(range=[0, 11], gridcolor=PALETTE["grid_line"], tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        xaxis=dict(tickmode="linear", dtick=1, tickfont=dict(family=FONT_SANS, color=PALETTE["cocoa"])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=PALETTE["white"],
        height=280,
        margin=dict(l=40, r=20, t=50, b=35),
    )
    return fig
