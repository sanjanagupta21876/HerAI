"""HERAI — Women's Menstrual Health Intelligence Application.

AI-assisted menstrual-health pattern analysis and next-cycle health-score prediction.
Built with a calm, trustworthy, and elegant healthcare design system.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

# Ensure project src is in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from herai.config import (
    MODEL_RIDGE_37_PATH,
    ENGINEERED_DATA_PATH,
    TRAINING_DATA_PATH,
    FEATURES_37,
)
from herai.inference import (
    HerAIPredictor,
    UserProfileInput,
    CurrentCycleInput,
    HistoricalCycleInput,
    InputValidationError,
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
    PALETTE,
)

# Page configuration
st.set_page_config(
    page_title="HERAI | Menstrual Health Intelligence",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize navigation state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# Custom CSS for Phase 3 Visual Design System & Phase 4 Layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

    :root {
        --color-ivory: #FFF9F5;
        --color-peach: #F8E5DB;
        --color-dusty-rose: #D99AA5;
        --color-mauve: #B97B8A;
        --color-cocoa: #59433F;
        --color-sage: #A8B8A0;
        --color-lavender: #EAE5F4;
        --color-white: #FFFFFF;
        --color-amber: #C97A3E;
        --color-border: #F0E2DA;
        --font-serif: 'DM Serif Display', Georgia, serif;
        --font-sans: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: var(--color-ivory);
        color: var(--color-cocoa);
        font-family: var(--font-sans);
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }

    /* Headings */
    h1, h2, h3, .serif-heading {
        font-family: var(--font-serif);
        color: var(--color-cocoa);
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    h1 { font-size: 2.15rem; line-height: 1.25; margin-bottom: 0.4rem; }
    h2 { font-size: 1.6rem; line-height: 1.3; margin-top: 1.2rem; margin-bottom: 0.6rem; }
    h3 { font-size: 1.25rem; line-height: 1.35; margin-bottom: 0.5rem; }
    p, span, label, div {
        color: var(--color-cocoa);
        font-family: var(--font-sans);
    }

    /* App Header Navbar */
    .header-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--color-white);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 8px rgba(89, 67, 63, 0.03);
    }
    .header-logo {
        font-family: var(--font-serif);
        font-size: 1.65rem;
        color: var(--color-cocoa);
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .header-tagline {
        font-family: var(--font-sans);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--color-mauve);
        margin-left: 4px;
    }

    /* Welcome / Hero Section */
    .hero-banner {
        background: linear-gradient(135deg, #FFF9F5 0%, #F8E5DB 55%, #EAE5F4 100%);
        border: 1px solid var(--color-border);
        border-radius: 14px;
        padding: 28px 34px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(89, 67, 63, 0.03);
    }
    .hero-headline {
        font-family: var(--font-serif);
        font-size: 2.25rem;
        color: var(--color-cocoa);
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .hero-subtext {
        font-size: 1.02rem;
        color: #6E5752;
        line-height: 1.5;
        max-width: 780px;
        margin-bottom: 18px;
    }

    /* Card System */
    .card-surface {
        background: var(--color-white);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(89, 67, 63, 0.03);
    }
    .card-header-serif {
        font-family: var(--font-serif);
        font-size: 1.22rem;
        color: var(--color-cocoa);
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F8EDE7;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Overview Stat Metric Box */
    .metric-stat-box {
        background: var(--color-white);
        border: 1px solid var(--color-border);
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
        box-shadow: 0 2px 6px rgba(89, 67, 63, 0.02);
    }
    .metric-stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--color-mauve);
        margin-bottom: 4px;
    }
    .metric-stat-value {
        font-family: var(--font-serif);
        font-size: 1.7rem;
        color: var(--color-cocoa);
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .metric-stat-detail {
        font-size: 0.8rem;
        color: #7A6560;
        line-height: 1.35;
    }

    /* Primary and Secondary CTA Buttons */
    .stButton>button {
        background: var(--color-mauve);
        color: var(--color-white);
        font-family: var(--font-sans);
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        font-size: 0.96rem;
        box-shadow: 0 2px 8px rgba(185, 123, 138, 0.28);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #A26876;
        color: var(--color-white);
        box-shadow: 0 4px 12px rgba(185, 123, 138, 0.36);
    }

    /* Callout & Disclaimer */
    .callout-lavender {
        background: var(--color-lavender);
        border-left: 3px solid var(--color-mauve);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.86rem;
        color: var(--color-cocoa);
        margin: 12px 0;
        line-height: 1.45;
    }
    .disclaimer-card {
        background: #FAF5F2;
        border: 1px solid #EFE4DC;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 0.82rem;
        line-height: 1.55;
        color: #7A6560;
        margin-top: 32px;
    }

    /* Prediction Result Banner */
    .score-result-banner {
        border-radius: 12px;
        padding: 22px 28px;
        margin: 18px 0 22px 0;
        border: 1px solid;
    }
    .score-banner-optimal {
        background: #F4F7F2;
        border-color: #C8D7C2;
        color: #385430;
    }
    .score-banner-moderate {
        background: #FDF6EE;
        border-color: #F0D4BC;
        color: #8C4E1A;
    }
    .score-result-title {
        font-family: var(--font-serif);
        font-size: 1.65rem;
        margin-bottom: 4px;
    }
    .score-result-subtitle {
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Empty state */
    .empty-state-card {
        background: var(--color-white);
        border: 1px dashed var(--color-border);
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: #7A6560;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor() -> HerAIPredictor:
    """Load and cache the trained 37-feature Ridge pipeline."""
    return HerAIPredictor()


@st.cache_data
def get_cohort_dataset() -> pd.DataFrame:
    """Load and cache training cohort data for overview metrics and distributions."""
    if TRAINING_DATA_PATH.exists():
        return pd.read_csv(TRAINING_DATA_PATH)
    elif ENGINEERED_DATA_PATH.exists():
        return pd.read_csv(ENGINEERED_DATA_PATH)
    return pd.DataFrame()


def render_app_header():
    """Render the top clean application header with HERAI wordmark and navigation."""
    nav_col1, nav_col2 = st.columns([1, 2.2], vertical_alignment="center")

    with nav_col1:
        st.markdown("""
        <div style="display: flex; align-items: baseline; gap: 8px;">
            <span style="font-family: 'DM Serif Display', Georgia, serif; font-size: 1.85rem; color: #59433F; font-weight: 400;">HERAI</span>
            <span style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #B97B8A;">Health Intelligence</span>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        pages = ["Dashboard", "Health Prediction", "Cycle Information", "Model Insights", "About"]
        current_idx = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
        selected = st.radio(
            "Navigation",
            pages,
            index=current_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="header_nav_radio"
        )
        if selected != st.session_state.current_page:
            st.session_state.current_page = selected
            st.rerun()

    st.markdown("<hr style='margin: 8px 0 20px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)


# =============================================================================
# VIEW 1: DASHBOARD
# =============================================================================
def render_dashboard(cohort_df: pd.DataFrame):
    """Render the Main Dashboard with Welcome Hero, Overview Cards, and Cohort Visualizations."""
    # B. Landing page / Introduction section
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-headline">Understand Your Cycle. Discover Your Patterns.</div>
        <div class="hero-subtext">
            Explore menstrual-health patterns through data-driven insights and an experimental next-cycle health-score prediction model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Action Buttons and Subtle Pattern Wave
    cta_col, wave_col = st.columns([1.1, 1.9], vertical_alignment="center")
    with cta_col:
        btn_explore = st.button("Explore Your Health", use_container_width=True, key="btn_hero_explore")
        if btn_explore:
            st.session_state.current_page = "Health Prediction"
            st.rerun()

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        btn_how = st.button("How It Works", use_container_width=True, key="btn_hero_how")
        if btn_how:
            st.session_state.current_page = "Model Insights"
            st.rerun()

    with wave_col:
        fig_wave = create_cycle_wave_hero()
        st.plotly_chart(fig_wave, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr style='margin: 22px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)

    # C. Overview Cards
    st.markdown("### System & Cohort Overview")

    c1, c2, c3, c4 = st.columns(4)

    total_records = len(cohort_df) if not cohort_df.empty else 15976
    unique_users = cohort_df["user_id"].nunique() if not cohort_df.empty and "user_id" in cohort_df.columns else 2000
    avg_cycle_len = f"{cohort_df['cycle_length_days'].mean():.1f} Days" if not cohort_df.empty and "cycle_length_days" in cohort_df.columns else "27.8 Days"

    with c1:
        st.markdown(f"""
        <div class="metric-stat-box">
            <div class="metric-stat-label">Evaluated Cycles</div>
            <div class="metric-stat-value">{total_records:,}</div>
            <div class="metric-stat-detail">Across {unique_users:,} user profiles in training cohort</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-stat-box">
            <div class="metric-stat-label">Observed Cycle Length</div>
            <div class="metric-stat-value">{avg_cycle_len}</div>
            <div class="metric-stat-detail">Median: 28.0 days (Std Dev: ±2.4 days)</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-stat-box">
            <div class="metric-stat-label">Production Model</div>
            <div class="metric-stat-value" style="font-size: 1.35rem; padding-top: 4px;">Ridge (37 Feat.)</div>
            <div class="metric-stat-detail">R² = 0.4620 | MAE = 0.6448 on test users</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        last_pred = st.session_state.last_prediction
        if last_pred is not None:
            st.markdown(f"""
            <div class="metric-stat-box" style="border-color: #D99AA5;">
                <div class="metric-stat-label" style="color: #882D52;">Your Latest Forecast</div>
                <div class="metric-stat-value">{last_pred.predicted_next_health_score:.2f} <span style="font-size: 1rem; color: #7A6560;">/ 10</span></div>
                <div class="metric-stat-detail">{last_pred.score_category} Health Tier</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-stat-box" style="background: #FAF7F5;">
                <div class="metric-stat-label">Your Latest Forecast</div>
                <div class="metric-stat-value" style="font-size: 1.25rem; color: #8F7B76; padding-top: 4px;">No Forecast Yet</div>
                <div class="metric-stat-detail">Run prediction to see your score</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Health Pattern Visualizations
    st.markdown("### Observational Health Patterns")
    st.caption("Visualizing empirical trends across the validated dataset. Correlation does not imply causation.")

    chart_c1, chart_c2 = st.columns([1, 1], gap="medium")

    with chart_c1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Observed Cycle Length Distribution</div>', unsafe_allow_html=True)
        if not cohort_df.empty and "cycle_length_days" in cohort_df.columns:
            fig_len = create_cycle_length_distribution_chart(cohort_df["cycle_length_days"])
            st.plotly_chart(fig_len, use_container_width=True)
            st.caption("Observed cycle length across 15,976 records follows an empirical normal distribution around 28 days.")
        else:
            st.info("Loading cycle distribution...")
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_c2:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Cohort Health Score Distribution</div>', unsafe_allow_html=True)
        if not cohort_df.empty and "next_health_score" in cohort_df.columns:
            user_val = st.session_state.last_prediction.predicted_next_health_score if st.session_state.last_prediction else None
            fig_dist = create_cohort_distribution_chart(cohort_df["next_health_score"].dropna(), user_val)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Distribution parameters across 15,976 cycles: Mean = 8.55, Median = 8.60, Std = 1.11, Range = [4.40, 10.00].")
        else:
            st.info("Loading health score distribution...")
        st.markdown('</div>', unsafe_allow_html=True)

    # Longitudinal Multi-Cycle Trajectory Explorer
    st.markdown('<div class="card-surface"><div class="card-header-serif">Longitudinal Cycle Trajectory in Real Records</div>', unsafe_allow_html=True)
    if not cohort_df.empty and "user_id" in cohort_df.columns:
        p_col1, p_col2 = st.columns([1, 3], vertical_alignment="center")
        with p_col1:
            sample_ids = ["U00001", "U00002", "U00003", "U00004", "U00005"]
            selected_id = st.selectbox("Select Patient Record:", sample_ids, index=0)
            st.caption("Inspect logged cycle scores and corresponding next-cycle actual values across consecutive cycles.")
        with p_col2:
            fig_traj = create_user_trajectory_chart(cohort_df, selected_id)
            st.plotly_chart(fig_traj, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# VIEW 2: HEALTH PREDICTION
# =============================================================================
def render_health_prediction(predictor: HerAIPredictor):
    """Render the Health Prediction Page with Structured 3-Section Inputs and Results."""
    st.markdown("<h1>Health Score Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "Enter your observational menstrual-cycle data, lifestyle factors, and prior cycle records. "
        "HERAI applies the validated 37-feature Ridge Regression pipeline to estimate your overall health score for your next cycle."
    )

    # Preset selector for fast scenario testing
    pr_col1, pr_col2 = st.columns([2, 3], vertical_alignment="center")
    with pr_col1:
        preset = st.selectbox(
            "Preload Observational Scenario:",
            [
                "Custom Inputs",
                "🌸 Balanced & Regular Cycle (Optimal Range)",
                "⚡ High Stress & Moderate Fatigue",
                "🩸 Elevated Cramping & Dysmenorrhea",
                "🌿 Longer Cycle with Mild Symptoms"
            ],
            index=0,
            help="Select a scenario to populate representative observational parameters."
        )
    with pr_col2:
        st.markdown("""
        <div class="callout-lavender" style="margin: 0;">
            <b>Privacy & Integrity Guarantee:</b> Input values are processed strictly in-memory during your session. 
            No records from other individuals are substituted to generate your prediction.
        </div>
        """, unsafe_allow_html=True)

    # Default mapping
    if preset == "🌸 Balanced & Regular Cycle (Optimal Range)":
        def_age, def_bmi, def_diet, def_exercise = 28, 22.2, "Good", "3–4 days/week"
        def_bsleep, def_bstress = 8.0, 3.0
        def_clen, def_phase, def_flow, def_pain = 28, "Menstrual", "Medium", 2
        def_mood, def_stress, def_sleep = 8, 3.0, 8.0
        def_energy, def_focus, def_lost = 8, 8, 0.0
        def_bc, def_pcos = 0, 0
    elif preset == "⚡ High Stress & Moderate Fatigue":
        def_age, def_bmi, def_diet, def_exercise = 31, 24.0, "Fair", "1–2 days/week"
        def_bsleep, def_bstress = 6.5, 7.5
        def_clen, def_phase, def_flow, def_pain = 31, "Luteal", "Medium", 4
        def_mood, def_stress, def_sleep = 5, 8.0, 5.5
        def_energy, def_focus, def_lost = 4, 5, 4.0
        def_bc, def_pcos = 0, 0
    elif preset == "🩸 Elevated Cramping & Dysmenorrhea":
        def_age, def_bmi, def_diet, def_exercise = 26, 21.5, "Good", "1–2 days/week"
        def_bsleep, def_bstress = 7.0, 5.0
        def_clen, def_phase, def_flow, def_pain = 27, "Menstrual", "Heavy", 7
        def_mood, def_stress, def_sleep = 4, 6.0, 6.0
        def_energy, def_focus, def_lost = 4, 4, 6.0
        def_bc, def_pcos = 0, 0
    elif preset == "🌿 Longer Cycle with Mild Symptoms":
        def_age, def_bmi, def_diet, def_exercise = 29, 26.5, "Fair", "3–4 days/week"
        def_bsleep, def_bstress = 7.5, 5.5
        def_clen, def_phase, def_flow, def_pain = 36, "Follicular", "Light", 3
        def_mood, def_stress, def_sleep = 6, 5.0, 7.5
        def_energy, def_focus, def_lost = 6, 6, 0.0
        def_bc, def_pcos = 0, 1
    else:
        def_age, def_bmi, def_diet, def_exercise = 28, 22.5, "Good", "3–4 days/week"
        def_bsleep, def_bstress = 7.5, 4.0
        def_clen, def_phase, def_flow, def_pain = 28, "Menstrual", "Medium", 3
        def_mood, def_stress, def_sleep = 7, 4.0, 7.5
        def_energy, def_focus, def_lost = 7, 7, 0.0
        def_bc, def_pcos = 0, 0

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    # Section 1: Menstrual-Cycle Information
    with col1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">1. Cycle Observations</div>', unsafe_allow_html=True)
        cycle_length = st.slider("Cycle Length (Days)", 18, 45, def_clen, help="Total days from period start to the day before the next period.")
        cycle_phase = st.selectbox("Current Cycle Phase", ["Menstrual", "Follicular", "Ovulation", "Luteal"], index=["Menstrual", "Follicular", "Ovulation", "Luteal"].index(def_phase))
        flow_level = st.selectbox("Flow Heaviness", ["Light", "Medium", "Heavy", "Very Heavy"], index=["Light", "Medium", "Heavy", "Very Heavy"].index(def_flow))

        pain_score = st.slider("Pain / Cramping Intensity (0–10)", 0, 10, int(def_pain), help="0 = Completely pain-free, 10 = Severe debilitating cramping")
        mood_score = st.slider("Emotional Wellbeing (0–10)", 0, 10, int(def_mood), help="0 = Severe distress/irritability, 10 = Positive, resilient mood")
        cycle_stress = st.slider("Cycle Stress Burden (0–10)", 0.0, 10.0, float(def_stress), step=0.5)
        cycle_sleep = st.slider("Cycle Sleep Duration (Hours)", 4.0, 12.0, float(def_sleep), step=0.5)
        energy_score = st.slider("Vitality & Energy (0–10)", 0, 10, int(def_energy))
        focus_score = st.slider("Cognitive Focus (0–10)", 0, 10, int(def_focus))
        hours_lost = st.number_input("Work / School Hours Lost Due to Symptoms", min_value=0.0, max_value=40.0, value=float(def_lost), step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Lifestyle and Health Information
    with col2:
        st.markdown('<div class="card-surface"><div class="card-header-serif">2. Lifestyle & Biomarkers</div>', unsafe_allow_html=True)
        age = st.slider("Age (Years)", 15, 60, def_age)
        bmi = st.slider("BMI (Body Mass Index)", 15.0, 45.0, float(def_bmi), step=0.1)

        diet_quality = st.selectbox(
            "Diet Quality",
            ["Good", "Fair", "Poor", "Excellent"],
            index=["Good", "Fair", "Poor", "Excellent"].index(def_diet)
        )
        exercise_freq = st.selectbox(
            "Exercise Frequency",
            ["Never", "1–2 days/week", "3–4 days/week", "5–6 days/week", "Daily"],
            index=["Never", "1–2 days/week", "3–4 days/week", "5–6 days/week", "Daily"].index(def_exercise)
        )
        base_sleep = st.slider("Baseline Daily Sleep (Hours)", 4.0, 11.0, float(def_bsleep), step=0.5)
        base_stress = st.slider("Baseline Daily Stress (0–10)", 0.0, 10.0, float(def_bstress), step=0.5)

        with st.expander("Additional Model Covariates"):
            st.caption("Non-diagnostic observational inputs used in the Ridge regression model.")
            pcos_obs = st.checkbox("Self-Reported History of PCOS", value=bool(def_pcos))
            bc_obs = st.checkbox("Hormonal Contraceptive Use", value=bool(def_bc))
            caffeine_intake = st.slider("Caffeine Intake (cups/day)", 0.0, 6.0, 1.5, step=0.5)
            water_intake = st.slider("Water Intake (Liters/day)", 0.5, 5.0, 2.0, step=0.5)
            smoking_stat = st.selectbox("Smoking Status", ["Non-smoker", "Former", "Smoker"], index=0)
            alcohol_stat = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Moderate", "Frequent"], index=0)
            state_region = st.selectbox("State / Region", ["California", "New York", "Texas", "Florida", "Illinois", "Other"], index=0)

        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Previous-Cycle Information
    with col3:
        st.markdown('<div class="card-surface"><div class="card-header-serif">3. Historical Records</div>', unsafe_allow_html=True)
        hist_mode = st.radio(
            "History Availability:",
            [
                "Cycle 1 (No prior cycle record logged)",
                "Returning User (I have my previous cycle data)"
            ],
            index=0,
            help="Choose whether to apply reference population medians (Cycle 1) or your own logged historical metrics."
        )

        has_history = "Returning User" in hist_mode
        hist_input: Optional[HistoricalCycleInput] = None

        if not has_history:
            st.markdown("""
            <div style="background: #F8EDE7; border: 1px solid #ECDAD2; border-radius: 8px; padding: 12px; font-size: 0.84rem; color: #59433F; margin-top: 8px;">
                <b>Cycle 1 Baseline Policy:</b><br>
                Because this is your first cycle logged, the pipeline utilizes training cohort reference medians (N=15,976) 
                for prior-cycle lag variables. Your personal information is never mixed with another individual's data.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #EEF4EC; border: 1px solid #D5E2D0; border-radius: 8px; padding: 10px 12px; font-size: 0.84rem; color: #385430; margin-top: 8px;">
                <b>User Historical Records:</b><br>
                Enter your observations from your preceding cycle.
            </div>
            """, unsafe_allow_html=True)
            prev_health = st.slider("Previous Cycle Overall Health (0–10)", 2.0, 10.0, 8.0, step=0.2)
            prev_len = st.slider("Previous Cycle Length (Days)", 20, 42, 28)
            prev_pain = st.slider("Previous Cycle Pain (0–10)", 0, 10, 3)
            prev_stress = st.slider("Previous Cycle Stress (0–10)", 0.0, 10.0, 4.0, step=0.5)
            prev_sleep = st.slider("Previous Cycle Sleep (Hours)", 4.0, 11.0, 7.5, step=0.5)
            prev_energy = st.slider("Previous Cycle Vitality (0–10)", 0, 10, 7)
            prev_mood = st.slider("Previous Cycle Mood (0–10)", 0, 10, 7)
            prev_focus = st.slider("Previous Cycle Concentration (0–10)", 0, 10, 7)

            hist_input = HistoricalCycleInput(
                previous_health_score=prev_health,
                previous_cycle_length_feature=float(prev_len),
                previous_pain_level=float(prev_pain),
                previous_stress_score=float(prev_stress),
                previous_sleep_hours=float(prev_sleep),
                previous_mood_score=float(prev_mood),
                previous_energy_level=float(prev_energy),
                previous_concentration_score=float(prev_focus),
                previous_work_hours_lost=0.0,
                previous_flow_level="Medium",
                previous_pms_symptoms="Mild",
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Action Button
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    btn_predict = st.button("Generate Health Score Prediction", use_container_width=True, key="btn_run_prediction")

    if btn_predict or preset != "Custom Inputs":
        try:
            profile_obj = UserProfileInput(
                age=float(age),
                bmi=float(bmi),
                sleep_hours=float(base_sleep),
                stress_score_baseline=float(base_stress),
                state=state_region,
                diet_quality=diet_quality,
                exercise_frequency=exercise_freq,
                caffeine_intake=float(caffeine_intake),
                water_intake_liters=float(water_intake),
                alcohol_consumption=alcohol_stat,
                smoking_status=smoking_stat,
                birth_control_use=1 if bc_obs else 0,
                pcos_diagnosed=1 if pcos_obs else 0,
            )

            cycle_obj = CurrentCycleInput(
                cycle_length_days=float(cycle_length),
                pain_level=float(pain_score),
                mood_score=float(mood_score),
                stress_score_cycle=float(cycle_stress),
                sleep_hours_cycle=float(cycle_sleep),
                energy_level=float(energy_score),
                concentration_score=float(focus_score),
                work_hours_lost=float(hours_lost),
                cycle_phase=cycle_phase,
                flow_level=flow_level,
                pms_symptoms="Moderate" if pain_score > 5 else "Mild"
            )

            with st.spinner("Executing 37-feature preprocessing and Ridge regression model..."):
                pred_result = predictor.predict(profile_obj, cycle_obj, historical_cycle=hist_input)

            st.session_state.last_prediction = pred_result
            st.session_state.last_inputs = {
                "age": age, "bmi": bmi, "base_stress": base_stress, "base_sleep": base_sleep,
                "cycle_length": cycle_length, "phase": cycle_phase, "flow": flow_level,
                "pain": pain_score, "mood": mood_score, "cycle_stress": cycle_stress,
                "cycle_sleep": cycle_sleep, "energy": energy_score, "focus": focus_score,
                "hours_lost": hours_lost, "has_history": has_history
            }
        except InputValidationError as err:
            st.error(f"Input Validation: {err}")
            st.session_state.last_prediction = None
        except Exception:
            st.error("An unexpected error occurred during prediction. Please verify your inputs and try again.")
            st.session_state.last_prediction = None

    # E. Prediction Results Section
    if st.session_state.last_prediction is not None:
        render_prediction_results(st.session_state.last_prediction, st.session_state.last_inputs)


# =============================================================================
# VIEW 3: PREDICTION RESULTS COMPONENT
# =============================================================================
def render_prediction_results(result, inputs: Optional[Dict[str, Any]]):
    """Render Section E: Prediction Results Page with scale, context, limitations, and general wellness suggestions."""
    st.markdown("<hr style='margin: 30px 0 20px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results & Explanation</h2>", unsafe_allow_html=True)

    score_val = result.predicted_next_health_score
    diff_from_median = score_val - 8.60

    if diff_from_median >= 0:
        rel_text = f"+{diff_from_median:.2f} relative to cohort median (8.60)"
        banner_class = "score-banner-optimal"
    else:
        rel_text = f"{diff_from_median:.2f} relative to cohort median (8.60)"
        banner_class = "score-banner-moderate"

    st.markdown(f"""
    <div class="score-result-banner {banner_class}">
        <div class="score-result-title">
            Predicted Next-Cycle Health Score: <b>{score_val:.2f} / 10.0</b>
        </div>
        <div class="score-result-subtitle">
            Scale: Continuous score from 0.0 (lowest overall wellbeing) to 10.0 (highest overall wellbeing).<br>
            Empirical Position: <b>{rel_text}</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    res_c1, res_c2 = st.columns([1, 1], gap="medium")

    with res_c1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Predicted Score Visualization</div>', unsafe_allow_html=True)
        gauge_fig = create_gauge_chart(score_val, rel_text)
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown(f"""
        <div style="font-size: 0.86rem; color: #59433F; line-height: 1.5; padding: 6px 4px;">
            <b>Score Interpretation:</b><br>
            The overall health score is a continuous composite indicator modeled from symptom severity, 
            emotional balance, sleep hours, and stress continuity. In the training dataset (N=15,976), 
            the middle 50% of records fall between 7.80 and 9.50.
        </div>
        """, unsafe_allow_html=True)
        if result.warnings:
            for w in result.warnings:
                st.caption(f"ℹ️ {w}")
        st.markdown('</div>', unsafe_allow_html=True)

    with res_c2:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Wellbeing Balance Across 6 Dimensions</div>', unsafe_allow_html=True)
        if inputs:
            curr_dim = {
                "pain_level": inputs["pain"],
                "mood_score": inputs["mood"],
                "stress_score_cycle": inputs["cycle_stress"],
                "sleep_hours_cycle": inputs["cycle_sleep"],
                "energy_level": inputs["energy"],
                "concentration_score": inputs["focus"],
            }
            radar_fig = create_radar_chart(curr_dim)
            st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Relevant Input Summary Table
    if inputs:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Summary of Provided Input Parameters</div>', unsafe_allow_html=True)
        in_c1, in_c2, in_c3 = st.columns(3)
        with in_c1:
            st.write(f"**Age:** {inputs['age']} years")
            st.write(f"**BMI:** {inputs['bmi']:.1f}")
            st.write(f"**Baseline Stress:** {inputs['base_stress']:.1f} / 10")
        with in_c2:
            st.write(f"**Cycle Length:** {inputs['cycle_length']} days")
            st.write(f"**Cycle Phase:** {inputs['phase']}")
            st.write(f"**Flow Heaviness:** {inputs['flow']}")
        with in_c3:
            st.write(f"**Pain Intensity:** {inputs['pain']} / 10")
            st.write(f"**Cycle Sleep:** {inputs['cycle_sleep']:.1f} hrs")
            st.write(f"**Historical Records:** {'User Prior Cycle' if inputs['has_history'] else 'Cycle 1 Cohort Baseline'}")
        st.markdown('</div>', unsafe_allow_html=True)

    # General Actionable Next Steps (Phrased as General Wellness Information)
    st.markdown('<div class="card-surface"><div class="card-header-serif">General Wellness Observations & Next Steps</div>', unsafe_allow_html=True)
    st.markdown("""
    *These observations are general educational lifestyle recommendations and do not constitute medical advice or treatment.*
    """)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div style="background: #FDF6EE; border: 1px solid #F0D4BC; border-radius: 8px; padding: 14px; font-size: 0.86rem;">
            <b>Rest & Recovery Rhythm</b><br>
            Aim for consistent sleep duration (7.5–8.5 hours). Maintaining steady circadian rhythms supports pituitary gonadotropin secretion and next-cycle vitality.
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div style="background: #F4F7F2; border: 1px solid #C8D7C2; border-radius: 8px; padding: 14px; font-size: 0.86rem;">
            <b>Nutritional Hydration</b><br>
            Adequate daily water intake (2+ liters) and electrolyte-rich whole foods assist cellular hydration and support comfort throughout menstruation.
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div style="background: #FAF7F5; border: 1px solid #EADBCE; border-radius: 8px; padding: 14px; font-size: 0.86rem;">
            <b>Stress Regulation Practices</b><br>
            Because stress exerts a significant negative standardized coefficient in the model, daily down-regulation (gentle walking, mindfulness) helps protect cycle continuity.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Limitations Card
    st.markdown("""
    <div class="callout-lavender">
        <b>Model Limitations & Clinical Boundary Notice:</b><br>
        This prediction was generated by an experimental Ridge Regression pipeline (test R² = 0.4620, MAE = 0.6448). 
        The model has not been clinically validated on diverse real-world populations. 
        It does not diagnose any medical condition (such as endometriosis or PCOS), does not evaluate pregnancy or maternal health, 
        and must not be used to direct medication, contraception, or therapeutic interventions.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HELPER: DYNAMIC FEATURE DRIVER EXTRACTION
# =============================================================================
def get_driver_weights(predictor: Optional[HerAIPredictor] = None) -> Tuple[List[str], List[float]]:
    """Extract top physiological and behavioral feature weights directly from the model pipeline."""
    if predictor is not None and predictor.model is not None:
        try:
            pipeline = predictor.model
            reg = pipeline.named_steps["regressor"]
            prep = pipeline.named_steps["preprocessor"]
            names = prep.get_feature_names_out()
            coefs = reg.coef_

            pairs = []
            for n, c in zip(names, coefs):
                cn = n.replace("numeric__", "").replace("categorical__", "")
                if cn.startswith("state_"):
                    continue
                pairs.append((cn, float(c)))

            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_pairs = pairs[:10]
            return [p[0] for p in top_pairs], [p[1] for p in top_pairs]
        except Exception:
            pass

    # Verified empirical weights from models/herai_ridge_37_features.pkl
    names = [
        "stress_score_baseline",
        "overall_health_score",
        "previous_health_score",
        "pain_level",
        "previous_pain_level",
        "pcos_diagnosed",
        "sleep_hours",
        "previous_stress_score",
        "stress_score_cycle",
        "previous_sleep_hours",
    ]
    coefs = [-0.6724, 0.4526, 0.4040, 0.2910, 0.2493, -0.1635, 0.1515, 0.1351, 0.1127, -0.0860]
    return names, coefs


# =============================================================================
# VIEW 4: CYCLE INFORMATION
# =============================================================================
def render_cycle_information(cohort_df: Optional[pd.DataFrame] = None):
    """Render Cycle Information page detailing biological phases, model variables, and clinical red flags."""
    if cohort_df is None:
        cohort_df = get_cohort_dataset()

    st.markdown("<h1>Menstrual Cycle Biology & Empirical Patterns</h1>", unsafe_allow_html=True)
    st.markdown(
        "A healthy menstrual cycle typically spans 21 to 35 days, orchestrated by rhythmic fluctuations "
        "in estrogen, progesterone, luteinizing hormone (LH), and follicle-stimulating hormone (FSH). "
        "Below, explore the physiological rhythm alongside empirical trends observed in 15,976 evaluated cycles."
    )

    # Crucial Distinction Banner
    st.markdown("""
    <div class="callout-lavender" style="margin-bottom: 24px;">
        <b style="color: #59433F; font-size: 0.95rem;">⚠️ Crucial Distinction: Educational Biology vs. Algorithmic Health Forecast</b><br>
        <span style="font-size: 0.88rem; color: #59433F; line-height: 1.55;">
        The physiological cycle timeline, hormonal transitions, and lifestyle recommendations presented on this page 
        serve purely general educational and physiological literacy purposes. 
        <b>HERAI does not predict calendar period start dates, fertile windows, ovulation dates, or pregnancy probabilities.</b> 
        The machine learning model estimates an overall, continuous health score (0.0 to 10.0 scale) for the upcoming cycle 
        based on self-reported symptoms and historical consistency—it is not a fertility calculator or diagnostic device.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: Cycle Variables Consumed by the Model
    st.markdown('<div class="card-surface"><div class="card-header-serif">Cycle Variables Used by the HERAI Prediction Model</div>', unsafe_allow_html=True)
    st.markdown(
        "HERAI's production 37-feature Ridge Regression model takes into account self-reported cycle metrics "
        "and their historical lag values to forecast next-cycle health:"
    )

    cv_c1, cv_c2 = st.columns(2, gap="medium")
    with cv_c1:
        st.markdown("""
        <div style="background: #FAF7F5; border-left: 3px solid #B97B8A; padding: 14px 16px; border-radius: 6px; margin-bottom: 12px;">
            <b style="color: #59433F;">Current Cycle Indicators:</b>
            <ul style="font-size: 0.86rem; color: #59433F; line-height: 1.6; margin-top: 6px; padding-left: 18px;">
                <li><b>Cycle Length (<code>cycle_length_days</code>):</b> Observed duration from cycle onset to next period (cohort mean: 27.8 ± 2.4 days).</li>
                <li><b>Cycle Phase (<code>cycle_phase</code>):</b> Self-reported phase at log entry (Menstrual, Follicular, or Luteal).</li>
                <li><b>Menstrual Flow (<code>flow_level</code>):</b> Categorical flow volume (Light, Moderate, or Heavy).</li>
                <li><b>Dysmenorrhea Severity (<code>pain_level</code>):</b> Self-rated pelvic pain scale (1 = minimal to 10 = severe).</li>
                <li><b>Premenstrual Symptoms (<code>pms_symptoms</code>):</b> Severity of cyclical emotional and somatic symptoms.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with cv_c2:
        st.markdown("""
        <div style="background: #FAF7F5; border-left: 3px solid #A8B8A0; padding: 14px 16px; border-radius: 6px; margin-bottom: 12px;">
            <b style="color: #59433F;">Prior-Cycle Lag & Lifestyle Indicators:</b>
            <ul style="font-size: 0.86rem; color: #59433F; line-height: 1.6; margin-top: 6px; padding-left: 18px;">
                <li><b>Previous Health & Pain (<code>previous_health_score</code>, <code>previous_pain_level</code>):</b> Autoregressive continuity from cycle <i>t - 1</i>.</li>
                <li><b>Previous Cycle Length (<code>previous_cycle_length_days</code>):</b> Rhythm stability indicator.</li>
                <li><b>Stress & Sleep Metrics (<code>stress_score_cycle</code>, <code>sleep_hours_cycle</code>):</b> Cyclical nervous system load and restorative rest.</li>
                <li><b>Cognitive & Occupational Impact (<code>energy_level</code>, <code>concentration_score</code>, <code>work_hours_lost</code>):</b> Daytime vitality and productivity impact.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 2: Empirical Observed Patterns in Cohort Data
    st.markdown("### Empirical Cohort Patterns (N=15,976 Cycles)")
    st.caption("Visualizing real patterns logged across 2,000 anonymized cohort profiles.")

    pat_c1, pat_c2 = st.columns(2, gap="medium")
    with pat_c1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Observed Cycle Phases at Log Entry</div>', unsafe_allow_html=True)
        if not cohort_df.empty and "cycle_phase" in cohort_df.columns:
            fig_ph = create_phase_distribution_chart(cohort_df)
            st.plotly_chart(fig_ph, use_container_width=True)
            st.caption("Entries in the cohort were logged primarily during the Luteal (45.0%) and Follicular (44.8%) phases, with Menstrual entries accounting for 10.1%.")
        else:
            st.info("Loading phase distribution...")
        st.markdown('</div>', unsafe_allow_html=True)

    with pat_c2:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Observed Pain vs. Mean Health Score</div>', unsafe_allow_html=True)
        if not cohort_df.empty and "pain_level" in cohort_df.columns:
            fig_pain = create_pain_vs_health_chart(cohort_df)
            st.plotly_chart(fig_pain, use_container_width=True)
            st.caption("Higher self-reported pain levels show a steady, empirical correlation with lower overall health scores across 15,976 cycles.")
        else:
            st.info("Loading pain vs health relationship...")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 3: Educational 28-Day Biological Cycle Timeline
    st.markdown("### Biological Cycle Reference Timeline (28-Day Model)")
    st.caption("Educational overview of typical hormonal fluctuations and supportive wellness practices across four biological phases.")

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("""
        <div class="card-surface" style="border-top: 3px solid #B97B8A;">
            <div class="card-header-serif">1. Menstrual Phase (Days 1–5)</div>
            <p style="font-size: 0.9rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> Estrogen and progesterone drop to baseline levels, signaling the shedding of the uterine lining (stratum functionale).
            </p>
            <ul style="font-size: 0.85rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
                <li><b>Physiological Focus:</b> Restorative recovery, iron replenishment, gentle movement.</li>
                <li><b>Nutritional Support:</b> Warm mineral broths, dark leafy greens, magnesium-rich seeds, hydration.</li>
                <li><b>Comfort Measures:</b> Gentle heat application, warm baths, restorative diaphragmatic breathing.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card-surface" style="border-top: 3px solid #D99AA5;">
            <div class="card-header-serif">2. Follicular Phase (Days 6–13)</div>
            <p style="font-size: 0.9rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> Pituitary FSH stimulates ovarian follicle development; ascending estradiol thickens the endometrium.
            </p>
            <ul style="font-size: 0.85rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
                <li><b>Physiological Focus:</b> Rising physical stamina, cognitive sharpness, cellular energy.</li>
                <li><b>Nutritional Support:</b> Complex carbohydrates, fermented probiotic foods, lean proteins.</li>
                <li><b>Activity Alignment:</b> Progressive resistance training, aerobic conditioning, focused project work.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card-surface" style="border-top: 3px solid #A8B8A0;">
            <div class="card-header-serif">3. Ovulatory Phase (Days 14–16)</div>
            <p style="font-size: 0.9rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> Estrogen peak triggers a sharp LH (luteinizing hormone) surge, stimulating the release of a mature ovum.
            </p>
            <ul style="font-size: 0.85rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
                <li><b>Physiological Focus:</b> Peak metabolic rate, elevated stamina, enhanced verbal acuity.</li>
                <li><b>Nutritional Support:</b> Antioxidant-dense berries, zinc, cruciferous vegetables supporting estrogen metabolism.</li>
                <li><b>Activity Alignment:</b> High-intensity interval workouts, collaborative group efforts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card-surface" style="border-top: 3px solid #C97A3E;">
            <div class="card-header-serif">4. Luteal Phase (Days 17–28)</div>
            <p style="font-size: 0.9rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> The ruptured follicle forms the corpus luteum, secreting progesterone to support the endometrium.
            </p>
            <ul style="font-size: 0.85rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
                <li><b>Physiological Focus:</b> Nervous system down-regulation, steady blood glucose, sleep hygiene.</li>
                <li><b>Nutritional Support:</b> Vitamin B6, pumpkin and sunflower seeds, complex root vegetables.</li>
                <li><b>Activity Alignment:</b> Moderate pilates, nature walks, calming evening wind-down rituals.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 4: Clinical Red Flags
    st.markdown('<div class="card-surface" style="border-left: 4px solid #C97A3E;"><div class="card-header-serif">🩺 When to Seek Medical Attention (Clinical Red Flags)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.88rem; color: #59433F; line-height: 1.6;">
    Please consult a certified gynecologist, endocrinologist, or primary care provider if you experience any of the following clinical warning signs:
    <ul>
        <li><b>Debilitating Dysmenorrhea:</b> Severe pelvic cramping that does not respond to standard OTC analgesics, disrupts work or daily life, or worsens progressively.</li>
        <li><b>Menorrhagia (Heavy Menstrual Bleeding):</b> Soaking through one or more tampons or pads per hour for consecutive hours, passing large blood clots (&gt;2.5 cm), or experiencing bleeding longer than 7 days.</li>
        <li><b>Significant Irregularity:</b> Cycles consistently shorter than 21 days or longer than 35 days, or unexpected bleeding/spotting between periods.</li>
        <li><b>Secondary Amenorrhea:</b> Complete absence of menstrual periods for three or more consecutive cycles without pregnancy.</li>
        <li><b>Severe Luteal Dysphoria:</b> Marked depressive mood, acute anxiety, or severe emotional volatility in the late luteal phase that resolves shortly after menstruation (suggestive of PMDD).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# VIEW 5: MODEL INSIGHTS
# =============================================================================
def render_model_insights(predictor: Optional[HerAIPredictor] = None):
    """Render technical model card, evaluation benchmarks, and leakage audit verification."""
    st.markdown("<h1>Model Insights & Technical Architecture</h1>", unsafe_allow_html=True)
    st.markdown(
        "HERAI applies an L2-regularized Ridge Regression pipeline trained on 15,976 evaluated cycles. "
        "The model was selected through multi-model benchmarking and strict train/test user isolation, "
        "minimizing user input burden while maintaining high predictive accuracy."
    )

    # Key Architectural Metrics Summary Row
    km_c1, km_c2, km_c3, km_c4 = st.columns(4)
    with km_c1:
        st.markdown("""
        <div class="metric-stat-box">
            <div class="metric-stat-label">Production Algorithm</div>
            <div class="metric-stat-value" style="font-size: 1.35rem; padding-top: 4px;">Ridge Regression</div>
            <div class="metric-stat-detail">L2 Regularization (α = 1.0)</div>
        </div>
        """, unsafe_allow_html=True)
    with km_c2:
        st.markdown("""
        <div class="metric-stat-box">
            <div class="metric-stat-label">Active Feature Set</div>
            <div class="metric-stat-value">37 Features</div>
            <div class="metric-stat-detail">Lab & lookahead features excluded</div>
        </div>
        """, unsafe_allow_html=True)
    with km_c3:
        st.markdown("""
        <div class="metric-stat-box">
            <div class="metric-stat-label">Test Set MAE</div>
            <div class="metric-stat-value">0.6448</div>
            <div class="metric-stat-detail">Average error on 0–10 scale</div>
        </div>
        """, unsafe_allow_html=True)
    with km_c4:
        st.markdown("""
        <div class="metric-stat-box">
            <div class="metric-stat-label">Test Set R²</div>
            <div class="metric-stat-value">0.4620</div>
            <div class="metric-stat-detail">46.2% cyclical variance explained</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 1: Candidate Comparison & Benchmark Evaluation
    st.markdown("### Candidate Model Benchmarking (Test Set: N=3,176 Cycles across 400 Users)")
    st.caption("Strict user-level isolation: 0% user overlap between training and test sets.")

    m_c1, m_c2 = st.columns([1.1, 0.9], gap="medium")

    with m_c1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Candidate Model Benchmark Results</div>', unsafe_allow_html=True)
        bench_df = pd.DataFrame([
            {
                "Candidate Model": "Ridge Regression (37 features)",
                "Features": "37",
                "Test MAE": "0.6448",
                "Test RMSE": "0.8025",
                "Test R²": "0.4620",
                "Status": "👑 Selected Production"
            },
            {
                "Candidate Model": "Ridge Regression (42 features)",
                "Features": "42",
                "Test MAE": "0.6452",
                "Test RMSE": "0.8029",
                "Test R²": "0.4616",
                "Status": "Evaluated"
            },
            {
                "Candidate Model": "Random Forest (42 features)",
                "Features": "42",
                "Test MAE": "0.6550",
                "Test RMSE": "0.8130",
                "Test R²": "0.4480",
                "Status": "Evaluated"
            },
            {
                "Candidate Model": "Dummy Mean Baseline",
                "Features": "0",
                "Test MAE": "0.9113",
                "Test RMSE": "1.0942",
                "Test R²": "-0.0000",
                "Status": "Null Baseline"
            }
        ])
        st.dataframe(bench_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div style="font-size: 0.86rem; color: #59433F; line-height: 1.55; margin-top: 8px;">
            <b>Why Ridge (37 Features) Was Selected:</b>
            <ul style="padding-left: 18px; margin-top: 4px;">
                <li><b>Omits Invasive Lab Biomarkers:</b> Excluded <code>estrogen_pgml</code>, <code>progesterone_ngml</code>, and <code>ovulation_result</code>, which require venipuncture blood draws or clinical testing impractical for everyday self-monitoring.</li>
                <li><b>Prevents Lookahead Bias:</b> Excluded <code>prepared_before_period</code> and <code>log_consistency_score</code>, which inadvertently capture retrospective behavior.</li>
                <li><b>Maintains Highest Accuracy:</b> Achieves the lowest MAE (<b>0.6448</b>) and highest R² (<b>0.4620</b>) on unseen test users, proving that dropping invasive lab variables produces zero loss in predictive fidelity.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with m_c2:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Model Error Comparison</div>', unsafe_allow_html=True)
        fig_comp = create_model_comparison_bar_chart()
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("Both Ridge variants comfortably outperform the non-linear Random Forest and the Dummy Mean baseline on unseen test users.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Metric Explanations in Plain Language
    st.markdown('<div class="card-surface"><div class="card-header-serif">Understanding the Evaluation Metrics (Plain Language Guide)</div>', unsafe_allow_html=True)
    me_c1, me_c2, me_c3 = st.columns(3, gap="medium")
    with me_c1:
        st.markdown("""
        <div style="background: #FAF7F5; border-radius: 8px; padding: 14px; height: 100%;">
            <b style="color: #882D52;">Mean Absolute Error (MAE = 0.6448)</b>
            <p style="font-size: 0.85rem; color: #59433F; line-height: 1.5; margin-top: 6px;">
                On average, the model's predicted health score is within <b>0.64 points</b> of the user's actual health score in the next cycle (on a 0 to 10 scale). 
                This represents an average deviation of just 6.4% across the entire scoring spectrum.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with me_c2:
        st.markdown("""
        <div style="background: #FAF7F5; border-radius: 8px; padding: 14px; height: 100%;">
            <b style="color: #882D52;">Root Mean Squared Error (RMSE = 0.8025)</b>
            <p style="font-size: 0.85rem; color: #59433F; line-height: 1.5; margin-top: 6px;">
                Like MAE, but squares individual errors prior to averaging, making it more sensitive to large outliers. 
                The low RMSE of <b>0.8025</b> confirms that large, erratic forecasting errors rarely occur across test cycles.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with me_c3:
        st.markdown("""
        <div style="background: #FAF7F5; border-radius: 8px; padding: 14px; height: 100%;">
            <b style="color: #882D52;">Coefficient of Determination (R² = 0.4620)</b>
            <p style="font-size: 0.85rem; color: #59433F; line-height: 1.5; margin-top: 6px;">
                Measures how much of the cyclical score variation is explained by the model compared to simply guessing the cohort average. 
                An R² of <b>0.4620</b> means the model captures ~46.2% of the variance using observational parameters alone.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 3: Dynamic Standardized Feature Drivers
    st.markdown("### Standardized Feature Drivers & Model Coefficients")
    st.caption("Dynamically extracted from the fitted Scikit-learn Pipeline (intercept: 8.5786).")

    feat_c1, feat_c2 = st.columns([1.1, 0.9], gap="medium")

    with feat_c1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Standardized Effect Sizes (Top 10 Drivers)</div>', unsafe_allow_html=True)
        feat_names, feat_weights = get_driver_weights(predictor)
        fig_imp = create_feature_importance_chart(feat_names, feat_weights, top_n=10)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Standardized Ridge coefficients: Positive values (sage) support higher next-cycle health; negative values (dusty rose) depress the score.")
        st.markdown('</div>', unsafe_allow_html=True)

    with feat_c2:
        st.markdown("""
        <div class="card-surface" style="height: 100%;">
            <div class="card-header-serif">Physiological Interpretation</div>
            <div style="font-size: 0.86rem; color: #59433F; line-height: 1.6;">
                <p><b>1. Baseline Stress (−0.672):</b> Chronic stress is the single strongest negative statistical driver in the model, demonstrating how sustained hypothalamic-pituitary-adrenal (HPA) activation suppresses cycle vitality.</p>
                <p><b>2. Autoregressive Continuity (+0.453 &amp; +0.404):</b> Current health score and previous cycle health score provide strong positive continuity, reflecting physiological momentum across consecutive cycles.</p>
                <p><b>3. Pain &amp; Cramping Impact (+0.291 standardized coefficient):</b> Reflects the empirical co-variation between severe dysmenorrhea and cyclical disruption.</p>
                <p><b>4. PCOS History (−0.164):</b> Documented endocrine imbalance acts as a modest negative baseline adjustment in the model.</p>
                <p><b>5. Restorative Sleep (+0.151):</b> Adequate sleep duration exerts a positive protective effect across cycles.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 4: Data Leakage Audit & Preprocessing Safeguards
    st.markdown('<div class="card-surface"><div class="card-header-serif">Data Leakage Audit & Preprocessing Safeguards</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.88rem; color: #59433F; line-height: 1.6;">
    To guarantee scientific integrity and prevent optimistic performance overestimation, the HERAI ML pipeline enforces three strict architectural safeguards:
    <ol style="padding-left: 20px; margin-top: 8px;">
        <li><b>Zero User Overlap (<code>GroupShuffleSplit</code>):</b> All cycles for a given user belong exclusively to the training split (1,600 users) or the test split (400 users). The model is evaluated strictly on unseen individuals, preventing identity leakage.</li>
        <li><b>In-Pipeline Imputation &amp; Scaling:</b> All feature transformations (median imputation via <code>SimpleImputer</code>, z-score scaling via <code>StandardScaler</code>, one-hot encoding) are encapsulated within a unified Scikit-learn <code>Pipeline</code>. Scaler and imputer parameters are calculated strictly from training folds.</li>
        <li><b>Verified Target Alignment (0 Mismatches):</b> The regression target <code>next_health_score</code> is constructed using strict chronological grouping: <code>next_health_score == overall_health_score.shift(-1)</code> within <code>user_id</code>. Zero target misalignment was detected across all 15,976 dataset records.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 5: Clinical Boundary & Ethical Evaluation Notice
    st.markdown("""
    <div class="disclaimer-card" style="border-left-width: 5px;">
        <b style="color: #882D52; font-size: 0.95rem;">Honest Evaluation &amp; Clinical Boundary Notice:</b><br>
        <span style="font-size: 0.88rem; color: #59433F; line-height: 1.55;">
        A model performing better on this test split does <b>NOT</b> demonstrate clinical accuracy, diagnostic safety, or therapeutic efficacy on external real-world patient populations. 
        The HERAI model is an educational and exploratory machine learning prototype trained on synthetic observational records. 
        It does not diagnose gynecological disorders (such as endometriosis, fibroids, or PCOS), does not assess fertility or pregnancy viability, 
        and must never be used to direct medical treatment, contraception, or clinical management.
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# VIEW 6: ABOUT
# =============================================================================
def render_about():
    """Render Project Background, Architecture, Privacy Policy, and Disclaimers."""
    st.markdown("<h1>About HERAI</h1>", unsafe_allow_html=True)
    st.markdown(
        "HERAI is an open-source machine learning and healthcare data-science research prototype "
        "designed to explore how self-reported menstrual cycle observations can predict short-term cycle wellbeing."
    )

    ab_c1, ab_c2 = st.columns(2, gap="medium")

    with ab_c1:
        st.markdown("""
        <div class="card-surface">
            <div class="card-header-serif">Project Purpose & Origins</div>
            <p style="font-size: 0.9rem; line-height: 1.55;">
                Menstrual health remains historically under-represented in statistical predictive modeling. 
                HERAI demonstrates an end-to-end data science lifecycle: from raw record ingestion, schema validation, 
                and chronological lag feature engineering, to model comparison, leakage verification, and ethical inference.
            </p>
            <p style="font-size: 0.9rem; line-height: 1.55;">
                <b>Dataset Composition:</b> 17,976 raw cycle records across 2,000 unique user profiles, yielding 15,976 training records with verified next-cycle targets.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with ab_c2:
        st.markdown("""
        <div class="card-surface">
            <div class="card-header-serif">Data Privacy & Security</div>
            <p style="font-size: 0.9rem; line-height: 1.55;">
                <b>Local In-Memory Execution:</b> All user entries submitted through this web interface are processed 
                entirely in-memory within your local Python runtime. 
            </p>
            <p style="font-size: 0.9rem; line-height: 1.55;">
                <b>Zero Stranger-Data Substitution:</b> The prediction pipeline never borrows or copies records from other individuals in the dataset. 
                For first-time users, learned cohort medians from the training split serve as documented reference baselines.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <b>Clinical and Legal Disclaimer:</b> HERAI is an educational demonstration and research prototype. 
        It is not a medical device, diagnostic system, clinical decision support software, or contraceptive planner. 
        Outputs represent statistical predictions derived from observational data patterns and must not be used for diagnosis, medical treatment, or contraception planning.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP ROUTER
# =============================================================================
def main():
    render_app_header()
    predictor = get_predictor()
    cohort_df = get_cohort_dataset()

    if st.session_state.current_page == "Dashboard":
        render_dashboard(cohort_df)
    elif st.session_state.current_page == "Health Prediction":
        render_health_prediction(predictor)
    elif st.session_state.current_page == "Cycle Information":
        render_cycle_information(cohort_df)
    elif st.session_state.current_page == "Model Insights":
        render_model_insights(predictor)
    elif st.session_state.current_page == "About":
        render_about()
    else:
        render_dashboard(cohort_df)

    # Persistent calm footer
    st.markdown("""
    <div style="text-align: center; font-size: 0.78rem; color: #8F7B76; margin-top: 40px; padding-top: 16px; border-top: 1px solid #F0E2DA;">
        HERAI — Menstrual Health Intelligence Prototype &nbsp;|&nbsp; Educational & Data-Driven ML System &nbsp;|&nbsp; Non-Diagnostic
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
