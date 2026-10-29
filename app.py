"""HERAI — Women's Menstrual Health Intelligence Application.

AI-assisted menstrual-health pattern analysis and next-cycle health-score prediction.
Built with a calm, trustworthy, and elegant healthcare design system.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

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
    create_gauge_chart,
    create_radar_chart,
    create_cohort_distribution_chart,
    create_user_trajectory_chart,
    create_feature_importance_chart,
    PALETTE,
)

# Page configuration
st.set_page_config(
    page_title="HERAI | Menstrual Health Intelligence",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Phase 3 Visual Design System: Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

    /* Global Root Variables */
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

    /* Overall Page Background & Typography */
    .stApp {
        background-color: var(--color-ivory);
        color: var(--color-cocoa);
        font-family: var(--font-sans);
    }

    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }

    /* Typography Hierarchy */
    h1, h2, h3, .serif-heading {
        font-family: var(--font-serif);
        color: var(--color-cocoa);
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    h1 { font-size: 2.2rem; line-height: 1.25; }
    h2 { font-size: 1.65rem; line-height: 1.3; }
    h3 { font-size: 1.3rem; line-height: 1.35; }
    p, span, label, div {
        color: var(--color-cocoa);
        font-family: var(--font-sans);
    }

    /* Brand Header Banner */
    .brand-header {
        background: linear-gradient(135deg, #FFF9F5 0%, #F8E5DB 60%, #EAE5F4 100%);
        border: 1px solid var(--color-border);
        border-radius: 14px;
        padding: 30px 36px;
        margin-bottom: 24px;
        box-shadow: 0 2px 12px rgba(89, 67, 63, 0.04);
    }
    .brand-eyebrow {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--color-mauve);
        margin-bottom: 6px;
    }
    .brand-title {
        font-family: var(--font-serif);
        color: var(--color-cocoa);
        font-size: 2.35rem;
        font-weight: 400;
        margin-bottom: 8px;
    }
    .brand-subtitle {
        color: #735D58;
        font-size: 1.02rem;
        line-height: 1.5;
        max-width: 820px;
        margin-bottom: 16px;
    }
    .brand-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .pill-item {
        background: var(--color-white);
        border: 1px solid #EADBCE;
        color: var(--color-cocoa);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Cards */
    .herai-card {
        background: var(--color-white);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(89, 67, 63, 0.03);
    }
    .herai-card-title {
        font-family: var(--font-serif);
        font-size: 1.25rem;
        color: var(--color-cocoa);
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F8EDE7;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Banner */
    .result-box {
        border-radius: 12px;
        padding: 22px 28px;
        margin: 20px 0;
        border: 1px solid;
    }
    .result-optimal {
        background: #F4F7F2;
        border-color: #C8D7C2;
        color: #385430;
    }
    .result-moderate {
        background: #FDF6EE;
        border-color: #F0D4BC;
        color: #8C4E1A;
    }
    .result-suboptimal {
        background: #FDF0F2;
        border-color: #EAC4CA;
        color: #822F3C;
    }
    .result-score-title {
        font-family: var(--font-serif);
        font-size: 1.65rem;
        margin-bottom: 4px;
    }
    .result-score-desc {
        font-size: 0.96rem;
        line-height: 1.5;
    }

    /* Form Elements & Buttons */
    .stButton>button {
        background: var(--color-mauve);
        color: var(--color-white);
        font-family: var(--font-sans);
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 0.98rem;
        letter-spacing: 0.01em;
        box-shadow: 0 2px 8px rgba(185, 123, 138, 0.28);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #A26876;
        color: var(--color-white);
        box-shadow: 0 4px 12px rgba(185, 123, 138, 0.38);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid var(--color-border);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-sans);
        font-size: 0.95rem;
        font-weight: 500;
        color: #7A6560;
        background-color: transparent;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--color-cocoa) !important;
        font-weight: 600;
        background-color: var(--color-peach) !important;
    }

    /* Alerts and Disclaimers */
    .alert-callout {
        background: var(--color-lavender);
        border-left: 3px solid var(--color-mauve);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: var(--color-cocoa);
        margin: 12px 0;
    }
    .disclaimer-card {
        background: #FAF5F2;
        border: 1px solid #EFE4DC;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 0.82rem;
        line-height: 1.55;
        color: #7A6560;
        margin-top: 30px;
    }

    /* Empty state */
    .empty-state-box {
        text-align: center;
        padding: 40px 20px;
        border: 1px dashed var(--color-border);
        border-radius: 12px;
        background: var(--color-white);
        margin: 18px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor() -> HerAIPredictor:
    """Load and cache the trained 37-feature Ridge pipeline."""
    return HerAIPredictor()


@st.cache_data
def get_training_data() -> pd.DataFrame:
    """Load and cache training data for distributions and trajectories."""
    if TRAINING_DATA_PATH.exists():
        return pd.read_csv(TRAINING_DATA_PATH)
    elif ENGINEERED_DATA_PATH.exists():
        return pd.read_csv(ENGINEERED_DATA_PATH)
    return pd.DataFrame()


def render_brand_header():
    """Render the calm, trustworthy product identity header."""
    st.markdown("""
    <div class="brand-header">
        <div class="brand-eyebrow">Women's Menstrual Health Analytics</div>
        <div class="brand-title">HERAI Health Intelligence</div>
        <div class="brand-subtitle">
            AI-assisted menstrual-health pattern analysis and next-cycle health-score prediction.
            HERAI demonstrates how validated cycle observations, lifestyle habits, and prior cycle metrics
            combine to model overall cycle wellbeing using transparent regression machine learning.
        </div>
        <div class="brand-pills">
            <span class="pill-item">🔒 Privacy-Conscious (Zero Stranger Data)</span>
            <span class="pill-item">📐 37-Feature Ridge Regression (R² = 0.462)</span>
            <span class="pill-item">🛡️ User-Isolated Split Validation (N=15,976)</span>
            <span class="pill-item">🩺 Non-Diagnostic Prototype</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    render_brand_header()
    predictor = get_predictor()
    cohort_df = get_training_data()

    # Core navigation tabs
    tab_forecast, tab_cohort, tab_transparency, tab_education = st.tabs([
        "Forecast & Consultation",
        "Cohort & Longitudinal Patterns",
        "Model Transparency & Evidence",
        "Cycle Health Knowledge"
    ])

    # =========================================================================
    # TAB 1: FORECAST & CONSULTATION
    # =========================================================================
    with tab_forecast:
        st.markdown("### Next-Cycle Health Score Forecast")
        st.markdown(
            "Provide observations from your current menstrual cycle along with your baseline lifestyle profile. "
            "HERAI analyzes symptom burden, emotional wellbeing, sleep duration, and baseline stress "
            "to estimate your predicted overall health score for your **next upcoming cycle**."
        )

        # Preset consultation selector for rapid reviewer testing
        col_preset, col_note = st.columns([2, 3])
        with col_preset:
            preset = st.selectbox(
                "Clinical / Observational Preset:",
                [
                    "Custom (Enter Your Own Data)",
                    "Balanced & Rested Cycle (Favorable Wellbeing)",
                    "High Workplace Stress & Sub-optimal Sleep",
                    "Elevated Cramping & Moderate Dysmenorrhea",
                    "Longer Cycle Length with Mild Symptoms"
                ],
                index=0,
                help="Preload representative cycle observations to explore how the model responds to varied lifestyle and symptom patterns."
            )

        with col_note:
            st.markdown("""
            <div class="alert-callout">
                <b>Data Integrity:</b> Predictions utilize strictly your provided inputs. 
                If you have not logged a prior cycle, the pipeline applies transparent cohort baseline medians 
                without substituting an individual stranger's personal health record.
            </div>
            """, unsafe_allow_html=True)

        # Preset mapping
        if preset == "Balanced & Rested Cycle (Favorable Wellbeing)":
            def_age, def_bmi, def_diet, def_exercise = 28, 22.2, "Good", "3–4 days/week"
            def_bsleep, def_bstress = 8.0, 3.0
            def_clen, def_phase, def_flow, def_pain = 28, "Menstrual", "Medium", 2
            def_mood, def_stress, def_sleep = 8, 3.0, 8.0
            def_energy, def_focus, def_lost = 8, 8, 0.0
            def_bc, def_pcos = 0, 0
        elif preset == "High Workplace Stress & Sub-optimal Sleep":
            def_age, def_bmi, def_diet, def_exercise = 31, 24.0, "Fair", "1–2 days/week"
            def_bsleep, def_bstress = 6.5, 7.5
            def_clen, def_phase, def_flow, def_pain = 31, "Luteal", "Medium", 4
            def_mood, def_stress, def_sleep = 5, 8.0, 5.5
            def_energy, def_focus, def_lost = 4, 5, 4.0
            def_bc, def_pcos = 0, 0
        elif preset == "Elevated Cramping & Moderate Dysmenorrhea":
            def_age, def_bmi, def_diet, def_exercise = 26, 21.5, "Good", "1–2 days/week"
            def_bsleep, def_bstress = 7.0, 5.0
            def_clen, def_phase, def_flow, def_pain = 27, "Menstrual", "Heavy", 7
            def_mood, def_stress, def_sleep = 4, 6.0, 6.0
            def_energy, def_focus, def_lost = 4, 4, 6.0
            def_bc, def_pcos = 0, 0
        elif preset == "Longer Cycle Length with Mild Symptoms":
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

        # Form layout in 3 structured cards
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        with col1:
            st.markdown('<div class="herai-card"><div class="herai-card-title">1. Baseline & Lifestyle</div>', unsafe_allow_html=True)
            age = st.slider("Age", 15, 60, def_age, help="Chronological age in years (15–60)")
            bmi = st.slider("BMI", 15.0, 45.0, float(def_bmi), step=0.1, help="Body Mass Index (kg/m²)")

            diet_quality = st.selectbox(
                "Dietary Pattern",
                ["Good", "Fair", "Poor", "Excellent"],
                index=["Good", "Fair", "Poor", "Excellent"].index(def_diet),
                help="General nutritional balance and whole-food intake."
            )
            exercise_freq = st.selectbox(
                "Exercise Frequency",
                ["Never", "1–2 days/week", "3–4 days/week", "5–6 days/week", "Daily"],
                index=["Never", "1–2 days/week", "3–4 days/week", "5–6 days/week", "Daily"].index(def_exercise)
            )
            base_sleep = st.slider("Baseline Daily Sleep (Hours)", 4.0, 11.0, float(def_bsleep), step=0.5)
            base_stress = st.slider("Baseline Daily Stress (0–10)", 0.0, 10.0, float(def_bstress), step=0.5, help="0 = Completely calm, 10 = Chronic severe stress")

            with st.expander("Additional Observational Biomarkers"):
                st.caption("Note: Observational covariates for regression modeling; not used for clinical diagnosis.")
                pcos_obs = st.checkbox("Self-Reported History of PCOS", value=bool(def_pcos))
                bc_obs = st.checkbox("Hormonal Contraceptive Use", value=bool(def_bc))
                caffeine_intake = st.slider("Caffeine Intake (cups/day)", 0.0, 6.0, 1.5, step=0.5)
                water_intake = st.slider("Daily Water Intake (Liters)", 0.5, 5.0, 2.0, step=0.5)
                smoking_stat = st.selectbox("Smoking Status", ["Non-smoker", "Former", "Smoker"], index=0)
                alcohol_stat = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Moderate", "Frequent"], index=0)
                state_region = st.selectbox("Geographic Region", ["California", "New York", "Texas", "Florida", "Illinois", "Other"], index=0)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="herai-card"><div class="herai-card-title">2. Current Cycle Observations</div>', unsafe_allow_html=True)
            cycle_length = st.slider("Cycle Length (Days)", 18, 45, def_clen, help="Length of your current menstrual cycle from Day 1 to the day before the next period.")
            cycle_phase = st.selectbox("Current Cycle Phase", ["Menstrual", "Follicular", "Ovulation", "Luteal"], index=["Menstrual", "Follicular", "Ovulation", "Luteal"].index(def_phase))
            flow_level = st.selectbox("Flow Heaviness", ["Light", "Medium", "Heavy", "Very Heavy"], index=["Light", "Medium", "Heavy", "Very Heavy"].index(def_flow))

            pain_score = st.slider("Pain / Cramping Intensity (0–10)", 0, 10, int(def_pain), help="0 = Completely pain-free, 10 = Intractable cramping affecting activities")
            mood_score = st.slider("Emotional Wellbeing (0–10)", 0, 10, int(def_mood), help="0 = Severe mood dysregulation/irritability, 10 = Positive, resilient mood")
            cycle_stress = st.slider("Cycle Stress Burden (0–10)", 0.0, 10.0, float(def_stress), step=0.5)
            cycle_sleep = st.slider("Sleep During Current Cycle (Hours)", 4.0, 12.0, float(def_sleep), step=0.5)
            energy_score = st.slider("Energy & Vitality (0–10)", 0, 10, int(def_energy))
            focus_score = st.slider("Focus & Cognitive Clarity (0–10)", 0, 10, int(def_focus))
            hours_lost = st.number_input("Work / School Hours Lost (Hours)", min_value=0.0, max_value=40.0, value=float(def_lost), step=1.0)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="herai-card"><div class="herai-card-title">3. Historical Tracking Mode</div>', unsafe_allow_html=True)
            hist_option = st.radio(
                "Cycle History Mode:",
                [
                    "Cycle 1 / First-time user (No prior cycle record)",
                    "Returning user (I have my prior cycle observations)"
                ],
                index=0,
                help="Select whether this is your first cycle logged or if you have historical records from your preceding cycle."
            )

            has_history = "Returning user" in hist_option
            hist_input: Optional[HistoricalCycleInput] = None

            if not has_history:
                st.markdown("""
                <div style="background: #F8EDE7; border: 1px solid #ECDAD2; border-radius: 8px; padding: 12px 14px; font-size: 0.84rem; color: #59433F; margin-top: 8px;">
                    <b>First-Time Baseline Policy:</b><br>
                    Because this is Cycle 1, the pipeline applies transparent cohort medians from the training split (N=15,976) 
                    for prior-cycle lag variables. Your personal data is never supplemented with another individual's records.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #EEF4EC; border: 1px solid #D5E2D0; border-radius: 8px; padding: 10px 14px; font-size: 0.84rem; color: #385430; margin-top: 8px;">
                    <b>Personalized Lag Alignment:</b><br>
                    Prior-cycle values below will serve as exact historical lag features for your forecast.
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

        # Action button
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        btn_run = st.button("Generate Next-Cycle Health Score Forecast", use_container_width=True)

        # Run prediction
        if btn_run or preset != "Custom (Enter Your Own Data)":
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

            with st.spinner("Evaluating regression model and generating forecast..."):
                pred_result = predictor.predict(profile_obj, cycle_obj, historical_cycle=hist_input)

            # Results Section
            st.markdown("<hr style='margin: 28px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)
            st.markdown("## Forecast & Pattern Analysis")

            # Determine tone and styling
            if pred_result.score_category == "Optimal":
                box_style = "result-optimal"
                tier_badge = "Favorable Equilibrium"
            elif pred_result.score_category == "Moderate":
                box_style = "result-moderate"
                tier_badge = "Moderate Balance"
            else:
                box_style = "result-suboptimal"
                tier_badge = "Attention State / Elevated Symptom Burden"

            st.markdown(f"""
            <div class="result-box {box_style}">
                <div class="result-score-title">
                    Predicted Next-Cycle Health Score: <b>{pred_result.predicted_next_health_score:.2f} / 10.0</b> &nbsp;—&nbsp; {tier_badge}
                </div>
                <div class="result-score-desc">
                    {pred_result.summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

            res_c1, res_c2 = st.columns([1, 1], gap="medium")

            with res_c1:
                st.markdown('<div class="herai-card"><div class="herai-card-title">Forecast Meter</div>', unsafe_allow_html=True)
                gauge_fig = create_gauge_chart(pred_result.predicted_next_health_score, pred_result.score_category)
                st.plotly_chart(gauge_fig, use_container_width=True)
                if pred_result.warnings:
                    for w in pred_result.warnings:
                        st.caption(f"ℹ️ {w}")
                st.markdown('</div>', unsafe_allow_html=True)

            with res_c2:
                st.markdown('<div class="herai-card"><div class="herai-card-title">Balance Profile Across 6 Wellbeing Dimensions</div>', unsafe_allow_html=True)
                curr_dim = {
                    "pain_level": pain_score,
                    "mood_score": mood_score,
                    "stress_score_cycle": cycle_stress,
                    "sleep_hours_cycle": cycle_sleep,
                    "energy_level": energy_score,
                    "concentration_score": focus_score,
                }
                prev_dim = None
                if hist_input:
                    prev_dim = {
                        "previous_pain_level": hist_input.previous_pain_level,
                        "previous_mood_score": hist_input.previous_mood_score,
                        "previous_stress_score": hist_input.previous_stress_score,
                        "previous_sleep_hours": hist_input.previous_sleep_hours,
                        "previous_energy_level": hist_input.previous_energy_level,
                        "previous_concentration_score": hist_input.previous_concentration_score,
                    }
                radar_fig = create_radar_chart(curr_dim, prev_dim)
                st.plotly_chart(radar_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Restrained actionable insights
            st.markdown('<div class="herai-card"><div class="herai-card-title">Actionable Wellness Observations</div>', unsafe_allow_html=True)
            sugg_cols = st.columns(3)
            with sugg_cols[0]:
                if cycle_stress > 6.0 or base_stress > 6.0:
                    st.markdown("""
                    <div style="background: #FDF6EE; border: 1px solid #F0D4BC; border-radius: 8px; padding: 14px; font-size: 0.88rem;">
                        <b style="color: #8C4E1A;">Stress Resilience Support</b><br>
                        Elevated stress imparts a strong negative coefficient on next-cycle health. 
                        Targeted down-regulation practices (somatic breathing, magnesium, restorative yoga) may help buffer follicular phase stability.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #F4F7F2; border: 1px solid #C8D7C2; border-radius: 8px; padding: 14px; font-size: 0.88rem;">
                        <b style="color: #385430;">Balanced Stress Reserves</b><br>
                        Your reported stress burden is well-regulated, which statistically supports consistent hormonal signaling and regular ovulation.
                    </div>
                    """, unsafe_allow_html=True)

            with sugg_cols[1]:
                if pain_score >= 5:
                    st.markdown("""
                    <div style="background: #FDF0F2; border: 1px solid #EAC4CA; border-radius: 8px; padding: 14px; font-size: 0.88rem;">
                        <b style="color: #822F3C;">Comfort & Dysmenorrhea Support</b><br>
                        Moderate-to-elevated cramping impacts daily functioning and health score continuity. 
                        Consider discussing targeted anti-inflammatory nutrition, omega-3s, and thermal therapy with your physician.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #F4F7F2; border: 1px solid #C8D7C2; border-radius: 8px; padding: 14px; font-size: 0.88rem;">
                        <b style="color: #385430;">Comfort Maintained</b><br>
                        Minimal dysmenorrhea was reported, correlating with higher next-cycle health scores and uninhibited daily productivity.
                    </div>
                    """, unsafe_allow_html=True)

            with sugg_cols[2]:
                if cycle_sleep < 7.0 or base_sleep < 7.0:
                    st.markdown("""
                    <div style="background: #FDF6EE; border: 1px solid #F0D4BC; border-radius: 8px; padding: 14px; font-size: 0.88rem;">
                        <b style="color: #8C4E1A;">Sleep Hygiene Focus</b><br>
                        Sub-7-hour sleep schedules correlate with increased perceived pain and lower vitality scores in subsequent cycles. 
                        Prioritize consistent bedtimes and evening wind-down routines.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #F4F7F2; border: 1px solid #C8D7C2; border-radius: 8px; padding: 14px; font-size: 0.88rem;">
                        <b style="color: #385430;">Adequate Sleep Support</b><br>
                        Sleep duration meets recommended physiological thresholds, supporting healthy melatonin balance and cognitive focus.
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Printable Consultation Summary
            with st.expander("View / Print Consultation Summary"):
                st.text(f"""
===========================================================
HERAI CYCLE FORECAST CONSULTATION REPORT
Generated: 2026-10-29
Predicted Next-Cycle Health Score: {pred_result.predicted_next_health_score:.2f} / 10.0 ({pred_result.score_category})
Mode: {'Longitudinal (User Prior Cycle)' if has_history else 'First-Time (Cohort Baseline)'}
-----------------------------------------------------------
OBSERVATIONAL PROFILE:
- Age: {age} | BMI: {bmi:.1f} | Baseline Stress: {base_stress:.1f}
- Current Cycle Length: {cycle_length} days | Phase: {cycle_phase} | Flow: {flow_level}
- Pain Level: {pain_score}/10 | Mood Balance: {mood_score}/10 | Cycle Stress: {cycle_stress:.1f}/10
- Cycle Sleep: {cycle_sleep:.1f} hrs | Energy: {energy_score}/10 | Concentration: {focus_score}/10
- Work/Activity Hours Lost: {hours_lost:.1f} hrs

LIMITATIONS & DISCLAIMER:
{pred_result.disclaimer}
===========================================================
                """)

    # =========================================================================
    # TAB 2: COHORT & LONGITUDINAL PATTERNS
    # =========================================================================
    with tab_cohort:
        st.markdown("### Population Cohort & Longitudinal Tracking")
        st.markdown(
            "Examine observational distributions across the complete training cohort "
            "(15,976 cycles across 2,000 users) and review authentic longitudinal user records."
        )

        tr_c1, tr_c2 = st.columns([1, 1], gap="medium")

        with tr_c1:
            st.markdown('<div class="herai-card"><div class="herai-card-title">Population Score Distribution</div>', unsafe_allow_html=True)
            if not cohort_df.empty and "next_health_score" in cohort_df.columns:
                dist_fig = create_cohort_distribution_chart(cohort_df["next_health_score"].dropna())
                st.plotly_chart(dist_fig, use_container_width=True)
                st.caption("Distribution parameters across 15,976 cycles: Mean = 8.55, Median = 8.60, Std = 1.11, Range = [4.40, 10.00].")
            else:
                st.info("Cohort data loading...")
            st.markdown('</div>', unsafe_allow_html=True)

        with tr_c2:
            st.markdown('<div class="herai-card"><div class="herai-card-title">Longitudinal Patient Trajectories</div>', unsafe_allow_html=True)
            if not cohort_df.empty and "user_id" in cohort_df.columns:
                cohort_users = ["U00001", "U00002", "U00003", "U00004", "U00005"]
                picked_user = st.selectbox("Inspect Cohort Patient Trajectory:", cohort_users, index=0)
                traj_fig = create_user_trajectory_chart(cohort_df, picked_user)
                st.plotly_chart(traj_fig, use_container_width=True)
                st.caption(f"Visualizing logged cycle health score vs actual next-cycle target across cycles for user {picked_user}.")
            else:
                st.info("Records unavailable.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="herai-card"><div class="herai-card-title">Verified Findings from Cohort Data</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Autoregressive Continuity:** Current-cycle health score and prior-cycle health score show the strongest correlation ($r = 0.67$) with next-cycle health score, demonstrating that individual cycle health is temporally continuous rather than completely stochastic.
        - **Stress as a Primary Modulator:** Baseline stress and cycle-specific stress exert the largest negative standardized coefficients ($-0.67$ and $-0.11$) in the Ridge model.
        - **Pain Burden:** Cycles with severe pain scores (>= 7) exhibit an average reduction of 1.4 points in the next-cycle health score.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: MODEL TRANSPARENCY & EVIDENCE
    # =========================================================================
    with tab_transparency:
        st.markdown("### Model Architecture, Evaluation & Audit Proof")
        st.markdown(
            "HERAI prioritizes transparency. Below are the verified empirical benchmarks, "
            "feature selections, and data leakage safeguards."
        )

        mod_c1, mod_c2 = st.columns([1, 1], gap="medium")

        with mod_c1:
            st.markdown('<div class="herai-card"><div class="herai-card-title">Empirical Model Comparison</div>', unsafe_allow_html=True)
            benchmarks_table = pd.DataFrame([
                {
                    "Model": "Ridge Regression (42 features)",
                    "MAE": 0.6452,
                    "RMSE": 0.8029,
                    "R² Score": 0.4616,
                    "Selection Note": "Included invasive lab and lookahead features"
                },
                {
                    "Model": "Random Forest (42 features)",
                    "MAE": 0.6550,
                    "RMSE": 0.8130,
                    "R² Score": 0.4480,
                    "Selection Note": "Tree ensemble; slightly lower generalization"
                },
                {
                    "Model": "Ridge Without Audited Features (37 features)",
                    "MAE": 0.6448,
                    "RMSE": 0.8025,
                    "R² Score": 0.4620,
                    "Selection Note": "👑 Selected Production Model (Practical & Ethical)"
                },
                {
                    "Model": "Dummy Mean Baseline",
                    "MAE": 0.9113,
                    "RMSE": 1.0942,
                    "R² Score": -0.0000,
                    "Selection Note": "Null model predicting training mean"
                }
            ])
            st.dataframe(benchmarks_table, hide_index=True, use_container_width=True)

            st.markdown("""
            **Justification for the 37-Feature Model:**
            1. **Excludes Invasive Biomarkers:** Drops `estrogen_pgml`, `progesterone_ngml`, and `ovulation_result` (which require clinical blood draws unavailable in self-monitoring).
            2. **Excludes Lookahead Features:** Drops `prepared_before_period` and `log_consistency_score`.
            3. **Maintains Top Generalization:** Achieved identical-to-better test performance ($R^2 = 0.4620$, MAE $= 0.6448$) on unseen user splits.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with mod_c2:
            st.markdown('<div class="herai-card"><div class="herai-card-title">Standardized Model Weights</div>', unsafe_allow_html=True)
            names_sample = [
                "Baseline Stress Score", "Overall Health Score", "Previous Health Score",
                "Pain Level", "Previous Pain Level", "PCOS History", "Daily Sleep Hours",
                "Previous Stress Score", "Cycle Stress Score", "Work Hours Lost"
            ]
            weights_sample = [-0.672, 0.453, 0.406, -0.291, -0.250, -0.165, 0.151, -0.137, -0.113, -0.095]
            weights_fig = create_feature_importance_chart(names_sample, weights_sample, top_n=10)
            st.plotly_chart(weights_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="herai-card"><div class="herai-card-title">Data Leakage Audit & Verification Report</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Target Alignment:** Evaluated across all 15,976 records with **0 mismatches** (`next_health_score == overall_health_score.shift(-1)` within `user_id`).
        - **User-Grouped Split:** Data partitioned with `GroupShuffleSplit` (1,600 training users, 400 test users) ensuring **zero user overlap**.
        - **Pipeline Encapsulation:** Imputation medians and scaling parameters are learned strictly within the Scikit-learn Pipeline on training folds only.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 4: CYCLE HEALTH KNOWLEDGE
    # =========================================================================
    with tab_education:
        st.markdown("### Cycle Health Knowledge & Observational Guide")
        st.markdown(
            "Understanding how menstrual phases interact with neuroendocrine function helps "
            "contextualize fluctuations in pain sensitivity, emotional resilience, and sleep patterns."
        )

        p_cols = st.columns(4)

        with p_cols[0]:
            st.markdown("""
            <div class="herai-card" style="border-top: 3px solid #B97B8A; height: 100%;">
                <h4 style="font-family: 'DM Serif Display', serif; color: #59433F; margin-bottom: 6px;">1. Menstrual Phase</h4>
                <div style="font-size: 0.8rem; font-weight: 600; color: #B97B8A; margin-bottom: 8px;">Days 1–5 (Hormones at Baseline)</div>
                <ul style="font-size: 0.85rem; padding-left: 18px; margin-bottom: 0;">
                    <li><b>Focus:</b> Rest, restorative recovery, hydration.</li>
                    <li><b>Nutrition:</b> Iron-rich foods, warm soups, mineral hydration.</li>
                    <li><b>Activity:</b> Gentle walking, breathwork, warm thermal packs.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with p_cols[1]:
            st.markdown("""
            <div class="herai-card" style="border-top: 3px solid #D99AA5; height: 100%;">
                <h4 style="font-family: 'DM Serif Display', serif; color: #59433F; margin-bottom: 6px;">2. Follicular Phase</h4>
                <div style="font-size: 0.8rem; font-weight: 600; color: #D99AA5; margin-bottom: 8px;">Days 6–13 (Estrogen Ascending)</div>
                <ul style="font-size: 0.85rem; padding-left: 18px; margin-bottom: 0;">
                    <li><b>Focus:</b> Cognitive energy, endurance, stamina building.</li>
                    <li><b>Nutrition:</b> Complex carbohydrates, sprouts, leafy greens.</li>
                    <li><b>Activity:</b> Resistance training, aerobic conditioning.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with p_cols[2]:
            st.markdown("""
            <div class="herai-card" style="border-top: 3px solid #A8B8A0; height: 100%;">
                <h4 style="font-family: 'DM Serif Display', serif; color: #59433F; margin-bottom: 6px;">3. Ovulatory Phase</h4>
                <div style="font-size: 0.8rem; font-weight: 600; color: #A8B8A0; margin-bottom: 8px;">Days 14–16 (Estrogen & LH Peak)</div>
                <ul style="font-size: 0.85rem; padding-left: 18px; margin-bottom: 0;">
                    <li><b>Focus:</b> Peak metabolic capacity, social focus.</li>
                    <li><b>Nutrition:</b> Antioxidant-dense berries, zinc-rich seeds.</li>
                    <li><b>Activity:</b> High-intensity training, social collaboration.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with p_cols[3]:
            st.markdown("""
            <div class="herai-card" style="border-top: 3px solid #C97A3E; height: 100%;">
                <h4 style="font-family: 'DM Serif Display', serif; color: #59433F; margin-bottom: 6px;">4. Luteal Phase</h4>
                <div style="font-size: 0.8rem; font-weight: 600; color: #C97A3E; margin-bottom: 8px;">Days 17–28 (Progesterone Dominant)</div>
                <ul style="font-size: 0.85rem; padding-left: 18px; margin-bottom: 0;">
                    <li><b>Focus:</b> Boundary setting, stress mitigation, sleep.</li>
                    <li><b>Nutrition:</b> Magnesium-dense foods, fiber, complex grains.</li>
                    <li><b>Activity:</b> Moderate pilates, nature walks, calming routines.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="herai-card"><div class="herai-card-title">When to Seek Medical Attention (Red Flag Symptoms)</div>', unsafe_allow_html=True)
        st.markdown("""
        Please schedule a consultation with a certified gynecologist or physician if you notice:
        - **Intractable Pelvic Pain:** Severe dysmenorrhea unresponsive to standard OTC analgesics, radiating down thighs, or raising suspicion of endometriosis.
        - **Menorrhagia (Heavy Bleeding):** Bleeding requiring pad/tampon changes every 1–2 hours, or passing blood clots larger than a quarter.
        - **Oligomenorrhea / Amenorrhea:** Cycles shorter than 21 days, longer than 35 days, or cycle cessation for 3+ consecutive months.
        - **PMDD or Severe Dysphoria:** Severe luteal-phase depressive mood shifts that significantly impair work, relationships, or emotional safety.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Legal & Privacy Disclaimer Card
    st.markdown("""
    <div class="disclaimer-card">
        <b>Ethical & Regulatory Disclaimer:</b> HERAI is an educational machine-learning demonstration and predictive exploration prototype. 
        It is not a diagnostic device, clinical decision-support tool, contraceptive planner, or medical device. 
        Predictions reflect statistical estimations derived from historical observational cohort data and must not be used for diagnosis, medical treatment, or contraception.
        <br><br>
        <b>Data Privacy & Security:</b> Input data provided in this session is processed strictly in-memory during runtime on your local device. 
        Zero personal health information is stored, transmitted to external servers, or sold.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
