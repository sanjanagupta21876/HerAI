"""HERAI — Professional Women's Menstrual Health Prediction Application.

An interactive clinical and data-driven web interface powered by the 37-feature
Ridge Regression pipeline, providing personalized next-cycle health score forecasts,
longitudinal analytics, and holistic wellness insights.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

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
    page_title="HERAI | Menstrual Health Prediction System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end healthcare UI/UX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #FFF5F8 0%, #F8EBF1 50%, #F5F0F8 100%);
        border: 1px solid #F0D5E2;
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(136, 45, 82, 0.04);
    }
    .hero-title {
        color: #882D52;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #5A4E54;
        font-size: 1.05rem;
        margin-bottom: 14px;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .badge-pill {
        background: #FFFFFF;
        color: #882D52;
        border: 1px solid #E8CDDC;
        padding: 5px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Card Containers */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #EFE4EA;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 2px 12px rgba(136, 45, 82, 0.03);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 6px 18px rgba(136, 45, 82, 0.06);
    }

    .card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #882D52;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #F6EDF1;
        padding-bottom: 10px;
    }

    /* Result Banner */
    .result-banner {
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        border: 2px solid;
    }
    .result-optimal {
        background: #F1F8E9;
        border-color: #81C784;
        color: #1B5E20;
    }
    .result-moderate {
        background: #FFF8E1;
        border-color: #FFB74D;
        color: #E65100;
    }
    .result-suboptimal {
        background: #FFEBEE;
        border-color: #E57373;
        color: #B71C1C;
    }

    /* Custom Form styling */
    .stButton>button {
        background: linear-gradient(135deg, #882D52 0%, #A23B64 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 1.05rem;
        box-shadow: 0 4px 14px rgba(136, 45, 82, 0.25);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #702242 0%, #882D52 100%);
        box-shadow: 0 6px 18px rgba(136, 45, 82, 0.35);
        color: white;
    }

    .disclaimer-box {
        background: #F9F7F8;
        border-left: 4px solid #882D52;
        padding: 14px 18px;
        border-radius: 0 12px 12px 0;
        font-size: 0.85rem;
        color: #6C5B64;
        margin-top: 25px;
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


def render_header():
    """Render the top visual banner."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">
            <span>🌸</span> HERAI — Women's Menstrual Health Intelligence
        </div>
        <div class="hero-subtitle">
            Next-cycle health score forecasting driven by clinical cycle observations, lifestyle biomarkers, and machine learning.
        </div>
        <div class="hero-badges">
            <span class="badge-pill">🔒 Zero Stranger-Data Substitution</span>
            <span class="badge-pill">📊 37-Feature Ridge Pipeline (R² = 0.462)</span>
            <span class="badge-pill">🛡️ User-Grouped Isolated Split (N=15,976)</span>
            <span class="badge-pill">🩺 Clinical Educational Prototype</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    render_header()
    predictor = get_predictor()
    data_df = get_training_data()

    # Main navigation tabs
    tab_predict, tab_analytics, tab_model, tab_guide = st.tabs([
        "🔮 Health Score Predictor",
        "📈 Cohort & Longitudinal Trends",
        "🔬 Model Architecture & Audit",
        "🌿 Holistic Care & Cycle Guide"
    ])

    # ==========================================
    # TAB 1: HEALTH SCORE PREDICTOR
    # ==========================================
    with tab_predict:
        st.markdown("### 📋 Interactive Cycle Consultation & Prediction")
        st.markdown(
            "Enter your personal profile and observations from your current cycle. "
            "HERAI evaluates physical pain, mood balance, sleep hygiene, and baseline stress "
            "to estimate your predicted overall health score for your **next cycle**."
        )

        # Quick preset scenarios
        col_preset, col_info = st.columns([2, 3])
        with col_preset:
            preset = st.selectbox(
                "⚡ Quick-Load Clinical Preset Scenario:",
                [
                    "Custom (Enter Your Own)",
                    "🌸 Balanced & Regular Cycle (Optimal)",
                    "⚡ High Stress & Moderate Fatigue",
                    "🩸 Heavy Flow & Dysmenorrhea (Elevated Pain)",
                    "🌿 PCOS / Irregular Cycle Pattern"
                ],
                index=0
            )

        # Set preset defaults
        if preset == "🌸 Balanced & Regular Cycle (Optimal)":
            p_age, p_bmi, p_diet, p_exercise = 28, 22.0, "Good", "3–4 days/week"
            p_base_sleep, p_base_stress = 8.0, 3.0
            p_cycle_len, p_phase, p_flow, p_pain = 28, "Menstrual", "Medium", 2
            p_mood, p_stress_c, p_sleep_c = 8, 3.0, 8.0
            p_energy, p_focus, p_work_lost = 8, 8, 0.0
            p_pcos, p_bc = 0, 0
        elif preset == "⚡ High Stress & Moderate Fatigue":
            p_age, p_bmi, p_diet, p_exercise = 31, 24.5, "Fair", "1–2 days/week"
            p_base_sleep, p_base_stress = 6.5, 7.5
            p_cycle_len, p_phase, p_flow, p_pain = 32, "Luteal", "Medium", 4
            p_mood, p_stress_c, p_sleep_c = 5, 8.0, 6.0
            p_energy, p_focus, p_work_lost = 4, 5, 4.0
            p_pcos, p_bc = 0, 0
        elif preset == "🩸 Heavy Flow & Dysmenorrhea (Elevated Pain)":
            p_age, p_bmi, p_diet, p_exercise = 26, 21.5, "Good", "1–2 days/week"
            p_base_sleep, p_base_stress = 7.0, 5.0
            p_cycle_len, p_phase, p_flow, p_pain = 27, "Menstrual", "Heavy", 8
            p_mood, p_stress_c, p_sleep_c = 4, 6.5, 6.0
            p_energy, p_focus, p_work_lost = 3, 4, 8.0
            p_pcos, p_bc = 0, 0
        elif preset == "🌿 PCOS / Irregular Cycle Pattern":
            p_age, p_bmi, p_diet, p_exercise = 29, 28.0, "Fair", "3–4 days/week"
            p_base_sleep, p_base_stress = 7.0, 6.0
            p_cycle_len, p_phase, p_flow, p_pain = 38, "Follicular", "Light", 5
            p_mood, p_stress_c, p_sleep_c = 6, 6.0, 7.0
            p_energy, p_focus, p_work_lost = 5, 6, 2.0
            p_pcos, p_bc = 1, 1
        else:
            p_age, p_bmi, p_diet, p_exercise = 28, 22.5, "Good", "3–4 days/week"
            p_base_sleep, p_base_stress = 7.5, 4.0
            p_cycle_len, p_phase, p_flow, p_pain = 28, "Menstrual", "Medium", 3
            p_mood, p_stress_c, p_sleep_c = 7, 4.0, 7.5
            p_energy, p_focus, p_work_lost = 7, 7, 0.0
            p_pcos, p_bc = 0, 0

        # Form layout in 3 clear columns
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        with col1:
            st.markdown('<div class="metric-card"><div class="card-header">👤 1. Profile & Lifestyle</div>', unsafe_allow_html=True)
            age = st.slider("Age (years)", 15, 60, p_age, help="Age in years")
            bmi = st.slider("BMI (Body Mass Index)", 15.0, 45.0, float(p_bmi), step=0.1)

            diet_quality = st.selectbox(
                "Diet Quality",
                ["Good", "Fair", "Poor", "Excellent"],
                index=["Good", "Fair", "Poor", "Excellent"].index(p_diet)
            )
            exercise_freq = st.selectbox(
                "Exercise Frequency",
                ["Never", "1–2 days/week", "3–4 days/week", "5–6 days/week", "Daily"],
                index=["Never", "1–2 days/week", "3–4 days/week", "5–6 days/week", "Daily"].index(p_exercise)
            )
            base_sleep = st.slider("Baseline Sleep Hours", 4.0, 11.0, float(p_base_sleep), step=0.5)
            base_stress = st.slider("Baseline Stress Level (0-10)", 0.0, 10.0, float(p_base_stress), step=0.5)

            with st.expander("Additional Health Factors"):
                pcos = st.checkbox("Diagnosed with PCOS", value=bool(p_pcos))
                birth_control = st.checkbox("Using Hormonal Birth Control", value=bool(p_bc))
                caffeine = st.slider("Caffeine Intake (cups/day)", 0.0, 6.0, 1.5, step=0.5)
                water = st.slider("Water Intake (liters/day)", 0.5, 5.0, 2.0, step=0.5)
                smoking = st.selectbox("Smoking Status", ["Non-smoker", "Former", "Smoker"], index=0)
                alcohol = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Moderate", "Frequent"], index=0)
                state = st.selectbox("State / Region", ["California", "New York", "Texas", "Florida", "Illinois", "Other"], index=0)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card"><div class="card-header">🩸 2. Current Cycle Observations</div>', unsafe_allow_html=True)
            cycle_len = st.slider("Cycle Length (Days)", 18, 45, p_cycle_len)
            cycle_phase = st.selectbox("Current Cycle Phase", ["Menstrual", "Follicular", "Ovulation", "Luteal"], index=["Menstrual", "Follicular", "Ovulation", "Luteal"].index(p_phase))
            flow_level = st.selectbox("Flow Level", ["Light", "Medium", "Heavy", "Very Heavy"], index=["Light", "Medium", "Heavy", "Very Heavy"].index(p_flow))

            pain_level = st.slider("Pain / Cramping Level (0-10)", 0, 10, int(p_pain), help="0 = No pain, 10 = Debilitating cramping")
            mood_score = st.slider("Emotional Wellbeing / Mood (0-10)", 0, 10, int(p_mood), help="0 = Severe distress/irritability, 10 = Stable, positive mood")
            cycle_stress = st.slider("Cycle Stress Score (0-10)", 0.0, 10.0, float(p_stress_c), step=0.5)
            cycle_sleep = st.slider("Cycle Sleep Duration (Hours)", 4.0, 12.0, float(p_sleep_c), step=0.5)
            energy_level = st.slider("Energy & Vitality (0-10)", 0, 10, int(p_energy))
            focus_score = st.slider("Cognitive Focus / Concentration (0-10)", 0, 10, int(p_focus))
            work_lost = st.number_input("Work / School Hours Lost Due to Symptoms", min_value=0.0, max_value=40.0, value=float(p_work_lost), step=1.0)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card"><div class="card-header">📜 3. Cycle History Tracking Mode</div>', unsafe_allow_html=True)
            history_mode = st.radio(
                "Tracking History Available:",
                [
                    "🌱 First-time user (Cycle 1 / No prior log)",
                    "📈 Returning user (I have last cycle's records)"
                ],
                index=0
            )

            has_history = "Returning user" in history_mode
            hist_cycle: Optional[HistoricalCycleInput] = None

            if not has_history:
                st.info(
                    "**Cohort Baseline Policy:** Since this is Cycle 1, the pipeline applies "
                    "reference medians derived from 15,976 training records for lag features. "
                    "Your personal data is never blended with an individual stranger's records."
                )
            else:
                st.success("Historical cycle metrics will personalize your lag features.")
                prev_health = st.slider("Previous Cycle Overall Health (0-10)", 2.0, 10.0, 8.0, step=0.2)
                prev_len = st.slider("Previous Cycle Length (Days)", 20, 42, 28)
                prev_pain = st.slider("Previous Cycle Pain (0-10)", 0, 10, 3)
                prev_stress = st.slider("Previous Cycle Stress (0-10)", 0.0, 10.0, 4.0, step=0.5)
                prev_sleep = st.slider("Previous Cycle Sleep (Hours)", 4.0, 11.0, 7.5, step=0.5)
                prev_energy = st.slider("Previous Cycle Energy (0-10)", 0, 10, 7)
                prev_mood = st.slider("Previous Cycle Mood (0-10)", 0, 10, 7)
                prev_focus = st.slider("Previous Cycle Concentration (0-10)", 0, 10, 7)

                hist_cycle = HistoricalCycleInput(
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

        # Trigger Prediction
        st.markdown("<br>", unsafe_allow_html=True)
        btn_predict = st.button("🔮 Generate Next-Cycle Health Prediction", use_container_width=True)

        # Execute Prediction
        if btn_predict or preset != "Custom (Enter Your Own)":
            profile_obj = UserProfileInput(
                age=float(age),
                bmi=float(bmi),
                sleep_hours=float(base_sleep),
                stress_score_baseline=float(base_stress),
                state=state,
                diet_quality=diet_quality,
                exercise_frequency=exercise_freq,
                caffeine_intake=float(caffeine),
                water_intake_liters=float(water),
                alcohol_consumption=alcohol,
                smoking_status=smoking,
                birth_control_use=1 if birth_control else 0,
                pcos_diagnosed=1 if pcos else 0,
            )

            cycle_obj = CurrentCycleInput(
                cycle_length_days=float(cycle_len),
                pain_level=float(pain_level),
                mood_score=float(mood_score),
                stress_score_cycle=float(cycle_stress),
                sleep_hours_cycle=float(cycle_sleep),
                energy_level=float(energy_level),
                concentration_score=float(focus_score),
                work_hours_lost=float(work_lost),
                cycle_phase=cycle_phase,
                flow_level=flow_level,
                pms_symptoms="Moderate" if pain_level > 5 else "Mild"
            )

            with st.spinner("Analyzing physiological patterns and computing forecast..."):
                result = predictor.predict(profile_obj, cycle_obj, historical_cycle=hist_cycle)

            st.markdown("<hr style='margin: 25px 0; border: none; border-top: 1px solid #EFE4EA;'>", unsafe_allow_html=True)
            st.markdown("## 📊 Forecast & Health Analysis")

            # Result status banner
            banner_class = "result-optimal" if result.score_category == "Optimal" else ("result-moderate" if result.score_category == "Moderate" else "result-suboptimal")
            badge_icon = "🟢" if result.score_category == "Optimal" else ("🟡" if result.score_category == "Moderate" else "🔴")

            st.markdown(f"""
            <div class="result-banner {banner_class}">
                <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 4px;">
                    {badge_icon} Next-Cycle Forecast: {result.predicted_next_health_score:.2f} / 10.0 — {result.score_category} Health
                </div>
                <div style="font-size: 1rem; opacity: 0.9;">
                    {result.summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Two-column result breakdown
            res_col1, res_col2 = st.columns([1, 1], gap="medium")

            with res_col1:
                st.markdown('<div class="metric-card"><div class="card-header">🎯 Predicted Score Gauge</div>', unsafe_allow_html=True)
                fig_gauge = create_gauge_chart(result.predicted_next_health_score, result.score_category)
                st.plotly_chart(fig_gauge, use_container_width=True)

                if result.warnings:
                    for w in result.warnings:
                        st.caption(f"ℹ️ **Pipeline Notice:** {w}")
                st.markdown('</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="metric-card"><div class="card-header">🕸️ Wellbeing Balance Profile</div>', unsafe_allow_html=True)
                curr_radar_dict = {
                    "pain_level": pain_level,
                    "mood_score": mood_score,
                    "stress_score_cycle": cycle_stress,
                    "sleep_hours_cycle": cycle_sleep,
                    "energy_level": energy_level,
                    "concentration_score": focus_score,
                }
                prev_radar_dict = None
                if hist_cycle:
                    prev_radar_dict = {
                        "previous_pain_level": hist_cycle.previous_pain_level,
                        "previous_mood_score": hist_cycle.previous_mood_score,
                        "previous_stress_score": hist_cycle.previous_stress_score,
                        "previous_sleep_hours": hist_cycle.previous_sleep_hours,
                        "previous_energy_level": hist_cycle.previous_energy_level,
                        "previous_concentration_score": hist_cycle.previous_concentration_score,
                    }

                fig_radar = create_radar_chart(curr_radar_dict, prev_radar_dict)
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Targeted clinical recommendations
            st.markdown('<div class="metric-card"><div class="card-header">💡 Personalized Cycle Recommendations</div>', unsafe_allow_html=True)
            rec_cols = st.columns(3)
            with rec_cols[0]:
                if cycle_stress > 6.0 or base_stress > 6.0:
                    st.error("🧘 **Stress Management Target**")
                    st.write("Elevated cortisol directly dampens next-cycle luteal stability. Integrate 15 min daily mindfulness, somatic breathing, or gentle restorative yoga.")
                else:
                    st.success("🧘 **Stress Resilience Strong**")
                    st.write("Stress indicators are favorable. Consistent stress moderation supports predictable ovulation and balanced prostaglandin levels.")

            with rec_cols[1]:
                if pain_level >= 5:
                    st.warning("🌿 **Prostaglandin & Pain Modulation**")
                    st.write("Cramping intensity suggests heightened inflammatory prostaglandins. Discuss magnesium glycinate (300mg), omega-3s, and targeted heat therapy with your doctor.")
                else:
                    st.success("🌿 **Pain Level Well Controlled**")
                    st.write("Minimal dysmenorrhea observed. Maintaining adequate hydration and anti-inflammatory whole foods helps preserve this comfort.")

            with rec_cols[2]:
                if cycle_sleep < 7.0 or base_sleep < 7.0:
                    st.warning("🌙 **Sleep Hygiene Restoration**")
                    st.write("Sub-7-hour sleep patterns correlate with decreased next-cycle energy. Aim for consistent sleep-wake cycles and evening blue-light restriction.")
                else:
                    st.success("🌙 **Sleep Restorative**")
                    st.write("Adequate sleep duration supports healthy nocturnal melatonin and pituitary hormone secretion.")

            st.markdown('</div>', unsafe_allow_html=True)

            # Exportable summary
            with st.expander("📄 View Clinician Consultation Summary"):
                st.text(f"""
HERAI CYCLE FORECAST REPORT
Date: 2026-10-29
Predicted Next-Cycle Health Score: {result.predicted_next_health_score:.2f} / 10.0 ({result.score_category})
Mode: {'Longitudinal (User Prior Cycle)' if has_history else 'First-Time (Cohort Baseline)'}

BIOMARKERS & OBSERVATIONS:
- Age: {age} | BMI: {bmi:.1f} | Baseline Stress: {base_stress:.1f}
- Current Cycle Length: {cycle_len} days | Phase: {cycle_phase} | Flow: {flow_level}
- Pain Level: {pain_level}/10 | Mood: {mood_score}/10 | Cycle Stress: {cycle_stress:.1f}/10
- Cycle Sleep: {cycle_sleep:.1f} hrs | Energy: {energy_level}/10 | Focus: {focus_score}/10
- Work/Activity Hours Lost: {work_lost:.1f} hrs

DISCLAIMER:
{result.disclaimer}
                """)

    # ==========================================
    # TAB 2: COHORT & LONGITUDINAL TRENDS
    # ==========================================
    with tab_analytics:
        st.markdown("### 📈 Population Cohort Distribution & User Trajectories")
        st.markdown(
            "Explore how menstrual health indicators vary across the validated training cohort "
            "(15,976 cycles across 2,000 users) and inspect longitudinal multi-cycle histories."
        )

        an_col1, an_col2 = st.columns([1, 1], gap="medium")

        with an_col1:
            st.markdown('<div class="metric-card"><div class="card-header">📊 Cohort Score Distribution</div>', unsafe_allow_html=True)
            if not data_df.empty and "next_health_score" in data_df.columns:
                fig_dist = create_cohort_distribution_chart(data_df["next_health_score"].dropna())
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption("Distribution Mean: **8.55**, Median: **8.60**, Std Dev: **1.11**, Range: **4.40 - 10.00**.")
            else:
                st.info("Loading cohort distribution data...")
            st.markdown('</div>', unsafe_allow_html=True)

        with an_col2:
            st.markdown('<div class="metric-card"><div class="card-header">🔍 Multi-Cycle Patient Trajectory</div>', unsafe_allow_html=True)
            if not data_df.empty and "user_id" in data_df.columns:
                sample_users = ["U00001", "U00002", "U00003", "U00004", "U00005"]
                selected_user = st.selectbox("Select Patient Cohort Record:", sample_users, index=0)
                fig_traj = create_user_trajectory_chart(data_df, selected_user)
                st.plotly_chart(fig_traj, use_container_width=True)
                st.caption(f"Tracking longitudinal score stability across consecutive logged cycles for {selected_user}.")
            else:
                st.info("Dataset records unavailable.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card"><div class="card-header">💡 Key Epidemiological Insights from Cohort Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Autoregressive Persistence:** A user's current-cycle and prior-cycle overall health scores represent the single strongest positive predictors of their next cycle ($r = 0.67$), showing that menstrual wellbeing exhibits high temporal continuity.
        2. **Stress Penalty:** Elevated cycle stress and baseline stress impart a heavy standardized negative weight ($-0.67$) on next-cycle health score.
        3. **Pain Interference:** Severe dysmenorrhea (pain >= 7) is strongly associated with an average drop of 1.4 points on the overall health scale.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 3: MODEL ARCHITECTURE & AUDIT
    # ==========================================
    with tab_model:
        st.markdown("### 🔬 Model Architecture, Evaluation & Leakage Verification")
        st.markdown(
            "HERAI employs a rigorous machine learning pipeline with strict train/test user isolation "
            "(`GroupShuffleSplit`), eliminating feature leakage and evaluating candidate regressors."
        )

        mod_col1, mod_col2 = st.columns([1, 1], gap="medium")

        with mod_col1:
            st.markdown('<div class="metric-card"><div class="card-header">📐 Model Candidate Benchmarking</div>', unsafe_allow_html=True)
            benchmarks = pd.DataFrame([
                {"Model": "Ridge Regression (42 features)", "MAE": 0.6452, "RMSE": 0.8029, "R² Score": 0.4616, "Status": "Evaluated"},
                {"Model": "Random Forest (42 features)", "MAE": 0.6550, "RMSE": 0.8130, "R² Score": 0.4480, "Status": "Evaluated"},
                {"Model": "Ridge Without Audited Features (37 features)", "MAE": 0.6448, "RMSE": 0.8025, "R² Score": 0.4620, "Status": "👑 Selected Production Model"},
                {"Model": "Dummy Mean Baseline", "MAE": 0.9113, "RMSE": 1.0942, "R² Score": -0.0000, "Status": "Baseline"}
            ])
            st.dataframe(benchmarks, hide_index=True, use_container_width=True)

            st.markdown("""
            **Why the 37-Feature Ridge Model was Selected:**
            - **Eliminates Invasive Lab Features:** Drops `estrogen_pgml`, `progesterone_ngml`, and `ovulation_result`, which require blood draws or clinical testing unavailable to home users.
            - **Removes Lookahead Bias:** Excludes `log_consistency_score` and `prepared_before_period`.
            - **Superior Generalization:** Achieved the lowest MAE ($0.6448$) and highest $R^2$ ($0.4620$) on unseen test users.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with mod_col2:
            st.markdown('<div class="metric-card"><div class="card-header">⚖️ Standardized Ridge Driver Weights</div>', unsafe_allow_html=True)
            feature_sample = [
                "Baseline Stress Score", "Overall Health Score", "Previous Health Score",
                "Pain Level", "Previous Pain Level", "PCOS Diagnosed", "Sleep Hours",
                "Previous Stress Score", "Cycle Stress Score", "Work Hours Lost"
            ]
            coeff_sample = [-0.672, 0.453, 0.406, -0.291, -0.250, -0.165, 0.151, -0.137, -0.113, -0.095]
            fig_coeff = create_feature_importance_chart(feature_sample, coeff_sample, top_n=10)
            st.plotly_chart(fig_coeff, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card"><div class="card-header">🛡️ Data Leakage Audit Summary</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Target Alignment:** Evaluated across 15,976 records with **0 mismatches** (`next_health_score == overall_health_score.shift(-1)` within `user_id`).
        - **User Group Isolation:** `GroupShuffleSplit` strictly partitions users (1,600 training, 400 test) with **zero user overlap**.
        - **In-Pipeline Imputation:** `SimpleImputer` and `StandardScaler` are fitted exclusively on training folds, eliminating global leakage.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 4: HOLISTIC CARE & CYCLE GUIDE
    # ==========================================
    with tab_guide:
        st.markdown("### 🌿 Comprehensive Menstrual Health & Cycle Guide")
        st.markdown(
            "Understanding the four distinct phases of the menstrual cycle empowers proactive symptom "
            "management, hormone balance, and lifestyle harmony."
        )

        phase_cols = st.columns(4)

        with phase_cols[0]:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #882D52;">
                <h4 style="color: #882D52;">1. Menstrual Phase</h4>
                <p><b>Days 1–5</b> (Hormones at baseline)</p>
                <ul>
                    <li><b>Focus:</b> Rest, gentle movement, hydration.</li>
                    <li><b>Nutrition:</b> Iron-rich foods, bone broth, warming stews.</li>
                    <li><b>Care:</b> Warm baths, heat packs, restorative sleep.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with phase_cols[1]:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #D47A94;">
                <h4 style="color: #D47A94;">2. Follicular Phase</h4>
                <p><b>Days 6–13</b> (Estrogen rising)</p>
                <ul>
                    <li><b>Focus:</b> Creative energy, high focus, building stamina.</li>
                    <li><b>Nutrition:</b> Fermented foods, sprouts, cruciferous vegetables.</li>
                    <li><b>Care:</b> Strength training, cardio, collaborative projects.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with phase_cols[2]:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #388E3C;">
                <h4 style="color: #388E3C;">3. Ovulatory Phase</h4>
                <p><b>Days 14–16</b> (Peak LH & Estrogen)</p>
                <ul>
                    <li><b>Focus:</b> Peak energy, confidence, metabolic zenith.</li>
                    <li><b>Nutrition:</b> Antioxidant berries, leafy greens, zinc.</li>
                    <li><b>Care:</b> High-intensity interval training (HIIT), social events.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with phase_cols[3]:
            st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #F57C00;">
                <h4 style="color: #F57C00;">4. Luteal Phase</h4>
                <p><b>Days 17–28</b> (Progesterone dominant)</p>
                <ul>
                    <li><b>Focus:</b> Wind-down, boundary-setting, restorative routines.</li>
                    <li><b>Nutrition:</b> Magnesium-rich foods, complex carbs, seeds.</li>
                    <li><b>Care:</b> Pilates, walking, reducing caffeine & alcohol.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="metric-card"><div class="card-header">🩺 When to Seek Medical Attention (Red Flag Symptoms)</div>', unsafe_allow_html=True)
        st.markdown("""
        Please consult a certified gynecologist, endocrinologist, or primary care provider if you experience:
        - **Severe or Debilitating Pain:** Cramping that impairs daily living, does not respond to OTC pain relief, or suggests endometriosis.
        - **Excessive Bleeding (Menorrhagia):** Soaking through a pad or tampon every 1–2 hours, or passing clots larger than a quarter.
        - **Sudden Irregularity:** Cycles shorter than 21 days, longer than 35 days, or stopping for 3+ consecutive months without pregnancy.
        - **Severe Mood Shifts:** Severe depressive episodes, extreme dysphoria, or PMDD symptoms during the luteal phase.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="disclaimer-box">
        <b>Clinical & Legal Disclaimer:</b> HERAI is an educational prototype and predictive exploration platform. 
        It is not a medical device, diagnostic test, or clinical decision support system. Predictions represent statistical estimations 
        based on historical observational cohort patterns and must not be used for diagnosis, medical treatment, or contraception.
        <br><b>Data Privacy:</b> All inputs provided during this session are processed locally within your runtime environment and are never transmitted to external cloud servers.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
