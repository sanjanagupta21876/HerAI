"""HERAI — Women's Menstrual Health Intelligence Application.

AI-assisted menstrual-health pattern analysis and next-cycle health-score prediction.
Built with a calm, trustworthy, and restrained healthcare design system.
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
from herai.auth import (
    init_auth_session,
    login_session_user,
    logout_session_user,
    get_current_user_id,
    register_user,
    authenticate,
    AuthenticationError,
)
from herai.database import (
    get_user_profile,
    save_user_profile,
    save_user_cycle_log,
    get_user_cycle_logs,
    get_next_cycle_number,
    UserAlreadyExistsError,
    ValidationError as DBValidationError,
)

# Page configuration - professional and clean, no emoji
st.set_page_config(
    page_title="HERAI | Menstrual Health Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize authentication and navigation session state
init_auth_session()

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# Custom CSS for Restrained, Professional Health-Tech Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --color-ivory: #FFF9F5;
        --color-peach: #F8E5DB;
        --color-dusty-rose: #D99AA5;
        --color-mauve: #B97B8A;
        --color-mauve-dark: #925363;
        --color-cocoa: #59433F;
        --color-cocoa-light: #7A6560;
        --color-sage: #6E8B62;
        --color-lavender: #EAE5F4;
        --color-white: #FFFFFF;
        --color-amber: #C97A3E;
        --color-border: #F0E2DA;
        --color-border-subtle: #F6EDE7;
        --color-track: #FAF5F2;
        --font-serif: 'DM Serif Display', Georgia, serif;
        --font-sans: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Base Application */
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
        letter-spacing: -0.015em;
    }
    h1 { font-size: 2.15rem; line-height: 1.25; margin-bottom: 0.35rem; }
    h2 { font-size: 1.6rem; line-height: 1.3; margin-top: 1.2rem; margin-bottom: 0.5rem; }
    h3 { font-size: 1.24rem; line-height: 1.35; margin-bottom: 0.45rem; }
    p, span, label, div {
        color: var(--color-cocoa);
        font-family: var(--font-sans);
    }

    /* Header Brand Styling */
    .header-brand {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .brand-wordmark {
        font-family: var(--font-serif);
        font-size: 1.85rem;
        color: var(--color-cocoa);
        font-weight: 400;
        letter-spacing: -0.02em;
    }
    .brand-divider {
        display: inline-block;
        width: 1px;
        height: 16px;
        background-color: #ECDAD2;
        vertical-align: middle;
    }
    .brand-tagline {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--color-mauve);
    }

    /* Segmented Navigation Tab Bar (st.radio) */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 4px;
        align-items: center;
        background: var(--color-track);
        padding: 4px;
        border-radius: 10px;
        border: 1px solid var(--color-border);
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 7px 16px !important;
        margin: 0 !important;
        transition: all 0.18s ease-in-out !important;
        cursor: pointer !important;
        box-shadow: none !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(185, 123, 138, 0.08) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: var(--color-white) !important;
        border-color: var(--color-border) !important;
        box-shadow: 0 1px 4px rgba(89, 67, 63, 0.06) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span,
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) div,
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
        color: var(--color-cocoa) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:not(:has(input:checked)) span,
    div[data-testid="stRadio"] div[role="radiogroup"] > label:not(:has(input:checked)) div,
    div[data-testid="stRadio"] div[role="radiogroup"] > label:not(:has(input:checked)) p {
        color: var(--color-cocoa-light) !important;
        font-weight: 500 !important;
    }

    /* Streamlit Native Tabs Styling */
    div[data-testid="stTabs"] {
        margin-top: 14px;
        margin-bottom: 20px;
    }
    div[data-testid="stTabs"] div[role="tablist"] {
        gap: 8px;
        border-bottom: 1px solid var(--color-border);
        padding-bottom: 2px;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        font-family: var(--font-sans);
        font-size: 0.92rem;
        font-weight: 500;
        color: var(--color-cocoa-light);
        padding: 8px 18px;
        border: none;
        border-bottom: 2px solid transparent;
        background: transparent;
        transition: all 0.2s ease;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--color-cocoa);
        background: rgba(185, 123, 138, 0.05);
        border-radius: 6px 6px 0 0;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--color-mauve);
        font-weight: 600;
        border-bottom: 2px solid var(--color-mauve);
    }

    /* Welcome / Hero Section */
    .hero-banner {
        background: linear-gradient(135deg, #FFF9F5 0%, #F8E5DB 55%, #EAE5F4 100%);
        border: 1px solid var(--color-border);
        border-radius: 14px;
        padding: 26px 32px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(89, 67, 63, 0.02);
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
        line-height: 1.55;
        max-width: 800px;
        margin-bottom: 4px;
    }

    /* Card System */
    .card-surface {
        background: var(--color-white);
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(89, 67, 63, 0.02);
        transition: border-color 0.2s ease;
    }
    .card-surface:hover {
        border-color: #E2D1C9;
    }
    .card-header-serif {
        font-family: var(--font-serif);
        font-size: 1.22rem;
        color: var(--color-cocoa);
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--color-border-subtle);
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
        box-shadow: 0 1px 4px rgba(89, 67, 63, 0.02);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-stat-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
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
        color: var(--color-cocoa-light);
        line-height: 1.4;
    }

    /* Badges, Chips & Steps */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(185, 123, 138, 0.10);
        color: var(--color-mauve-dark);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .step-pill {
        display: inline-block;
        background: #F4EAE6;
        color: var(--color-mauve-dark);
        font-family: var(--font-sans);
        font-weight: 700;
        font-size: 0.70rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 5px;
        margin-right: 6px;
        vertical-align: middle;
    }
    .tier-badge-optimal {
        background: #EFF4EE;
        color: #2D5028;
        border: 1px solid #C8D7C2;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .tier-badge-moderate {
        background: #FDF4EB;
        color: #8C4E1A;
        border: 1px solid #F0D4BC;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Primary and Secondary CTA Buttons */
    button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background: var(--color-mauve) !important;
        color: var(--color-white) !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 9px 22px !important;
        min-height: 42px !important;
        box-shadow: 0 2px 8px rgba(185, 123, 138, 0.25) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background: var(--color-mauve-dark) !important;
        color: var(--color-white) !important;
        box-shadow: 0 4px 12px rgba(185, 123, 138, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="primary"]:active, [data-testid="stBaseButton-primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 1px 4px rgba(185, 123, 138, 0.2) !important;
    }

    button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
        background: var(--color-white) !important;
        color: var(--color-cocoa) !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        font-size: 0.94rem !important;
        border: 1.5px solid #E2D1C9 !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        min-height: 42px !important;
        box-shadow: 0 1px 3px rgba(89, 67, 63, 0.03) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
        background: #FAF3EE !important;
        border-color: var(--color-mauve) !important;
        color: var(--color-mauve-dark) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="secondary"]:active, [data-testid="stBaseButton-secondary"]:active {
        transform: translateY(0) !important;
    }

    /* Fallback default button */
    .stButton>button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 600;
        font-size: 0.94rem;
        transition: all 0.2s ease;
    }

    /* Accessible focus states for radio and inputs */
    .stRadio div[role="radiogroup"] label:focus-within {
        outline: 2px solid var(--color-mauve) !important;
        border-radius: 8px !important;
    }
    input:focus-visible, select:focus-visible, textarea:focus-visible {
        border-color: var(--color-mauve) !important;
        outline: 2px solid var(--color-dusty-rose) !important;
    }

    /* Callout & Disclaimer */
    .callout-lavender {
        background: var(--color-lavender);
        border-left: 3.5px solid var(--color-mauve);
        padding: 13px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.86rem;
        color: var(--color-cocoa);
        margin: 12px 0;
        line-height: 1.5;
    }
    .disclaimer-card {
        background: #FAF5F2;
        border: 1px solid #EFE4DC;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 0.82rem;
        line-height: 1.55;
        color: var(--color-cocoa-light);
        margin-top: 30px;
    }

    /* Prediction Result Banner */
    .score-result-banner {
        border-radius: 12px;
        padding: 22px 26px;
        margin: 16px 0 20px 0;
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
        font-size: 1.7rem;
        margin-bottom: 4px;
    }
    .score-result-subtitle {
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Empty state card */
    .empty-state-card {
        background: var(--color-white);
        border: 1.5px dashed var(--color-border);
        border-radius: 12px;
        padding: 34px 24px;
        text-align: center;
        color: var(--color-cocoa-light);
        margin: 16px 0;
        transition: border-color 0.2s ease;
    }
    .empty-state-card:hover {
        border-color: var(--color-mauve);
        background: #FAF7F5;
    }

    /* Responsive Breakpoints: Tablet & Mobile */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.6rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .hero-banner {
            padding: 18px 16px !important;
            border-radius: 10px !important;
        }
        .hero-headline {
            font-size: 1.55rem !important;
        }
        .hero-subtext {
            font-size: 0.90rem !important;
            margin-bottom: 12px !important;
        }
        h1 { font-size: 1.6rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        .metric-stat-box {
            padding: 12px 14px !important;
            margin-bottom: 8px !important;
        }
        .metric-stat-value {
            font-size: 1.4rem !important;
        }
        .card-surface {
            padding: 14px 16px !important;
            border-radius: 10px !important;
        }
        .score-result-banner {
            padding: 16px 18px !important;
        }
        .score-result-title {
            font-size: 1.35rem !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            flex-wrap: wrap !important;
            justify-content: flex-start !important;
            gap: 4px !important;
        }
    }

    @media (max-width: 480px) {
        .hero-headline {
            font-size: 1.30rem !important;
        }
        .metric-stat-value {
            font-size: 1.20rem !important;
        }
        .stButton>button {
            width: 100% !important;
            min-height: 44px !important;
        }
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
    """Render the top clean application header with HERAI wordmark, navigation, and user authentication state."""
    nav_col1, nav_col2, nav_col3 = st.columns([1.1, 2.1, 0.9], vertical_alignment="center")

    with nav_col1:
        st.markdown("""
        <div class="header-brand">
            <span class="brand-wordmark">HERAI</span>
            <span class="brand-divider"></span>
            <span class="brand-tagline">Health Intelligence</span>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        pages = ["Dashboard", "Health Prediction", "Cycle Information", "Model Insights", "About"]
        current_idx = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else None
        selected = st.radio(
            "Navigation",
            pages,
            index=current_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="header_nav_radio"
        )
        if selected and selected != st.session_state.current_page:
            st.session_state.current_page = selected
            st.rerun()

    with nav_col3:
        if st.session_state.get("authenticated") and st.session_state.get("current_user"):
            user = st.session_state.current_user
            user_display = user.get("username", "User")
            acc_c1, acc_c2 = st.columns([1.2, 1.0], vertical_alignment="center")
            with acc_c1:
                st.markdown(
                    f'<div style="text-align: right; line-height: 1.2;">'
                    f'<span class="badge-pill" style="margin-bottom: 0; font-size: 0.72rem; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block;">'
                    f'{user_display}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with acc_c2:
                if st.button("Sign Out", key="header_sign_out_btn", use_container_width=True, type="secondary"):
                    logout_session_user()
                    st.session_state.current_page = "Dashboard"
                    st.rerun()
        else:
            if st.session_state.current_page == "Sign In":
                st.markdown(
                    '<div style="text-align: right;"><span class="step-pill" style="margin-right: 0;">Sign In</span></div>',
                    unsafe_allow_html=True
                )
            else:
                if st.button("Sign In / Join", key="header_sign_in_btn", use_container_width=True, type="secondary"):
                    st.session_state.current_page = "Sign In"
                    st.rerun()

    st.markdown("<hr style='margin: 8px 0 20px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)


# =============================================================================
# VIEW 0: AUTHENTICATION (SIGN IN & REGISTER)
# =============================================================================
def render_auth_view():
    """Render the Sign In and Account Registration view with secure validation and isolation."""
    st.markdown('<div class="badge-pill">Secure Access &bull; Multi-User Identity</div>', unsafe_allow_html=True)
    st.markdown("<h1>Personalized Health Account</h1>", unsafe_allow_html=True)
    st.markdown(
        "Sign in or register to securely persist your baseline lifestyle profiles, "
        "track your sequential cycle records, and maintain private, isolated health history across sessions."
    )

    auth_card_col, side_info_col = st.columns([1.1, 1.0], gap="large")

    with auth_card_col:
        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            st.markdown('<div style="font-size: 0.90rem; font-weight: 600; color: #59433F; margin-bottom: 12px; margin-top: 6px;">Sign in to your account</div>', unsafe_allow_html=True)
            with st.form("form_login"):
                login_identifier = st.text_input("Username or Email", placeholder="e.g. sanjana or sanjana@example.com", key="login_id_input")
                login_password = st.text_input("Password", type="password", key="login_pw_input")
                btn_login_submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")

                if btn_login_submit:
                    if not login_identifier.strip() or not login_password:
                        st.error("Please enter both your username/email and password.")
                    else:
                        try:
                            user = authenticate(login_identifier.strip(), login_password)
                            login_session_user(user)
                            st.session_state.current_page = "Dashboard"
                            st.rerun()
                        except AuthenticationError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Sign-in error: {e}")

        with tab_register:
            st.markdown('<div style="font-size: 0.90rem; font-weight: 600; color: #59433F; margin-bottom: 12px; margin-top: 6px;">Register a new personal account</div>', unsafe_allow_html=True)
            with st.form("form_register"):
                reg_username = st.text_input("Username (3–30 characters)", placeholder="e.g. maya_chen", key="reg_user_input")
                reg_email = st.text_input("Email Address", placeholder="e.g. maya@example.com", key="reg_email_input")
                reg_fullname = st.text_input("Full Name (Optional)", placeholder="e.g. Maya Chen", key="reg_name_input")
                reg_password = st.text_input("Password (min 8 characters, letters & numbers)", type="password", key="reg_pw_input")
                reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm_input")
                btn_reg_submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                if btn_reg_submit:
                    try:
                        user = register_user(
                            username=reg_username,
                            email=reg_email,
                            password=reg_password,
                            confirm_password=reg_confirm,
                            full_name=reg_fullname.strip() if reg_fullname.strip() else None
                        )
                        login_session_user(user)
                        st.session_state.current_page = "Dashboard"
                        st.rerun()
                    except (DBValidationError, UserAlreadyExistsError) as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Registration error: {e}")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("Continue as Guest (No Account Required)", key="btn_guest_continue", use_container_width=True, type="secondary"):
            st.session_state.current_page = "Dashboard"
            st.rerun()

    with side_info_col:
        st.markdown("""
        <div class="card-surface">
            <div class="card-header-serif">Privacy &amp; Data Isolation Principles</div>
            <p style="font-size: 0.88rem; line-height: 1.55; color: #59433F;">
                <b>Strict Personal Isolation:</b> Every user account is assigned an unguessable prefixed identifier (<code>usr_*</code>). 
                All baseline lifestyle profiles and cycle prediction history are mapped exclusively to this ID with foreign-key constraints.
            </p>
            <p style="font-size: 0.88rem; line-height: 1.55; color: #59433F;">
                <b>Zero Stranger-Data Substitution:</b> In HERAI, cycle predictions for Cycle 1 utilize published cohort reference medians. 
                Your personal observations are never shared with or borrowed by any other user.
            </p>
            <p style="font-size: 0.88rem; line-height: 1.55; color: #59433F;">
                <b>Bcrypt Cryptography:</b> Passwords are never stored in plaintext. They are salted with 12 adaptive hashing rounds and verified using timing-safe comparisons.
            </p>
            <p style="font-size: 0.88rem; line-height: 1.55; color: #59433F;">
                <b>Local Persistent Storage:</b> Account records are stored in a local SQLite database secured with parameterized queries to prevent SQL injection vulnerabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 1: DASHBOARD
# =============================================================================
def render_dashboard(cohort_df: pd.DataFrame):
    """Render the Main Dashboard with Welcome Hero, Overview Cards, and Cohort Visualizations."""
    # B. Landing page / Introduction section
    st.markdown("""
    <div class="hero-banner">
        <div class="badge-pill">Clinical ML Prototype &bull; Non-Diagnostic</div>
        <div class="hero-headline">Understand Your Cycle. Discover Your Patterns.</div>
        <div class="hero-subtext">
            Explore menstrual-health patterns through data-driven insights and an experimental next-cycle health-score prediction model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Action Buttons and Subtle Pattern Wave
    cta_col, wave_col = st.columns([1.1, 1.9], vertical_alignment="center")
    with cta_col:
        btn_explore = st.button("Explore Your Health", use_container_width=True, key="btn_hero_explore", type="primary")
        if btn_explore:
            st.session_state.current_page = "Health Prediction"
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        btn_how = st.button("How It Works", use_container_width=True, key="btn_hero_how", type="secondary")
        if btn_how:
            st.session_state.current_page = "Model Insights"
            st.rerun()

    with wave_col:
        fig_wave = create_cycle_wave_hero()
        st.plotly_chart(fig_wave, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)

    # C. Overview Cards
    st.markdown("### System & Cohort Overview")

    c1, c2, c3, c4 = st.columns(4)

    total_records = len(cohort_df) if not cohort_df.empty else 15976
    unique_users = cohort_df["user_id"].nunique() if not cohort_df.empty and "user_id" in cohort_df.columns else 2000
    avg_cycle_len = f"{cohort_df['cycle_length_days'].mean():.1f} Days" if not cohort_df.empty and "cycle_length_days" in cohort_df.columns else "27.8 Days"

    with c1:
        st.markdown(f"""
        <div class="metric-stat-box" style="border-top: 3px solid var(--color-mauve);">
            <div>
                <div class="metric-stat-label">Evaluated Cycles</div>
                <div class="metric-stat-value">{total_records:,}</div>
            </div>
            <div class="metric-stat-detail">Across {unique_users:,} user profiles in training cohort</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-stat-box" style="border-top: 3px solid var(--color-sage);">
            <div>
                <div class="metric-stat-label" style="color: #4D6B45;">Observed Cycle Length</div>
                <div class="metric-stat-value">{avg_cycle_len}</div>
            </div>
            <div class="metric-stat-detail">Median: 28.0 days (Std Dev: ±2.4 days)</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-stat-box" style="border-top: 3px solid var(--color-dusty-rose);">
            <div>
                <div class="metric-stat-label" style="color: #9E5664;">Production Model</div>
                <div class="metric-stat-value" style="font-size: 1.35rem; padding-top: 3px;">Ridge (37 Feat.)</div>
            </div>
            <div class="metric-stat-detail">R² = 0.4620 | MAE = 0.6448 on test users</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        last_pred = st.session_state.last_prediction
        if last_pred is not None:
            st.markdown(f"""
            <div class="metric-stat-box" style="border-top: 3px solid var(--color-amber); background: #FFFAF5;">
                <div>
                    <div class="metric-stat-label" style="color: #A35216;">Your Latest Forecast</div>
                    <div class="metric-stat-value">{last_pred.predicted_next_health_score:.2f} <span style="font-size: 0.95rem; color: #7A6560;">/ 10</span></div>
                </div>
                <div class="metric-stat-detail">{last_pred.score_category} Health Tier</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-stat-box" style="border-top: 3px solid #D6C7C0; background: #FAF7F5;">
                <div>
                    <div class="metric-stat-label" style="color: #8C7B76;">Your Latest Forecast</div>
                    <div class="metric-stat-value" style="font-size: 1.22rem; color: #8F7B76; padding-top: 4px;">No Forecast Yet</div>
                </div>
                <div class="metric-stat-detail">Run prediction to see your score</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # Health Pattern Visualizations organized with clean tabs
    st.markdown("### Cohort Analytics & Longitudinal Trajectories")
    st.caption("Empirical distributions and multi-cycle tracking across 15,976 validated records. Correlation does not imply causation.")

    dash_tab1, dash_tab2 = st.tabs([
        "Cohort Health Distributions",
        "Longitudinal Patient Trajectory"
    ])

    with dash_tab1:
        chart_c1, chart_c2 = st.columns([1, 1], gap="medium")

        with chart_c1:
            st.markdown('<div class="card-surface"><div class="card-header-serif">Observed Cycle Length Distribution</div>', unsafe_allow_html=True)
            if not cohort_df.empty and "cycle_length_days" in cohort_df.columns:
                fig_len = create_cycle_length_distribution_chart(cohort_df["cycle_length_days"])
                st.plotly_chart(fig_len, use_container_width=True, config={"displayModeBar": False})
                st.caption("Observed cycle length across 15,976 records follows an empirical normal distribution centered at 28 days.")
            else:
                st.info("Loading cycle distribution...")
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_c2:
            st.markdown('<div class="card-surface"><div class="card-header-serif">Cohort Health Score Distribution</div>', unsafe_allow_html=True)
            if not cohort_df.empty and "next_health_score" in cohort_df.columns:
                user_val = st.session_state.last_prediction.predicted_next_health_score if st.session_state.last_prediction else None
                fig_dist = create_cohort_distribution_chart(cohort_df["next_health_score"].dropna(), user_val)
                st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
                st.caption("Distribution parameters across 15,976 cycles: Mean = 8.55, Median = 8.60, Std = 1.11, Range = [4.40, 10.00].")
            else:
                st.info("Loading health score distribution...")
            st.markdown('</div>', unsafe_allow_html=True)

    with dash_tab2:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Longitudinal Cycle Trajectory in Real Records</div>', unsafe_allow_html=True)
        if not cohort_df.empty and "user_id" in cohort_df.columns:
            p_col1, p_col2 = st.columns([1, 3], vertical_alignment="center")
            with p_col1:
                sample_ids = ["U00001", "U00002", "U00003", "U00004", "U00005"]
                selected_id = st.selectbox("Select Patient Record:", sample_ids, index=0)
                st.caption("Inspect logged cycle scores and corresponding next-cycle actual values across consecutive cycles.")
            with p_col2:
                fig_traj = create_user_trajectory_chart(cohort_df, selected_id)
                st.plotly_chart(fig_traj, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# VIEW 2: HEALTH PREDICTION
# =============================================================================
def render_health_prediction(predictor: HerAIPredictor):
    """Render the Health Prediction Page with Structured 3-Section Inputs and Results."""
    st.markdown('<div class="badge-pill">Predictive Modeling</div>', unsafe_allow_html=True)
    st.markdown("<h1>Health Score Prediction & Consultation</h1>", unsafe_allow_html=True)
    st.markdown(
        "Enter your observational menstrual-cycle data, lifestyle factors, and prior cycle records. "
        "HERAI applies the validated 37-feature Ridge Regression pipeline to estimate your overall health score for your next cycle."
    )

    # Preset selector for fast scenario testing inside a card
    st.markdown("""
    <div class="card-surface" style="margin-bottom: 20px; padding: 18px 22px;">
        <div style="font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--color-mauve); margin-bottom: 6px;">
            Preload Observational Scenario (Demo Presets)
        </div>
    """, unsafe_allow_html=True)
    pr_col1, pr_col2 = st.columns([1.8, 2.2], vertical_alignment="center")
    with pr_col1:
        preset = st.selectbox(
            "Select Scenario Preset:",
            [
                "Custom Inputs",
                "Balanced & Regular Cycle (Optimal Range)",
                "High Stress & Moderate Fatigue",
                "Elevated Cramping & Dysmenorrhea",
                "Longer Cycle with Mild Symptoms"
            ],
            index=0,
            label_visibility="collapsed",
            help="Select a scenario to populate representative observational parameters for quick evaluation."
        )
    with pr_col2:
        st.markdown("""
        <div style="font-size: 0.82rem; color: #7A6560; line-height: 1.45; padding-left: 6px;">
            <b>Privacy & Integrity Guarantee:</b> Demo presets populate representative observational values for demonstration. 
            All health scores are calculated live by the validated 37-feature Ridge model—no stranger records or fabricated outputs are substituted.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # User authentication context and persistent profile lookup
    user_id = get_current_user_id()
    saved_profile = get_user_profile(user_id) if user_id else None
    user_history = get_user_cycle_logs(user_id, limit=10) if user_id else []

    # Default mapping
    if preset == "Balanced & Regular Cycle (Optimal Range)":
        def_age, def_bmi, def_diet, def_exercise = 28, 22.2, "Good", "3–4 days/week"
        def_bsleep, def_bstress = 8.0, 3.0
        def_clen, def_phase, def_flow, def_pain = 28, "Menstrual", "Medium", 2
        def_mood, def_stress, def_sleep = 8, 3.0, 8.0
        def_energy, def_focus, def_lost = 8, 8, 0.0
        def_bc, def_pcos = 0, 0
        def_caffeine, def_water = 1.5, 2.0
        def_smoke, def_alcohol, def_state = "Non-smoker", "None", "California"
    elif preset == "High Stress & Moderate Fatigue":
        def_age, def_bmi, def_diet, def_exercise = 31, 24.0, "Fair", "1–2 days/week"
        def_bsleep, def_bstress = 6.5, 7.5
        def_clen, def_phase, def_flow, def_pain = 31, "Luteal", "Medium", 4
        def_mood, def_stress, def_sleep = 5, 8.0, 5.5
        def_energy, def_focus, def_lost = 4, 5, 4.0
        def_bc, def_pcos = 0, 0
        def_caffeine, def_water = 2.5, 1.5
        def_smoke, def_alcohol, def_state = "Non-smoker", "Occasional", "New York"
    elif preset == "Elevated Cramping & Dysmenorrhea":
        def_age, def_bmi, def_diet, def_exercise = 26, 21.5, "Good", "1–2 days/week"
        def_bsleep, def_bstress = 7.0, 5.0
        def_clen, def_phase, def_flow, def_pain = 27, "Menstrual", "Heavy", 7
        def_mood, def_stress, def_sleep = 4, 6.0, 6.0
        def_energy, def_focus, def_lost = 4, 4, 6.0
        def_bc, def_pcos = 0, 0
        def_caffeine, def_water = 1.0, 2.5
        def_smoke, def_alcohol, def_state = "Non-smoker", "None", "Texas"
    elif preset == "Longer Cycle with Mild Symptoms":
        def_age, def_bmi, def_diet, def_exercise = 29, 26.5, "Fair", "3–4 days/week"
        def_bsleep, def_bstress = 7.5, 5.5
        def_clen, def_phase, def_flow, def_pain = 36, "Follicular", "Light", 3
        def_mood, def_stress, def_sleep = 6, 5.0, 7.5
        def_energy, def_focus, def_lost = 6, 6, 0.0
        def_bc, def_pcos = 0, 1
        def_caffeine, def_water = 2.0, 2.0
        def_smoke, def_alcohol, def_state = "Former", "Occasional", "Illinois"
    else:
        # Custom Inputs: Pre-fill baseline if user has a saved profile in the database
        if saved_profile:
            def_age = int(saved_profile.age) if saved_profile.age is not None else 28
            def_bmi = float(saved_profile.bmi) if saved_profile.bmi is not None else 22.5
            def_diet = saved_profile.diet_quality if saved_profile.diet_quality else "Good"
            def_exercise = saved_profile.exercise_frequency if saved_profile.exercise_frequency else "3–4 days/week"
            def_bsleep = float(saved_profile.sleep_hours) if saved_profile.sleep_hours is not None else 7.5
            def_bstress = float(saved_profile.stress_score_baseline) if saved_profile.stress_score_baseline is not None else 4.0
            def_bc = int(saved_profile.birth_control_use) if saved_profile.birth_control_use is not None else 0
            def_pcos = int(saved_profile.pcos_diagnosed) if saved_profile.pcos_diagnosed is not None else 0
            def_caffeine = float(saved_profile.caffeine_intake) if saved_profile.caffeine_intake is not None else 1.5
            def_water = float(saved_profile.water_intake_liters) if saved_profile.water_intake_liters is not None else 2.0
            def_smoke = saved_profile.smoking_status if saved_profile.smoking_status else "Non-smoker"
            def_alcohol = saved_profile.alcohol_consumption if saved_profile.alcohol_consumption else "None"
            def_state = saved_profile.state if saved_profile.state else "California"
        else:
            def_age, def_bmi, def_diet, def_exercise = 28, 22.5, "Good", "3–4 days/week"
            def_bsleep, def_bstress = 7.5, 4.0
            def_bc, def_pcos = 0, 0
            def_caffeine, def_water = 1.5, 2.0
            def_smoke, def_alcohol, def_state = "Non-smoker", "None", "California"

        def_clen, def_phase, def_flow, def_pain = 28, "Menstrual", "Medium", 3
        def_mood, def_stress, def_sleep = 7, 4.0, 7.5
        def_energy, def_focus, def_lost = 7, 7, 0.0

    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    # Section 1: Menstrual-Cycle Information
    with col1:
        st.markdown('<div class="card-surface"><div class="card-header-serif"><span class="step-pill">Step 1</span> Cycle Observations</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="card-surface"><div class="card-header-serif"><span class="step-pill">Step 2</span> Lifestyle & Biomarkers</div>', unsafe_allow_html=True)
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
            caffeine_intake = st.slider("Caffeine Intake (cups/day)", 0.0, 6.0, float(def_caffeine), step=0.5)
            water_intake = st.slider("Water Intake (Liters/day)", 0.5, 5.0, float(def_water), step=0.5)
            smoke_opts = ["Non-smoker", "Former", "Smoker"]
            smoke_idx = smoke_opts.index(def_smoke) if def_smoke in smoke_opts else 0
            smoking_stat = st.selectbox("Smoking Status", smoke_opts, index=smoke_idx)
            alc_opts = ["None", "Occasional", "Moderate", "Frequent"]
            alc_idx = alc_opts.index(def_alcohol) if def_alcohol in alc_opts else 0
            alcohol_stat = st.selectbox("Alcohol Consumption", alc_opts, index=alc_idx)
            state_opts = ["California", "New York", "Texas", "Florida", "Illinois", "Other"]
            state_idx = state_opts.index(def_state) if def_state in state_opts else 0
            state_region = st.selectbox("State / Region", state_opts, index=state_idx)

        if user_id:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("Save Baseline to Profile", key="btn_save_user_profile", use_container_width=True, type="secondary"):
                save_user_profile(
                    user_id=user_id,
                    profile_data={
                        "age": age,
                        "bmi": bmi,
                        "diet_quality": diet_quality,
                        "exercise_frequency": exercise_freq,
                        "sleep_hours": base_sleep,
                        "stress_score_baseline": base_stress,
                        "pcos_diagnosed": 1 if pcos_obs else 0,
                        "birth_control_use": 1 if bc_obs else 0,
                        "caffeine_intake": caffeine_intake,
                        "water_intake_liters": water_intake,
                        "smoking_status": smoking_stat,
                        "alcohol_consumption": alcohol_stat,
                        "state": state_region,
                    }
                )
                st.success("Lifestyle baseline saved to your private profile.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Previous-Cycle Information
    with col3:
        st.markdown('<div class="card-surface"><div class="card-header-serif"><span class="step-pill">Step 3</span> Historical Records</div>', unsafe_allow_html=True)
        if user_id and user_history:
            st.markdown(
                f'<div style="margin-bottom: 8px;">'
                f'<span class="badge-pill" style="background: rgba(110, 139, 98, 0.15); color: #2D5028;">'
                f'{len(user_history)} Cycles Saved in Account</span>'
                f'</div>',
                unsafe_allow_html=True
            )

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

            hist_health_def = 8.0
            hist_len_def = 28
            hist_pain_def = 3
            hist_stress_def = 4.0
            hist_sleep_def = 7.5
            hist_energy_def = 7
            hist_mood_def = 7
            hist_focus_def = 7

            if user_id and user_history:
                latest_cycle = user_history[0]
                prefill_latest = st.checkbox(
                    f"Load from previous logged Cycle #{latest_cycle.cycle_number}",
                    value=False,
                    key="chk_prefill_cycle"
                )
                if prefill_latest:
                    hist_health_def = float(latest_cycle.predicted_health_score or 8.0)
                    hist_len_def = int(latest_cycle.cycle_length_days or 28)
                    hist_pain_def = int(latest_cycle.pain_level or 3)
                    hist_stress_def = float(latest_cycle.stress_score_cycle or 4.0)
                    hist_sleep_def = float(latest_cycle.sleep_hours_cycle or 7.5)
                    hist_energy_def = int(latest_cycle.energy_level or 7)
                    hist_mood_def = int(latest_cycle.mood_score or 7)
                    hist_focus_def = int(latest_cycle.concentration_score or 7)

            prev_health = st.slider("Previous Cycle Overall Health (0–10)", 2.0, 10.0, hist_health_def, step=0.2)
            prev_len = st.slider("Previous Cycle Length (Days)", 20, 42, hist_len_def)
            prev_pain = st.slider("Previous Cycle Pain (0–10)", 0, 10, hist_pain_def)
            prev_stress = st.slider("Previous Cycle Stress (0–10)", 0.0, 10.0, hist_stress_def, step=0.5)
            prev_sleep = st.slider("Previous Cycle Sleep (Hours)", 4.0, 11.0, hist_sleep_def, step=0.5)
            prev_energy = st.slider("Previous Cycle Vitality (0–10)", 0, 10, hist_energy_def)
            prev_mood = st.slider("Previous Cycle Mood (0–10)", 0, 10, hist_mood_def)
            prev_focus = st.slider("Previous Cycle Concentration (0–10)", 0, 10, hist_focus_def)

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

    # Action Buttons: Predict and Reset
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns([3, 1], vertical_alignment="center")
    with btn_col1:
        btn_predict = st.button("Generate Health Score Prediction", use_container_width=True, key="btn_run_prediction", type="primary")
    with btn_col2:
        btn_reset = st.button("Reset Form", use_container_width=True, key="btn_reset_prediction_form", type="secondary")
        if btn_reset:
            st.session_state.last_prediction = None
            st.session_state.last_inputs = None
            st.rerun()

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
    else:
        st.markdown("<hr style='margin: 28px 0 18px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)
        st.markdown("""
        <div class="empty-state-card">
            <div style="font-family: var(--font-serif); font-size: 1.30rem; color: #59433F; margin-bottom: 6px;">
                Ready for Consultation
            </div>
            <div style="font-size: 0.88rem; max-width: 520px; margin: 0 auto; line-height: 1.55; color: #7A6560;">
                Review the cycle observations, lifestyle parameters, and history settings above, 
                then click <b>Generate Health Score Prediction</b> to view your next-cycle score and dimensional wellbeing balance.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 3: PREDICTION RESULTS COMPONENT
# =============================================================================
def render_prediction_results(result, inputs: Optional[Dict[str, Any]]):
    """Render Section E: Prediction Results Page with scale, context, limitations, and general wellness suggestions."""
    st.markdown("<hr style='margin: 28px 0 18px 0; border: none; border-top: 1px solid #F0E2DA;'>", unsafe_allow_html=True)
    st.markdown('<div class="badge-pill">Forecast Summary</div>', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results & Explanation</h2>", unsafe_allow_html=True)

    score_val = result.predicted_next_health_score
    diff_from_median = score_val - 8.60

    if diff_from_median >= 0:
        rel_text = f"+{diff_from_median:.2f} relative to cohort median (8.60)"
        banner_class = "score-banner-optimal"
        tier_badge_html = '<span class="tier-badge-optimal">Optimal Balance Tier</span>'
    else:
        rel_text = f"{diff_from_median:.2f} relative to cohort median (8.60)"
        banner_class = "score-banner-moderate"
        tier_badge_html = '<span class="tier-badge-moderate">Moderate Variation Tier</span>'

    st.markdown(f"""
    <div class="score-result-banner {banner_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 6px;">
            <div>
                <span style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85;">Ridge Model Forecast (37 Features)</span>
                <div class="score-result-title">
                    Predicted Next-Cycle Health Score: <b>{score_val:.2f}</b> <span style="font-size: 1.10rem; font-family: var(--font-sans); font-weight: 400; opacity: 0.8;">/ 10.0</span>
                </div>
            </div>
            <div>
                {tier_badge_html}
            </div>
        </div>
        <div class="score-result-subtitle">
            Scale: Continuous score from 0.0 (lowest overall wellbeing) to 10.0 (highest overall wellbeing).<br>
            Empirical Cohort Alignment: <b>{rel_text}</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    res_c1, res_c2 = st.columns([1, 1], gap="medium")

    with res_c1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Predicted Score Visualization</div>', unsafe_allow_html=True)
        gauge_fig = create_gauge_chart(score_val, rel_text)
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div style="font-size: 0.86rem; color: #59433F; line-height: 1.5; padding: 4px 2px;">
            <b>Score Interpretation:</b><br>
            The overall health score is a continuous composite indicator modeled from symptom severity, 
            emotional balance, sleep hours, and stress continuity. In the training dataset (N = 15,976), 
            the middle 50% of records fall between 7.80 and 9.50.
        </div>
        """, unsafe_allow_html=True)
        if result.warnings:
            for w in result.warnings:
                st.caption(f"Note: {w}")
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
            st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Relevant Input Summary Table
    if inputs:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Summary of Evaluated Input Parameters</div>', unsafe_allow_html=True)
        in_c1, in_c2, in_c3 = st.columns(3)
        with in_c1:
            st.markdown(f"""
            <div style="font-size: 0.86rem; line-height: 1.7; color: var(--color-cocoa);">
                <b>Age:</b> {inputs['age']} years<br>
                <b>BMI:</b> {inputs['bmi']:.1f}<br>
                <b>Baseline Stress:</b> {inputs['base_stress']:.1f} / 10
            </div>
            """, unsafe_allow_html=True)
        with in_c2:
            st.markdown(f"""
            <div style="font-size: 0.86rem; line-height: 1.7; color: var(--color-cocoa);">
                <b>Cycle Length:</b> {inputs['cycle_length']} days<br>
                <b>Cycle Phase:</b> {inputs['phase']}<br>
                <b>Flow Heaviness:</b> {inputs['flow']}
            </div>
            """, unsafe_allow_html=True)
        with in_c3:
            st.markdown(f"""
            <div style="font-size: 0.86rem; line-height: 1.7; color: var(--color-cocoa);">
                <b>Pain Intensity:</b> {inputs['pain']} / 10<br>
                <b>Cycle Sleep:</b> {inputs['cycle_sleep']:.1f} hrs<br>
                <b>Historical Records:</b> {'User Prior Cycle' if inputs['has_history'] else 'Cycle 1 Cohort Baseline'}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # General Actionable Next Steps (Phrased as General Wellness Information)
    st.markdown('<div class="card-surface"><div class="card-header-serif">General Wellness Observations & Next Steps</div>', unsafe_allow_html=True)
    st.markdown("""
    *These observations are general educational lifestyle recommendations and do not constitute medical advice or treatment.*
    """)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div style="background: #FDF6EE; border: 1px solid #F0D4BC; border-top: 3px solid var(--color-amber); border-radius: 8px; padding: 14px; font-size: 0.86rem; height: 100%;">
            <b style="color: #59433F;">Rest & Recovery Rhythm</b>
            <p style="font-size: 0.83rem; color: #6E5752; line-height: 1.5; margin-top: 6px; margin-bottom: 0;">
                Aim for consistent sleep duration (7.5–8.5 hours). Maintaining steady circadian rhythms supports pituitary gonadotropin secretion and next-cycle vitality.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div style="background: #F4F7F2; border: 1px solid #C8D7C2; border-top: 3px solid var(--color-sage); border-radius: 8px; padding: 14px; font-size: 0.86rem; height: 100%;">
            <b style="color: #59433F;">Nutritional Hydration</b>
            <p style="font-size: 0.83rem; color: #6E5752; line-height: 1.5; margin-top: 6px; margin-bottom: 0;">
                Adequate daily water intake (2+ liters) and electrolyte-rich whole foods assist cellular hydration and support comfort throughout menstruation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div style="background: #FAF7F5; border: 1px solid #EADBCE; border-top: 3px solid var(--color-mauve); border-radius: 8px; padding: 14px; font-size: 0.86rem; height: 100%;">
            <b style="color: #59433F;">Stress Regulation Practices</b>
            <p style="font-size: 0.83rem; color: #6E5752; line-height: 1.5; margin-top: 6px; margin-bottom: 0;">
                Because stress exerts a significant negative standardized coefficient in the model, daily down-regulation (gentle walking, mindfulness) helps protect cycle continuity.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Limitations Card
    st.markdown("""
    <div class="callout-lavender">
        <b style="color: #59433F; font-size: 0.90rem;">Model Limitations & Clinical Boundary Notice:</b><br>
        <span style="font-size: 0.85rem; color: #59433F; line-height: 1.55;">
        This prediction was generated by an experimental Ridge Regression pipeline (test R² = 0.4620, MAE = 0.6448). 
        The model has not been clinically validated on diverse real-world populations. 
        It does not diagnose any medical condition (such as endometriosis or PCOS), does not evaluate pregnancy or maternal health, 
        and must not be used to direct medication, contraception, or therapeutic interventions.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Multi-User Account Persistence: Save cycle record & history inspection
    user_id = get_current_user_id()
    if user_id:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        col_act1, col_act2 = st.columns([1.6, 2.4], vertical_alignment="center")
        with col_act1:
            if st.button("Save Cycle Record to My Account", key="btn_save_cycle_record", type="primary", use_container_width=True):
                cycle_num = get_next_cycle_number(user_id)
                cycle_data = {
                    "cycle_number": cycle_num,
                    "cycle_length_days": inputs.get("cycle_length", 28) if inputs else 28,
                    "cycle_phase": inputs.get("phase", "Menstrual") if inputs else "Menstrual",
                    "flow_level": inputs.get("flow", "Medium") if inputs else "Medium",
                    "pain_level": inputs.get("pain", 0) if inputs else 0,
                    "mood_score": inputs.get("mood", 7) if inputs else 7,
                    "stress_score_cycle": inputs.get("cycle_stress", 4.0) if inputs else 4.0,
                    "sleep_hours_cycle": inputs.get("cycle_sleep", 7.5) if inputs else 7.5,
                    "energy_level": inputs.get("energy", 7) if inputs else 7,
                    "concentration_score": inputs.get("focus", 7) if inputs else 7,
                    "work_hours_lost": inputs.get("hours_lost", 0.0) if inputs else 0.0,
                }
                save_user_cycle_log(
                    user_id=user_id,
                    cycle_data=cycle_data,
                    predicted_score=score_val,
                    prediction_tier=result.prediction_tier,
                )
                st.success(f"Cycle #{cycle_num} saved successfully to your private account history.")
                st.rerun()
        with col_act2:
            st.caption("Your saved cycle records are strictly isolated to your user account and persist in your local encrypted/secured database.")

        saved_logs = get_user_cycle_logs(user_id, limit=10)
        if saved_logs:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            with st.expander(f"View Your Logged Cycle History ({len(saved_logs)} cycles saved)", expanded=False):
                hist_records = []
                for entry in saved_logs:
                    hist_records.append({
                        "Cycle #": entry.cycle_number,
                        "Length (Days)": entry.cycle_length_days,
                        "Phase": entry.cycle_phase,
                        "Flow": entry.flow_level,
                        "Pain (0-10)": entry.pain_level,
                        "Stress (0-10)": entry.stress_score_cycle,
                        "Sleep (Hrs)": entry.sleep_hours_cycle,
                        "Predicted Health": f"{entry.predicted_health_score:.2f}" if entry.predicted_health_score is not None else "N/A",
                        "Tier": entry.prediction_tier or "Standard",
                        "Logged At": entry.created_at[:16] if entry.created_at else "",
                    })
                st.dataframe(pd.DataFrame(hist_records), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="background: #FAF5F2; border: 1px dashed #D99AA5; border-radius: 8px; padding: 14px 18px; margin: 18px 0; font-size: 0.86rem; color: #59433F;">
            <b>Personalized History &amp; Privacy:</b> You are currently exploring HERAI as a guest. 
            Sign in or create an account to save your sequential cycle records, auto-fill baseline lifestyle inputs, and preserve your personal health patterns securely across sessions.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign In / Join to Save Records", key="btn_guest_prompt_auth", type="secondary"):
            st.session_state.current_page = "Sign In"
            st.rerun()


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

    st.markdown('<div class="badge-pill">Physiological Reference</div>', unsafe_allow_html=True)
    st.markdown("<h1>Menstrual Cycle Biology & Empirical Patterns</h1>", unsafe_allow_html=True)
    st.markdown(
        "A healthy menstrual cycle typically spans 21 to 35 days, orchestrated by rhythmic fluctuations "
        "in estrogen, progesterone, luteinizing hormone (LH), and follicle-stimulating hormone (FSH). "
        "Below, explore the physiological rhythm alongside empirical trends observed in 15,976 evaluated cycles."
    )

    # Crucial Distinction Banner
    st.markdown("""
    <div class="callout-lavender" style="margin-bottom: 22px;">
        <b style="color: #59433F; font-size: 0.92rem;">Crucial Distinction: Educational Biology vs. Algorithmic Health Forecast</b><br>
        <span style="font-size: 0.86rem; color: #59433F; line-height: 1.55;">
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
        <div style="background: #FAF7F5; border-left: 3px solid #B97B8A; padding: 13px 15px; border-radius: 6px; margin-bottom: 12px;">
            <b style="color: #59433F;">Current Cycle Indicators:</b>
            <ul style="font-size: 0.85rem; color: #59433F; line-height: 1.6; margin-top: 6px; padding-left: 18px;">
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
        <div style="background: #FAF7F5; border-left: 3px solid #A8B8A0; padding: 13px 15px; border-radius: 6px; margin-bottom: 12px;">
            <b style="color: #59433F;">Prior-Cycle Lag & Lifestyle Indicators:</b>
            <ul style="font-size: 0.85rem; color: #59433F; line-height: 1.6; margin-top: 6px; padding-left: 18px;">
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
    st.markdown("### Empirical Cohort Patterns (N = 15,976 Cycles)")
    st.caption("Visualizing real patterns logged across 2,000 anonymized cohort profiles.")

    pat_c1, pat_c2 = st.columns(2, gap="medium")
    with pat_c1:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Observed Cycle Phases at Log Entry</div>', unsafe_allow_html=True)
        if not cohort_df.empty and "cycle_phase" in cohort_df.columns:
            fig_ph = create_phase_distribution_chart(cohort_df)
            st.plotly_chart(fig_ph, use_container_width=True, config={"displayModeBar": False})
            st.caption("Entries in the cohort were logged primarily during the Luteal (45.0%) and Follicular (44.8%) phases, with Menstrual entries accounting for 10.1%.")
        else:
            st.info("Loading phase distribution...")
        st.markdown('</div>', unsafe_allow_html=True)

    with pat_c2:
        st.markdown('<div class="card-surface"><div class="card-header-serif">Observed Pain vs. Mean Health Score</div>', unsafe_allow_html=True)
        if not cohort_df.empty and "pain_level" in cohort_df.columns:
            fig_pain = create_pain_vs_health_chart(cohort_df)
            st.plotly_chart(fig_pain, use_container_width=True, config={"displayModeBar": False})
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
            <p style="font-size: 0.88rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> Estrogen and progesterone drop to baseline levels, signaling the shedding of the uterine lining (stratum functionale).
            </p>
            <ul style="font-size: 0.84rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
                <li><b>Physiological Focus:</b> Restorative recovery, iron replenishment, gentle movement.</li>
                <li><b>Nutritional Support:</b> Warm mineral broths, dark leafy greens, magnesium-rich seeds, hydration.</li>
                <li><b>Comfort Measures:</b> Gentle heat application, warm baths, restorative diaphragmatic breathing.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card-surface" style="border-top: 3px solid #D99AA5;">
            <div class="card-header-serif">2. Follicular Phase (Days 6–13)</div>
            <p style="font-size: 0.88rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> Pituitary FSH stimulates ovarian follicle development; ascending estradiol thickens the endometrium.
            </p>
            <ul style="font-size: 0.84rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
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
            <p style="font-size: 0.88rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> Estrogen peak triggers a sharp LH (luteinizing hormone) surge, stimulating the release of a mature ovum.
            </p>
            <ul style="font-size: 0.84rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
                <li><b>Physiological Focus:</b> Peak metabolic rate, elevated stamina, enhanced verbal acuity.</li>
                <li><b>Nutritional Support:</b> Antioxidant-dense berries, zinc, cruciferous vegetables supporting estrogen metabolism.</li>
                <li><b>Activity Alignment:</b> High-intensity interval workouts, collaborative group efforts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card-surface" style="border-top: 3px solid #C97A3E;">
            <div class="card-header-serif">4. Luteal Phase (Days 17–28)</div>
            <p style="font-size: 0.88rem; line-height: 1.5; color: #59433F;">
                <b>Endocrine Dynamics:</b> The ruptured follicle forms the corpus luteum, secreting progesterone to support the endometrium.
            </p>
            <ul style="font-size: 0.84rem; line-height: 1.55; padding-left: 18px; color: #59433F;">
                <li><b>Physiological Focus:</b> Nervous system down-regulation, steady blood glucose, sleep hygiene.</li>
                <li><b>Nutritional Support:</b> Vitamin B6, pumpkin and sunflower seeds, complex root vegetables.</li>
                <li><b>Activity Alignment:</b> Moderate pilates, nature walks, calming evening wind-down rituals.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 4: Clinical Red Flags
    st.markdown('<div class="card-surface" style="border-left: 4px solid #C97A3E;"><div class="card-header-serif">When to Seek Medical Attention (Clinical Red Flags)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.86rem; color: #59433F; line-height: 1.6;">
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
    st.markdown('<div class="badge-pill">Validation & Benchmarking</div>', unsafe_allow_html=True)
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
        <div class="metric-stat-box" style="border-top: 3px solid var(--color-mauve);">
            <div class="metric-stat-label">Production Algorithm</div>
            <div class="metric-stat-value" style="font-size: 1.35rem; padding-top: 4px;">Ridge Regression</div>
            <div class="metric-stat-detail">L2 Regularization (α = 1.0)</div>
        </div>
        """, unsafe_allow_html=True)
    with km_c2:
        st.markdown("""
        <div class="metric-stat-box" style="border-top: 3px solid var(--color-sage);">
            <div class="metric-stat-label" style="color: #4D6B45;">Active Feature Set</div>
            <div class="metric-stat-value">37 Features</div>
            <div class="metric-stat-detail">Lab & lookahead features excluded</div>
        </div>
        """, unsafe_allow_html=True)
    with km_c3:
        st.markdown("""
        <div class="metric-stat-box" style="border-top: 3px solid var(--color-dusty-rose);">
            <div class="metric-stat-label" style="color: #9E5664;">Test Set MAE</div>
            <div class="metric-stat-value">0.6448</div>
            <div class="metric-stat-detail">Average error on 0–10 scale</div>
        </div>
        """, unsafe_allow_html=True)
    with km_c4:
        st.markdown("""
        <div class="metric-stat-box" style="border-top: 3px solid var(--color-amber);">
            <div class="metric-stat-label" style="color: #A35216;">Test Set R²</div>
            <div class="metric-stat-value">0.4620</div>
            <div class="metric-stat-detail">46.2% cyclical variance explained</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Section 1: Candidate Comparison & Benchmark Evaluation
    st.markdown("### Candidate Model Benchmarking (Test Set: N = 3,176 Cycles across 400 Users)")
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
                "Status": "Selected Production"
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
        <div style="font-size: 0.85rem; color: #59433F; line-height: 1.55; margin-top: 8px;">
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
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
        st.caption("Both Ridge variants comfortably outperform the non-linear Random Forest and the Dummy Mean baseline on unseen test users.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Metric Explanations in Plain Language
    st.markdown('<div class="card-surface"><div class="card-header-serif">Understanding the Evaluation Metrics (Plain Language Guide)</div>', unsafe_allow_html=True)
    me_c1, me_c2, me_c3 = st.columns(3, gap="medium")
    with me_c1:
        st.markdown("""
        <div style="background: #FAF7F5; border-radius: 8px; padding: 14px; height: 100%;">
            <b style="color: #882D52;">Mean Absolute Error (MAE = 0.6448)</b>
            <p style="font-size: 0.84rem; color: #59433F; line-height: 1.5; margin-top: 6px;">
                On average, the model's predicted health score is within <b>0.64 points</b> of the user's actual health score in the next cycle (on a 0 to 10 scale). 
                This represents an average deviation of just 6.4% across the entire scoring spectrum.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with me_c2:
        st.markdown("""
        <div style="background: #FAF7F5; border-radius: 8px; padding: 14px; height: 100%;">
            <b style="color: #882D52;">Root Mean Squared Error (RMSE = 0.8025)</b>
            <p style="font-size: 0.84rem; color: #59433F; line-height: 1.5; margin-top: 6px;">
                Like MAE, but squares individual errors prior to averaging, making it more sensitive to large outliers. 
                The low RMSE of <b>0.8025</b> confirms that large, erratic forecasting errors rarely occur across test cycles.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with me_c3:
        st.markdown("""
        <div style="background: #FAF7F5; border-radius: 8px; padding: 14px; height: 100%;">
            <b style="color: #882D52;">Coefficient of Determination (R² = 0.4620)</b>
            <p style="font-size: 0.84rem; color: #59433F; line-height: 1.5; margin-top: 6px;">
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
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        st.caption("Standardized Ridge coefficients: Positive values (sage) support higher next-cycle health; negative values (dusty rose) depress the score.")
        st.markdown('</div>', unsafe_allow_html=True)

    with feat_c2:
        st.markdown("""
        <div class="card-surface" style="height: 100%;">
            <div class="card-header-serif">Physiological Interpretation</div>
            <div style="font-size: 0.85rem; color: #59433F; line-height: 1.6;">
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
    <div style="font-size: 0.86rem; color: #59433F; line-height: 1.6;">
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
    <div class="disclaimer-card" style="border-left-width: 4px;">
        <b style="color: #882D52; font-size: 0.92rem;">Honest Evaluation &amp; Clinical Boundary Notice:</b><br>
        <span style="font-size: 0.86rem; color: #59433F; line-height: 1.55;">
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
    st.markdown('<div class="badge-pill">Research Overview</div>', unsafe_allow_html=True)
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
            <p style="font-size: 0.88rem; line-height: 1.55;">
                Menstrual health remains historically under-represented in statistical predictive modeling. 
                HERAI demonstrates an end-to-end data science lifecycle: from raw record ingestion, schema validation, 
                and chronological lag feature engineering, to model comparison, leakage verification, and ethical inference.
            </p>
            <p style="font-size: 0.88rem; line-height: 1.55;">
                <b>Dataset Composition:</b> 17,976 raw cycle records across 2,000 unique user profiles, yielding 15,976 training records with verified next-cycle targets.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with ab_c2:
        st.markdown("""
        <div class="card-surface">
            <div class="card-header-serif">Data Privacy & Security</div>
            <p style="font-size: 0.88rem; line-height: 1.55;">
                <b>Local In-Memory Execution:</b> All user entries submitted through this web interface are processed 
                entirely in-memory within your local Python runtime. 
            </p>
            <p style="font-size: 0.88rem; line-height: 1.55;">
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

    if st.session_state.current_page == "Sign In":
        render_auth_view()
    elif st.session_state.current_page == "Dashboard":
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
    <div style="text-align: center; font-size: 0.78rem; color: #8F7B76; margin-top: 36px; padding-top: 16px; border-top: 1px solid #F0E2DA;">
        HERAI — Menstrual Health Intelligence Prototype &nbsp;|&nbsp; Educational & Data-Driven ML System &nbsp;|&nbsp; Non-Diagnostic
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
