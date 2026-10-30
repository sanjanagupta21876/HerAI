# HERAI — Menstrual Health Intelligence & Prediction System

HERAI is a personalized, privacy-preserving women's wellness application and machine learning research prototype. It combines observational menstrual cycle tracking, demographic and lifestyle factors, and sequential cycle patterns to forecast an overall health score for the upcoming cycle using a validated 37-feature Ridge Regression pipeline.

The application features an interactive, multi-user web interface with secure local authentication, persistent relational storage, personalized onboarding, clinical baseline validation, dynamic cycle tracking, longitudinal trajectory charts, and population cohort benchmarks.

> **Clinical & Legal Disclaimer:** HERAI is an educational demonstration and research prototype. It is not a medical device, diagnostic tool, clinical decision-support system, or contraceptive planner. Outputs represent statistical predictions derived from self-reported observational data patterns and must not be used for diagnosis, medical treatment, or contraception planning.

---

## 1. Key Application Capabilities

### 1.1 Secure Multi-User Authentication & Data Isolation
- **Encrypted Credentials:** Salted password hashing using industry-standard `bcrypt` (`bcrypt.gensalt(12)`).
- **Relational Storage:** SQLite database with Write-Ahead Logging (`PRAGMA journal_mode = WAL;`) and strict foreign key cascading (`PRAGMA foreign_keys = ON;`).
- **SQL Injection Defense:** 100% of database queries use parameterized SQL.
- **Tenant Isolation:** Every profile read, cycle log query, and update is scoped strictly by the authenticated `user_id`. One user can never inspect, alter, or leak another user's records.

### 1.2 First-Time User Onboarding & Clinical Profile Management
- **Progressive Disclosure:** 4-step guided onboarding wizard (Basic Profile, Menstrual Cycle Baseline, Wellness Focus, Review & Confirm).
- **Flexible & Non-Blocking:** Users may choose not to disclose dates or skip cycle tracking without blocking access.
- **Clinical Bounds Validation:** Automated input validation enforcing physiological ranges (age 15–100, cycle length 18–60 days, period duration 1–15 days, past dates within 365 days).
- **Full Profile Management:** View baseline biomarkers, toggle edit mode with prepopulated controls, and perform non-destructive updates that preserve unedited fields.

### 1.3 Personalized Dashboard & Longitudinal Analytics
- **Dynamic Cycle Tracking:** Calculates real-time cycle day and physiological phase (Menstrual, Follicular, Ovulatory, Luteal, or Late) from the user's saved last period date and typical cycle length.
- **Truthful & Grounded Metrics:** Summary cards display actual saved user data (observed cycle length, logged cycles, latest health score and tier, baseline sleep and hydration).
- **Reassuring Empty States:** First-time users receive clear, actionable onboarding guidance rather than broken or fake statistics.
- **Longitudinal Health Trajectory:** Interactive Plotly visualization plotting the user's sequential health forecasts with hover details (phase, pain, sleep, stress) alongside the research cohort mean benchmark (8.55).
- **Biomarker Consistency Chart:** Multi-cycle trend analysis tracking sleep duration, subjective stress, and discomfort across logged cycles.
- **Population Cohort Benchmarks:** Separate reference tab displaying empirical distributions across 15,976 research dataset records.

### 1.4 Ethical 37-Feature Machine Learning Inference
- **Zero Stranger-Data Substitution:** HERAI never borrows or copies records from other individuals in the dataset. For first-time users, learned cohort medians from the training split serve as principled baseline defaults for lag features.
- **Audited Feature Set:** 37 strictly observable home-accessible variables. Excludes lookahead features and home-inaccessible lab metrics (estrogen, progesterone, ovulation test kits).
- **Input Bounds Enforcement:** Strict validation against physiological limits before inference.

---

## 2. Technology Stack

- **Language:** Python (3.9+)
- **Frontend Framework:** Streamlit (1.30.0+)
- **Visualizations:** Plotly (5.15.0+)
- **Machine Learning & Data Processing:** Scikit-learn (1.4.0+), Pandas (2.0.0+), NumPy (1.24.0+), Joblib (1.3.0+)
- **Security & Authentication:** Bcrypt (4.0.0+)
- **Database:** SQLite 3 (WAL mode, Foreign Keys enabled)
- **Data Schemas & Validation:** Pydantic (2.5.0+)
- **Automated Testing:** Pytest (7.0.0+)

---

## 3. Project Structure

```text
HERAI/
├── app.py                          # Streamlit application: router, auth, onboarding, dashboard, ML UI
├── data/
│   ├── raw/
│   │   ├── Period_Log.csv          # 17,976 raw menstrual cycle records
│   │   └── User_Profile.csv        # 2,000 raw user demographic & lifestyle records
│   └── processed/
│       ├── merged_data.csv         # Validated many-to-one merged dataset
│       ├── engineered_data.csv     # Chronological lag features and aligned target
│       └── training_data.csv       # Filtered training dataset (15,976 records)
├── models/
│   ├── herai_ridge_model.pkl       # 42-feature Ridge pipeline
│   └── herai_ridge_37_features.pkl # 37-feature Ridge production model
├── src/
│   ├── herai/                      # Core reusable package
│   │   ├── __init__.py
│   │   ├── auth.py                 # User authentication, registration, session state
│   │   ├── config.py               # Paths, feature sets, validation bounds
│   │   ├── data.py                 # Schema validation, merging, lag generation
│   │   ├── database.py             # SQLite WAL connection, migrations, profiles, cycle logs
│   │   ├── inference.py            # Ethical inference engine, validation, PredictionOutput
│   │   ├── models.py               # Training pipelines, user-grouped splits, evaluation
│   │   └── visualizations.py       # Plotly charts (personal trajectory, biomarkers, cohort)
│   ├── 01_inspect_data.py          # Data inspection & empirical formula audit
│   ├── 02_merge_data.py            # Schema validation and dataset merge
│   ├── 03_create_features.py       # Chronological sort and lag feature creation
│   ├── 04_prepare_data.py          # Training dataset preparation
│   ├── 05_train_model.py           # Trains 42-feature Ridge model
│   ├── 06_compare_models.py        # Compares Ridge vs Random Forest vs 37-feature Ridge
│   ├── 06_predict.py               # Programmatic prediction demonstration
│   ├── 07_user_input.py            # Interactive CLI prediction
│   ├── 08_feature_leakage_audit.py # Target leakage verification
│   └── inspect_features.py         # Feature timing exploration
├── tests/
│   ├── test_auth_and_db.py         # Multi-user auth, database, onboarding, profile, dashboard tests
│   ├── test_herai.py               # Core ML pipeline & inference unit tests
│   ├── test_qa_pipeline.py         # QA, presets, bounds & script tests
│   └── test_ui.py                  # Visualization & UI component unit tests
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git exclusion rules
├── pyproject.toml                  # Package configuration & pytest settings
├── requirements.txt                # Dependency manifest
└── README.md
```

---

## 4. Model Benchmarks & Validation

All models were evaluated on the same 80/20 user-grouped split (`GroupShuffleSplit`, 1,600 training users, 400 test users):

| Model | MAE | RMSE | R² Score | Description |
|---|---:|---:|---:|---|
| **Ridge Without Audited Features (37 features)** | **0.6448** | **0.8025** | **0.4620** | **Selected production model** (excludes lookahead & lab features) |
| Ridge Regression (42 features) | 0.6452 | 0.8029 | 0.4616 | Includes all features (some lab/inaccessible at home) |
| Random Forest (42 features) | 0.6550 | 0.8130 | 0.4480 | Non-linear tree ensemble |
| Dummy Mean Baseline | 0.9113 | 1.0942 | -0.0000 | Null baseline predicting training target mean |

---

## 5. Installation and Setup

### 5.1 Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/HerAI-1.git
   cd HerAI-1
   ```

2. **Create and activate a virtual environment (Python 3.9+):**
   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 5.2 Environment Configuration (Optional)

Copy `.env.example` to `.env` if you wish to override the default database path or Streamlit server settings:
```bash
cp .env.example .env
```
Default SQLite database location: `data/herai.db`.

### 5.3 Database Initialization

Database schemas and indexes initialize automatically upon application startup via `init_db()`. Existing databases automatically apply backwards-compatible migrations without data loss.

---

## 6. How to Run the Application

Launch the interactive Streamlit web application:

```bash
streamlit run app.py
```

### 6.1 Application Walkthrough

1. **Sign Up & Onboarding:**
   - Click **Sign In / Join** in the top navigation header.
   - Select **Create Account** and register with a unique username, email, and secure password (minimum 8 characters with at least one number).
   - Upon first login, complete the 4-step Onboarding flow (Preferred Name, Menstrual Cycle Baseline, Wellness Interests, and Confirmation).
2. **Explore Your Personalized Dashboard:**
   - View your personalized greeting, today's date, live cycle day, and estimated menstrual cycle phase.
   - Inspect your private metrics: Cycle Baseline, Logged Cycles count, Latest Health Score, and Lifestyle Baselines.
   - Switch between **My Personal Health Journey**, **Research Cohort Benchmarks**, and **Personalized Wellness Guidance**.
3. **Health Score Prediction & Consultation:**
   - Navigate to **Health Prediction** via the top navigation bar or the dashboard CTA.
   - Your saved profile values (age, BMI, baseline sleep, water intake, cycle length) pre-populate the form automatically.
   - Adjust cycle parameters (pain level, flow heaviness, cycle stress, sleep duration).
   - Click **Generate Health Score Prediction** to execute the 37-feature Ridge pipeline.
   - Inspect the radial score gauge, wellbeing radar balance across 6 dimensions, and model explanation.
   - Click **Save Cycle Record to My Account** to persist the prediction and update your personal longitudinal trajectory.
4. **Manage Profile:**
   - Click **My Profile** in the top right header to review your account details and baseline parameters.
   - Click **Edit Profile Information** to update your preferred name, cycle lengths, or wellness interests.
   - Use **Save Profile Changes** or **Cancel Editing** safely without losing state.
5. **Sign Out:**
   - Click **Sign Out** in the top header. Your session is securely cleared, and you are safely redirected to the public guest dashboard.

---

## 7. Running the ML Pipeline Scripts (Optional)

To reproduce data preparation and model training from scratch:

```bash
# 1. Inspect raw data schemas and correlations
python src/01_inspect_data.py

# 2. Validate schemas and merge raw datasets
python src/02_merge_data.py

# 3. Chronologically sort and engineer lag features
python src/03_create_features.py

# 4. Prepare training dataset
python src/04_prepare_data.py

# 5. Train Ridge baseline model
python src/05_train_model.py

# 6. Verify zero target leakage
python src/08_feature_leakage_audit.py

# 7. Compare models and save selected 37-feature Ridge model
python src/06_compare_models.py

# 8. Run sample prediction demonstration
python src/06_predict.py

# 9. Run interactive CLI prediction
python src/07_user_input.py
```

---

## 8. Automated Testing

HERAI includes a comprehensive 53-test automated test suite covering all functional, architectural, and security requirements:

```bash
pytest -v
```

### Test Suite Breakdown:
- `tests/test_auth_and_db.py` (29 tests): Password hashing, username/email validation, duplicate registration rejection, authentication, session state, multi-tenant database isolation, profile validation and partial updating, cycle log persistence, dashboard statistics, dynamic cycle day and phase calculation, dual-access `UserCycleLogRecord`, personal Plotly charts, and full end-to-end user lifecycle integration.
- `tests/test_herai.py` (6 tests): Raw data loading, duplicate key detection, chronological ordering, lag feature engineering, user-grouped split isolation, and bounds validation.
- `tests/test_qa_pipeline.py` (9 tests): Model artifact integrity, feature order alignment, preset scenario prediction, first-time user Cycle 1 fallback, returning user history, input validation, Plotly figure generation, empty states, and CLI script safety.
- `tests/test_ui.py` (9 tests): Visualizations unit tests (cycle wave hero, radial gauge, radar profile, cohort distribution, cycle length histogram, user trajectory, feature importances, model comparison, phase distribution, pain vs. health).

---

## 9. Current Limitations & Known Scope Boundaries

- **Informational Prototype:** HERAI is an educational research system designed to evaluate predictive modeling on self-reported observational menstrual records.
- **Non-Diagnostic:** The model does not diagnose PCOS, endometriosis, hormonal disorders, or any medical condition.
- **Fertility & Pregnancy:** HERAI does not predict ovulation dates, fertile windows, pregnancy probability, or maternal health outcomes.
- **Continuous Score Scale:** Predictions estimate an overall continuous health score (1.0 to 10.0 scale) based on self-reported patterns; they do not dictate clinical care or therapeutic interventions.
