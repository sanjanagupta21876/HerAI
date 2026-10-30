"""Automated test suite for HERAI authentication, database storage, and cross-user data isolation."""

import os
from pathlib import Path
import pytest
import sqlite3

from herai.database import (
    init_db,
    get_db_connection,
    create_user,
    get_user_by_id,
    get_user_by_username,
    get_user_by_email,
    get_user_by_identifier,
    update_user_last_login,
    save_user_profile,
    get_user_profile,
    save_user_cycle_log,
    get_user_cycle_logs,
    get_next_cycle_number,
    delete_user_account,
    UserAlreadyExistsError,
    UserNotFoundError,
    OwnershipError,
    ValidationError as DBValidationError,
)
from herai.auth import (
    hash_password,
    verify_password,
    validate_username,
    validate_email,
    validate_password_strength,
    register_user,
    authenticate,
    AuthenticationError,
    ValidationError as AuthValidationError,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, ephemeral SQLite database for testing."""
    db_file = tmp_path / "test_herai.db"
    init_db(db_file)
    return db_file


# =============================================================================
# 1. CRYPTOGRAPHIC PASSWORD HASHING
# =============================================================================
def test_password_hashing_and_verification():
    """Verify that bcrypt hashes are salted, unique, and verify accurately."""
    pwd = "SecurePassword123"
    hash1 = hash_password(pwd)
    hash2 = hash_password(pwd)

    # Hashes must differ because of unique salts
    assert hash1 != hash2
    assert hash1.startswith("$2b$") or hash1.startswith("$2a$")

    # Positive verification
    assert verify_password(pwd, hash1) is True
    assert verify_password(pwd, hash2) is True

    # Negative verification
    assert verify_password("WrongPassword123", hash1) is False
    assert verify_password("", hash1) is False
    assert verify_password(pwd, "invalid_hash_string") is False


def test_password_hashing_rejects_too_short():
    """Verify that hash_password rejects passwords under 8 characters."""
    with pytest.raises(DBValidationError):
        hash_password("short1")


# =============================================================================
# 2. INPUT VALIDATION TESTS
# =============================================================================
def test_username_validation():
    """Verify strict username length and format constraints."""
    valid, user = validate_username("sarah_connor")
    assert valid is True
    assert user == "sarah_connor"

    valid_caps, user_caps = validate_username("  Maya99  ")
    assert valid_caps is True
    assert user_caps == "maya99"

    # Too short
    valid_short, _ = validate_username("ab")
    assert valid_short is False

    # Invalid characters (spaces, special symbols)
    valid_space, _ = validate_username("user name")
    assert valid_space is False

    valid_sym, _ = validate_username("user@name!")
    assert valid_sym is False


def test_email_validation():
    """Verify email format validation and lowercase normalization."""
    valid, email = validate_email("  User.Test@domain.com  ")
    assert valid is True
    assert email == "user.test@domain.com"

    # Invalid emails
    assert validate_email("missing_at_symbol.com")[0] is False
    assert validate_email("missing_domain@")[0] is False
    assert validate_email("@missing_username.com")[0] is False
    assert validate_email("")[0] is False


def test_password_complexity_validation():
    """Verify password strength validation rules."""
    assert validate_password_strength("validPass123")[0] is True
    # Too short
    assert validate_password_strength("short1")[0] is False
    # No numbers
    assert validate_password_strength("alllettersonly")[0] is False
    # No letters
    assert validate_password_strength("1234567890")[0] is False


# =============================================================================
# 3. USER REGISTRATION & INTEGRITY CONSTRAINTS
# =============================================================================
def test_user_registration_success(temp_db):
    """Verify that user registration succeeds and persists in SQLite."""
    user = register_user(
        username="elena_rostova",
        email="elena@example.com",
        password="MySecretPassword88",
        confirm_password="MySecretPassword88",
        full_name="Elena Rostova",
        db_path=temp_db
    )

    assert user.id.startswith("usr_")
    assert user.username == "elena_rostova"
    assert user.email == "elena@example.com"
    assert user.full_name == "Elena Rostova"
    assert user.is_active is True
    assert user.password_hash != "MySecretPassword88"

    # Verify retrieval
    retrieved = get_user_by_id(user.id, db_path=temp_db)
    assert retrieved is not None
    assert retrieved.id == user.id
    assert retrieved.username == "elena_rostova"


def test_registration_password_mismatch(temp_db):
    """Verify registration fails when confirmation password does not match."""
    with pytest.raises(AuthValidationError, match="Passwords do not match"):
        register_user(
            username="testuser1",
            email="test1@example.com",
            password="Password123",
            confirm_password="DifferentPassword456",
            db_path=temp_db
        )


def test_registration_duplicate_username_rejected(temp_db):
    """Verify unique constraint on usernames (case-insensitive)."""
    register_user(
        username="clara_oswald",
        email="clara@example.com",
        password="Password123",
        confirm_password="Password123",
        db_path=temp_db
    )

    with pytest.raises(UserAlreadyExistsError, match="already registered"):
        register_user(
            username="CLARA_OSWALD",  # Case variant
            email="clara_new@example.com",
            password="Password123",
            confirm_password="Password123",
            db_path=temp_db
        )


def test_registration_duplicate_email_rejected(temp_db):
    """Verify unique constraint on email addresses (case-insensitive)."""
    register_user(
        username="user_alpha",
        email="shared@example.com",
        password="Password123",
        confirm_password="Password123",
        db_path=temp_db
    )

    with pytest.raises(UserAlreadyExistsError, match="already registered"):
        register_user(
            username="user_beta",
            email="SHARED@example.com",  # Case variant
            password="Password123",
            confirm_password="Password123",
            db_path=temp_db
        )


# =============================================================================
# 4. AUTHENTICATION & LOGIN FLOW
# =============================================================================
def test_authentication_with_username_and_email(temp_db):
    """Verify authentication succeeds with both username and email."""
    register_user(
        username="dana_scully",
        email="scully@fbi.gov",
        password="TrustNoOne1993",
        confirm_password="TrustNoOne1993",
        full_name="Dana Scully",
        db_path=temp_db
    )

    # Auth via username
    user_by_name = authenticate("dana_scully", "TrustNoOne1993", db_path=temp_db)
    assert user_by_name is not None
    assert user_by_name.email == "scully@fbi.gov"

    # Auth via email
    user_by_email = authenticate("scully@fbi.gov", "TrustNoOne1993", db_path=temp_db)
    assert user_by_email is not None
    assert user_by_email.username == "dana_scully"

    # Verify last_login_at is updated
    updated_user = get_user_by_id(user_by_name.id, db_path=temp_db)
    assert updated_user.last_login_at is not None


def test_authentication_failure_invalid_credentials(temp_db):
    """Verify that invalid password or missing account raises AuthenticationError."""
    register_user(
        username="auth_tester",
        email="auth@example.com",
        password="CorrectPassword1",
        confirm_password="CorrectPassword1",
        db_path=temp_db
    )

    # Incorrect password
    with pytest.raises(AuthenticationError, match="Invalid"):
        authenticate("auth_tester", "WrongPassword99", db_path=temp_db)

    # Non-existent username
    with pytest.raises(AuthenticationError, match="Invalid"):
        authenticate("non_existent_user", "AnyPassword1", db_path=temp_db)


# =============================================================================
# 5. USER PROFILE PERSISTENCE & DATA ISOLATION
# =============================================================================
def test_profile_persistence_and_cross_user_isolation(temp_db):
    """Verify user baseline profile persistence and strict isolation between User A and User B."""
    user_a = register_user("user_a", "a@example.com", "PasswordA1", "PasswordA1", db_path=temp_db)
    user_b = register_user("user_b", "b@example.com", "PasswordB1", "PasswordB1", db_path=temp_db)

    profile_data_a = {
        "age": 29.0,
        "bmi": 23.4,
        "sleep_hours": 7.5,
        "stress_score_baseline": 3.5,
        "state": "California",
        "diet_quality": "Good",
        "exercise_frequency": "3–4 days/week",
        "caffeine_intake": 2.0,
        "water_intake_liters": 2.5,
        "alcohol_consumption": "None",
        "smoking_status": "Non-smoker",
        "birth_control_use": 0,
        "pcos_diagnosed": 0,
    }

    # Save User A's profile
    save_user_profile(user_a.id, profile_data_a, db_path=temp_db)

    # User A can retrieve their own profile
    profile_retrieved_a = get_user_profile(user_a.id, db_path=temp_db)
    assert profile_retrieved_a is not None
    assert profile_retrieved_a["age"] == 29.0
    assert profile_retrieved_a["state"] == "California"

    # User B's profile must be completely empty (isolated)
    profile_retrieved_b = get_user_profile(user_b.id, db_path=temp_db)
    assert profile_retrieved_b is None

    # Upsert User A's profile with updated BMI
    profile_data_a["bmi"] = 24.0
    save_user_profile(user_a.id, profile_data_a, db_path=temp_db)
    updated_a = get_user_profile(user_a.id, db_path=temp_db)
    assert updated_a["bmi"] == 24.0


def test_profile_saving_with_invalid_user_raises(temp_db):
    """Verify that saving profile for non-existent user raises UserNotFoundError."""
    with pytest.raises(UserNotFoundError):
        save_user_profile("usr_nonexistent999", {"age": 30}, db_path=temp_db)


# =============================================================================
# 6. CYCLE LOGS PERSISTENCE & DATA ISOLATION
# =============================================================================
def test_cycle_logs_persistence_and_multi_tenant_isolation(temp_db):
    """Verify that cycle records are stored chronologically and isolated strictly by user_id."""
    user1 = register_user("pat_one", "one@example.com", "PassUser1", "PassUser1", db_path=temp_db)
    user2 = register_user("pat_two", "two@example.com", "PassUser2", "PassUser2", db_path=temp_db)

    # User 1 logs Cycle 1 and Cycle 2
    log1_id = save_user_cycle_log(
        user1.id,
        {
            "cycle_number": 1,
            "cycle_length_days": 28.0,
            "cycle_phase": "Menstrual",
            "flow_level": "Medium",
            "pain_level": 3.0,
            "predicted_next_health_score": 8.75,
            "score_category": "Optimal",
        },
        db_path=temp_db
    )
    assert log1_id.startswith("log_")

    log2_id = save_user_cycle_log(
        user1.id,
        {
            "cycle_number": 2,
            "cycle_length_days": 29.0,
            "cycle_phase": "Follicular",
            "flow_level": "Light",
            "pain_level": 2.0,
            "predicted_next_health_score": 8.90,
            "score_category": "Optimal",
        },
        db_path=temp_db
    )

    # User 2 logs Cycle 1
    save_user_cycle_log(
        user2.id,
        {
            "cycle_number": 1,
            "cycle_length_days": 32.0,
            "cycle_phase": "Luteal",
            "flow_level": "Heavy",
            "pain_level": 6.0,
            "predicted_next_health_score": 7.40,
            "score_category": "Moderate",
        },
        db_path=temp_db
    )

    # User 1 queries logs: MUST receive exactly 2 logs
    logs_u1 = get_user_cycle_logs(user1.id, db_path=temp_db)
    assert len(logs_u1) == 2
    assert logs_u1[0]["cycle_number"] == 1
    assert logs_u1[1]["cycle_number"] == 2
    assert logs_u1[0]["cycle_length_days"] == 28.0

    # User 2 queries logs: MUST receive exactly 1 log
    logs_u2 = get_user_cycle_logs(user2.id, db_path=temp_db)
    assert len(logs_u2) == 1
    assert logs_u2[0]["cycle_number"] == 1
    assert logs_u2[0]["cycle_length_days"] == 32.0

    # Automatic next cycle calculation
    assert get_next_cycle_number(user1.id, db_path=temp_db) == 3
    assert get_next_cycle_number(user2.id, db_path=temp_db) == 2


# =============================================================================
# 7. ACCOUNT DELETION & CASCADE VERIFICATION
# =============================================================================
def test_cascade_deletion_on_account_removal(temp_db):
    """Verify that deleting an account cascades and purges user profile and cycle logs."""
    user = register_user("delete_me", "delete@example.com", "DeletePass1", "DeletePass1", db_path=temp_db)

    save_user_profile(user.id, {"age": 25.0, "bmi": 21.0}, db_path=temp_db)
    save_user_cycle_log(user.id, {"cycle_number": 1, "cycle_length_days": 28.0}, db_path=temp_db)

    # Verify present
    assert get_user_profile(user.id, db_path=temp_db) is not None
    assert len(get_user_cycle_logs(user.id, db_path=temp_db)) == 1

    # Delete account
    success = delete_user_account(user.id, db_path=temp_db)
    assert success is True

    # Account gone
    assert get_user_by_id(user.id, db_path=temp_db) is None
    # Profile cascaded
    assert get_user_profile(user.id, db_path=temp_db) is None
    # Cycle logs cascaded
    assert len(get_user_cycle_logs(user.id, db_path=temp_db)) == 0


# =============================================================================
# 8. SQL INJECTION IMMUNITY
# =============================================================================
def test_parameterization_against_sql_injection(temp_db):
    """Verify that SQL injection strings are safely treated as literal values."""
    injection_user = "' OR '1'='1' --"
    injection_pwd = "' OR 1=1; DROP TABLE users; --"

    # Attempting to authenticate with SQL injection should safely fail
    with pytest.raises(AuthenticationError):
        authenticate(injection_user, injection_pwd, db_path=temp_db)

    # Database tables must remain intact
    with get_db_connection(temp_db) as conn:
        cursor = conn.execute("SELECT count(*) as cnt FROM users")
        row = cursor.fetchone()
        assert row["cnt"] == 0


# =============================================================================
# 9. SESSION STATE HELPERS & APP MODULE INTEGRATION
# =============================================================================
def test_auth_session_management():
    """Verify Streamlit session state initialization, login, user ID retrieval, and logout."""
    import streamlit as st
    from herai.auth import (
        init_auth_session,
        login_session_user,
        logout_session_user,
        get_current_user_id,
    )
    from herai.database import UserRecord

    # Reset any existing session state
    st.session_state.clear()

    # Step 1: Session initialization
    init_auth_session()
    assert st.session_state.authenticated is False
    assert st.session_state.current_user is None
    assert get_current_user_id() is None

    # Step 2: Establish login
    mock_user = UserRecord(
        id="usr_test1234567890ab",
        username="test_session_user",
        email="session@example.com",
        password_hash="fakehash",
        full_name="Session Tester",
        created_at="2026-10-30T10:00:00Z",
        last_login_at=None,
        is_active=True,
    )
    login_session_user(mock_user)
    assert st.session_state.authenticated is True
    assert st.session_state.current_user["id"] == "usr_test1234567890ab"
    assert st.session_state.current_user["username"] == "test_session_user"
    assert get_current_user_id() == "usr_test1234567890ab"

    # Step 3: Logout
    logout_session_user()
    assert st.session_state.authenticated is False
    assert st.session_state.current_user is None
    assert get_current_user_id() is None


def test_app_imports_and_handlers():
    """Verify that app.py defines the expected auth and navigation entry points."""
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import app
    assert hasattr(app, "render_auth_view")
    assert hasattr(app, "render_onboarding_flow")
    assert hasattr(app, "render_profile_view")
    assert hasattr(app, "render_app_header")
    assert hasattr(app, "render_dashboard")
    assert hasattr(app, "render_health_prediction")
    assert hasattr(app, "get_predictor")


# =============================================================================
# 10. PHASE 2: ONBOARDING AND PROFILE MANAGEMENT TESTS
# =============================================================================
def test_phase2_onboarding_and_profile_persistence(temp_db):
    """Verify Section 1 & 3: New user completes onboarding, profile is saved, and onboarding status is set."""
    from herai.database import has_completed_onboarding, UserProfileRecord

    user = register_user("elena_health", "elena@example.com", "Password123!", "Password123!", "Elena Gilbert", db_path=temp_db)
    assert has_completed_onboarding(user.id, db_path=temp_db) is False

    onboarding_data = {
        "preferred_name": "Elena",
        "age": 27.0,
        "track_cycle": True,
        "last_period_date": "2026-10-15",
        "typical_cycle_length": 29,
        "typical_period_duration": 5,
        "cycle_regularity": "Regular",
        "wellness_interests": ["Menstrual health", "PCOS/PCOD education", "Hormonal health education"],
        "onboarding_completed": True,
    }

    success = save_user_profile(user.id, onboarding_data, db_path=temp_db)
    assert success is True
    assert has_completed_onboarding(user.id, db_path=temp_db) is True

    profile = get_user_profile(user.id, db_path=temp_db)
    assert isinstance(profile, UserProfileRecord)
    assert profile.preferred_name == "Elena"
    assert profile.age == 27.0
    assert profile.track_cycle is True
    assert profile.last_period_date == "2026-10-15"
    assert profile.typical_cycle_length == 29
    assert profile.typical_period_duration == 5
    assert profile.cycle_regularity == "Regular"
    assert "Menstrual health" in profile.wellness_interests
    assert "PCOS/PCOD education" in profile.wellness_interests
    assert profile.onboarding_completed is True


def test_phase2_profile_partial_update_preserves_existing_data(temp_db):
    """Verify Section 2: Editing preferred name or interests does not wipe out previously saved cycle fields."""
    user = register_user("update_test", "update@example.com", "UpdatePass1", "UpdatePass1", db_path=temp_db)

    # Initial onboarding save
    save_user_profile(
        user.id,
        {
            "preferred_name": "Initial Name",
            "age": 28.0,
            "typical_cycle_length": 30,
            "typical_period_duration": 4,
            "cycle_regularity": "Regular",
            "wellness_interests": ["Fitness and general wellbeing"],
            "onboarding_completed": True,
        },
        db_path=temp_db
    )

    # Partial edit: User changes only preferred_name and wellness_interests
    save_user_profile(
        user.id,
        {
            "preferred_name": "Updated Name",
            "wellness_interests": ["Fitness and general wellbeing", "Sleep and stress resilience"],
        },
        db_path=temp_db
    )

    updated_profile = get_user_profile(user.id, db_path=temp_db)
    assert updated_profile.preferred_name == "Updated Name"
    # Previously saved fields MUST remain intact
    assert updated_profile.age == 28.0
    assert updated_profile.typical_cycle_length == 30
    assert updated_profile.typical_period_duration == 4
    assert updated_profile.cycle_regularity == "Regular"
    assert len(updated_profile.wellness_interests) == 2


def test_phase2_profile_validation_rejects_invalid_inputs(temp_db):
    """Verify Section 6: Input validation catches invalid dates, out-of-range cycle lengths, and bad numbers."""
    from herai.database import ProfileValidationError
    from datetime import date, timedelta

    user = register_user("val_test", "val@example.com", "ValPass123", "ValPass123", db_path=temp_db)

    # 1. Future period date rejected
    future_date = (date.today() + timedelta(days=2)).isoformat()
    with pytest.raises(ProfileValidationError) as exc:
        save_user_profile(user.id, {"last_period_date": future_date}, db_path=temp_db)
    assert "future" in str(exc.value).lower()

    # 2. Age out of bounds (< 15 or > 100)
    with pytest.raises(ProfileValidationError) as exc:
        save_user_profile(user.id, {"age": 12}, db_path=temp_db)
    assert "age" in str(exc.value).lower()

    with pytest.raises(ProfileValidationError) as exc:
        save_user_profile(user.id, {"age": 115}, db_path=temp_db)
    assert "age" in str(exc.value).lower()

    # 3. Typical cycle length out of bounds (< 18 or > 60)
    with pytest.raises(ProfileValidationError) as exc:
        save_user_profile(user.id, {"typical_cycle_length": 12}, db_path=temp_db)
    assert "cycle length" in str(exc.value).lower()

    with pytest.raises(ProfileValidationError) as exc:
        save_user_profile(user.id, {"typical_cycle_length": 75}, db_path=temp_db)
    assert "cycle length" in str(exc.value).lower()

    # 4. Period duration out of bounds (< 1 or > 15)
    with pytest.raises(ProfileValidationError) as exc:
        save_user_profile(user.id, {"typical_period_duration": 25}, db_path=temp_db)
    assert "period duration" in str(exc.value).lower()

    # 5. Invalid cycle regularity category
    with pytest.raises(ProfileValidationError) as exc:
        save_user_profile(user.id, {"cycle_regularity": "Completely Chaotic"}, db_path=temp_db)
    assert "regularity" in str(exc.value).lower()


def test_phase2_multi_user_profile_isolation(temp_db):
    """Verify Section 3 & 4: Two users have strictly separated profiles; user A cannot view or corrupt user B's profile."""
    user1 = register_user("user_maya", "maya@wellness.org", "MayaPass123", "MayaPass123", "Maya Lin", db_path=temp_db)
    user2 = register_user("user_sarah", "sarah@wellness.org", "SarahPass123", "SarahPass123", "Sarah Connor", db_path=temp_db)

    save_user_profile(
        user1.id,
        {
            "preferred_name": "Maya",
            "age": 29.0,
            "typical_cycle_length": 28,
            "wellness_interests": ["Cycle tracking", "Menstrual health"],
            "onboarding_completed": True,
        },
        db_path=temp_db
    )

    save_user_profile(
        user2.id,
        {
            "preferred_name": "Sarah",
            "age": 34.0,
            "typical_cycle_length": 35,
            "wellness_interests": ["PCOS/PCOD education", "Hormonal health education"],
            "onboarding_completed": True,
        },
        db_path=temp_db
    )

    # User 1 profile inspection
    prof1 = get_user_profile(user1.id, db_path=temp_db)
    assert prof1.preferred_name == "Maya"
    assert prof1.age == 29.0
    assert prof1.typical_cycle_length == 28
    assert "Cycle tracking" in prof1.wellness_interests
    assert "PCOS/PCOD education" not in prof1.wellness_interests

    # User 2 profile inspection
    prof2 = get_user_profile(user2.id, db_path=temp_db)
    assert prof2.preferred_name == "Sarah"
    assert prof2.age == 34.0
    assert prof2.typical_cycle_length == 35
    assert "PCOS/PCOD education" in prof2.wellness_interests
    assert "Cycle tracking" not in prof2.wellness_interests


def test_phase2_missing_optional_profile_fields_permitted(temp_db):
    """Verify Section 4 & 6: Missing optional information does not prevent saving or accessing the profile."""
    from herai.database import has_completed_onboarding

    user = register_user("opt_user", "opt@example.com", "Optional123", "Optional123", db_path=temp_db)

    # User provides only preferred_name and track_cycle=False, omitting age, dates, cycle lengths, etc.
    save_user_profile(
        user.id,
        {
            "preferred_name": "Alex",
            "track_cycle": False,
            "onboarding_completed": True,
        },
        db_path=temp_db
    )

    assert has_completed_onboarding(user.id, db_path=temp_db) is True
    prof = get_user_profile(user.id, db_path=temp_db)
    assert prof.preferred_name == "Alex"
    assert prof.track_cycle is False
    assert prof.age is None
    assert prof.last_period_date is None
    assert prof.typical_cycle_length is None
    assert prof.wellness_interests == []


# =============================================================================
# PHASE 3: DASHBOARD STATS, PERSONAL METRICS & DATA ISOLATION
# =============================================================================
def test_phase3_dashboard_stats_new_user_empty_state(temp_db):
    """Verify Phase 3: New user without cycle logs receives empty state metrics without fabricated data."""
    from herai.database import get_user_dashboard_stats

    user = register_user("brand_new_user", "new@example.com", "SecurePass123", "SecurePass123", db_path=temp_db)
    save_user_profile(
        user.id,
        {
            "preferred_name": "Clara",
            "typical_cycle_length": 28,
            "typical_period_duration": 5,
            "cycle_regularity": "Regular",
            "sleep_hours": 8.0,
            "stress_score_baseline": 3.0,
            "wellness_interests": ["Sleep optimization"],
            "onboarding_completed": True,
        },
        db_path=temp_db
    )

    stats = get_user_dashboard_stats(user.id, db_path=temp_db)
    assert stats["total_cycles"] == 0
    assert stats["latest_cycle"] is None
    assert stats["latest_health_score"] is None
    assert stats["latest_health_tier"] is None
    assert stats["avg_cycle_length"] == 28.0
    assert stats["avg_sleep"] == 8.0
    assert stats["avg_stress"] == 3.0
    assert stats["profile"].preferred_name == "Clara"


def test_phase3_dashboard_stats_returning_user_with_cycles(temp_db):
    """Verify Phase 3: Returning user with multiple cycles computes correct personal aggregations."""
    from herai.database import get_user_dashboard_stats

    user = register_user("returning_user", "returning@example.com", "SecurePass123", "SecurePass123", db_path=temp_db)
    save_user_profile(
        user.id,
        {
            "preferred_name": "Elena",
            "typical_cycle_length": 28,
            "onboarding_completed": True,
        },
        db_path=temp_db
    )

    # Save Cycle 1
    save_user_cycle_log(
        user.id,
        log_data={
            "cycle_number": 1,
            "cycle_length_days": 28.0,
            "cycle_phase": "Menstrual",
            "flow_level": "Medium",
            "pain_level": 3.0,
            "sleep_hours_cycle": 7.5,
            "stress_score_cycle": 4.0,
            "predicted_next_health_score": 8.65,
            "score_category": "Optimal",
        },
        db_path=temp_db
    )

    # Save Cycle 2
    save_user_cycle_log(
        user.id,
        log_data={
            "cycle_number": 2,
            "cycle_length_days": 30.0,
            "cycle_phase": "Follicular",
            "flow_level": "Light",
            "pain_level": 1.0,
            "sleep_hours_cycle": 8.5,
            "stress_score_cycle": 2.0,
            "predicted_next_health_score": 9.10,
            "score_category": "Optimal",
        },
        db_path=temp_db
    )

    stats = get_user_dashboard_stats(user.id, db_path=temp_db)
    assert stats["total_cycles"] == 2
    assert stats["avg_cycle_length"] == 29.0
    assert stats["latest_cycle"].cycle_number == 2
    assert stats["latest_health_score"] == 9.10
    assert stats["latest_health_tier"] == "Optimal"
    assert stats["avg_sleep"] == 8.0
    assert stats["avg_stress"] == 3.0
    assert stats["avg_pain"] == 2.0


def test_phase3_dashboard_stats_isolation_between_users(temp_db):
    """Verify Phase 3: Strict user isolation prevents cross-account leak of cycle logs and metrics."""
    from herai.database import get_user_dashboard_stats

    user_a = register_user("user_alpha", "alpha@example.com", "Password123", "Password123", db_path=temp_db)
    user_b = register_user("user_beta", "beta@example.com", "Password123", "Password123", db_path=temp_db)

    # User A logs cycles
    save_user_cycle_log(
        user_a.id,
        log_data={
            "cycle_number": 1,
            "cycle_length_days": 27.0,
            "predicted_next_health_score": 8.80,
            "score_category": "Optimal",
        },
        db_path=temp_db
    )
    save_user_cycle_log(
        user_a.id,
        log_data={
            "cycle_number": 2,
            "cycle_length_days": 29.0,
            "predicted_next_health_score": 8.95,
            "score_category": "Optimal",
        },
        db_path=temp_db
    )

    stats_a = get_user_dashboard_stats(user_a.id, db_path=temp_db)
    stats_b = get_user_dashboard_stats(user_b.id, db_path=temp_db)

    # User A has 2 cycles
    assert stats_a["total_cycles"] == 2
    assert stats_a["latest_health_score"] == 8.95

    # User B has 0 cycles, completely isolated
    assert stats_b["total_cycles"] == 0
    assert stats_b["latest_health_score"] is None
    assert stats_b["latest_cycle"] is None


def test_phase3_user_cycle_log_record_duality(temp_db):
    """Verify Phase 3: UserCycleLogRecord implements dual attribute and dictionary access."""
    from herai.database import UserCycleLogRecord

    record = UserCycleLogRecord(
        id="log_test1",
        user_id="usr_test1",
        cycle_number=3,
        cycle_length_days=28.5,
        predicted_next_health_score=8.75,
        score_category="Optimal",
    )

    # Attribute access
    assert record.cycle_number == 3
    assert record.cycle_length_days == 28.5
    assert record.predicted_next_health_score == 8.75
    assert record.predicted_health_score == 8.75  # Alias property
    assert record.prediction_tier == "Optimal"      # Alias property

    # Dictionary access
    assert record["cycle_number"] == 3
    assert record["cycle_length_days"] == 28.5
    assert record["predicted_next_health_score"] == 8.75
    assert record["predicted_health_score"] == 8.75
    assert record["prediction_tier"] == "Optimal"

    # Dictionary helper methods
    assert record.get("cycle_number") == 3
    assert record.get("non_existent", "fallback") == "fallback"
    assert "cycle_number" in record
    assert "predicted_health_score" in record
    assert isinstance(record.to_dict(), dict)


def test_phase3_personal_visualizations_render_safely():
    """Verify Phase 3: Personal trajectory and biomarker trend charts generate valid Plotly figures."""
    from herai.visualizations import (
        create_personal_health_trajectory_chart,
        create_personal_biomarker_trend_chart,
    )
    from herai.database import UserCycleLogRecord

    # 1. Empty logs
    fig_empty = create_personal_health_trajectory_chart([])
    assert fig_empty is not None
    assert hasattr(fig_empty, "to_plotly_json")

    fig_bio_empty = create_personal_biomarker_trend_chart([])
    assert fig_bio_empty is not None
    assert hasattr(fig_bio_empty, "to_plotly_json")

    # 2. Populated logs
    logs = [
        UserCycleLogRecord(
            id="log1", user_id="u1", cycle_number=1, cycle_length_days=28.0,
            cycle_phase="Menstrual", pain_level=3.0, sleep_hours_cycle=7.5,
            stress_score_cycle=4.0, predicted_next_health_score=8.50, score_category="Optimal"
        ),
        UserCycleLogRecord(
            id="log2", user_id="u1", cycle_number=2, cycle_length_days=29.0,
            cycle_phase="Follicular", pain_level=1.0, sleep_hours_cycle=8.0,
            stress_score_cycle=3.0, predicted_next_health_score=8.85, score_category="Optimal"
        ),
    ]

    fig_populated = create_personal_health_trajectory_chart(logs, user_name="Maya's")
    assert len(fig_populated.data) >= 2  # Cohort mean + user trajectory

    fig_bio_populated = create_personal_biomarker_trend_chart(logs)
    assert len(fig_bio_populated.data) == 3  # Sleep + Stress + Pain


# =============================================================================
# PHASE 4: END-TO-END DEMO READINESS & LIFECYCLE INTEGRATION
# =============================================================================
def test_phase4_end_to_end_user_journey(temp_db):
    """End-to-end integration test verifying the complete user lifecycle for demonstration readiness:
    
    1. Registration & duplicate rejection
    2. Password verification & authentication
    3. Onboarding completion & profile persistence
    4. ML prediction inference with user-owned data
    5. Cycle log persistence scoped to user_id
    6. Dashboard stats & physiological phase estimation
    7. Multi-user isolation against cross-account leaks
    8. Account deletion cascade
    """
    from datetime import date, timedelta
    from herai.database import (
        get_user_dashboard_stats,
        has_completed_onboarding,
        delete_user,
    )
    from herai.inference import (
        HerAIPredictor,
        UserProfileInput,
        CurrentCycleInput,
    )

    # 1. Registration
    user = register_user(
        username="sarah_demo",
        email="sarah.demo@example.com",
        password="SecurePassword123",
        confirm_password="SecurePassword123",
        db_path=temp_db
    )
    assert user.id.startswith("usr_")

    # Duplicate registration rejected
    with pytest.raises(UserAlreadyExistsError):
        register_user("sarah_demo", "diff@example.com", "SecurePassword123", "SecurePassword123", db_path=temp_db)

    # 2. Authentication with valid & invalid credentials
    auth_user = authenticate("sarah_demo", "SecurePassword123", db_path=temp_db)
    assert auth_user.id == user.id

    with pytest.raises(AuthenticationError):
        authenticate("sarah_demo", "WrongPassword999", db_path=temp_db)

    # Initial state: onboarding not completed
    assert has_completed_onboarding(user.id, db_path=temp_db) is False

    # 3. Onboarding & Profile Persistence
    ten_days_ago = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    save_user_profile(
        user.id,
        {
            "preferred_name": "Sarah",
            "age": 27.0,
            "typical_cycle_length": 28,
            "typical_period_duration": 5,
            "cycle_regularity": "Regular",
            "last_period_date": ten_days_ago,
            "wellness_interests": ["Sleep optimization", "Cycle tracking"],
            "sleep_hours": 8.0,
            "water_intake_liters": 2.2,
            "exercise_frequency": "3–4 days/week",
            "stress_score_baseline": 3.0,
            "onboarding_completed": True,
        },
        db_path=temp_db
    )
    assert has_completed_onboarding(user.id, db_path=temp_db) is True
    profile = get_user_profile(user.id, db_path=temp_db)
    assert profile.preferred_name == "Sarah"
    assert profile.typical_cycle_length == 28

    # 4. ML Prediction with user-owned data
    predictor = HerAIPredictor()
    prof_input = UserProfileInput(
        age=profile.age,
        bmi=22.4,
        sleep_hours=profile.sleep_hours,
        stress_score_baseline=profile.stress_score_baseline,
        state="California",
        diet_quality="Good",
        exercise_frequency=profile.exercise_frequency,
        caffeine_intake=1.5,
        water_intake_liters=profile.water_intake_liters,
        alcohol_consumption="None",
        smoking_status="Non-smoker",
        birth_control_use=0,
        pcos_diagnosed=0,
    )
    cycle_input = CurrentCycleInput(
        cycle_length_days=float(profile.typical_cycle_length),
        pain_level=2.0,
        mood_score=8.0,
        stress_score_cycle=3.0,
        sleep_hours_cycle=8.0,
        energy_level=8.0,
        concentration_score=8.0,
        work_hours_lost=0.0,
        cycle_phase="Follicular",
        flow_level="Medium",
    )
    prediction = predictor.predict(prof_input, cycle_input)
    assert 1.0 <= prediction.predicted_next_health_score <= 10.0
    assert prediction.score_category in ("Optimal", "Moderate", "Sub-optimal")
    assert prediction.prediction_tier == prediction.score_category

    # 5. Save Cycle Log
    log_id = save_user_cycle_log(
        user_id=user.id,
        cycle_data={
            "cycle_number": 1,
            "cycle_length_days": 28.0,
            "cycle_phase": "Follicular",
            "flow_level": "Medium",
            "pain_level": 2.0,
            "mood_score": 8.0,
            "stress_score_cycle": 3.0,
            "sleep_hours_cycle": 8.0,
            "energy_level": 8.0,
            "concentration_score": 8.0,
            "work_hours_lost": 0.0,
        },
        predicted_score=prediction.predicted_next_health_score,
        prediction_tier=prediction.prediction_tier,
        db_path=temp_db
    )
    assert log_id.startswith("log_")

    # 6. Dashboard Stats & Dynamic Phase Estimation
    stats = get_user_dashboard_stats(user.id, db_path=temp_db)
    assert stats["total_cycles"] == 1
    assert stats["latest_cycle"].cycle_number == 1
    assert stats["latest_health_score"] == prediction.predicted_next_health_score
    assert stats["latest_health_tier"] == prediction.prediction_tier
    assert stats["estimated_cycle_day"] == 11
    assert stats["estimated_phase"] == "Follicular"

    # 7. Multi-User Isolation
    other_user = register_user(
        username="elena_demo",
        email="elena.demo@example.com",
        password="SecurePassword456",
        confirm_password="SecurePassword456",
        db_path=temp_db
    )
    other_stats = get_user_dashboard_stats(other_user.id, db_path=temp_db)
    assert other_stats["total_cycles"] == 0
    assert other_stats["latest_cycle"] is None
    assert other_stats["latest_health_score"] is None
    assert other_stats["profile"] is None

    # 8. Account Deletion Cascade
    delete_user(user.id, db_path=temp_db)
    assert get_user_profile(user.id, db_path=temp_db) is None
    assert len(get_user_cycle_logs(user.id, db_path=temp_db)) == 0


