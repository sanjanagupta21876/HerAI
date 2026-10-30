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
    assert hasattr(app, "render_app_header")
    assert hasattr(app, "render_dashboard")
    assert hasattr(app, "render_health_prediction")
    assert hasattr(app, "get_predictor")
