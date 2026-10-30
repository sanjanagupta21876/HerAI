"""Authentication and session management module for HERAI.

Features:
- Cryptographically secure salted password hashing via bcrypt (adaptive work factor = 12).
- Timing-safe password verification.
- Input hygiene: email normalization, strict username constraints, password complexity checks.
- Streamlit session state management for authenticated multi-user sessions.
"""

import re
from typing import Optional, Tuple, Dict, Any
import bcrypt
import streamlit as st

from herai.database import (
    UserRecord,
    create_user,
    get_user_by_identifier,
    update_user_last_login,
    UserAlreadyExistsError,
    ValidationError,
)


class AuthenticationError(Exception):
    """Raised when authentication credentials fail verification."""
    pass


# =============================================================================
# PASSWORD CRYPTOGRAPHY (BCRYPT)
# =============================================================================
def hash_password(plain_password: str) -> str:
    """Hash a plaintext password with a unique, cryptographically random salt."""
    if not plain_password or len(plain_password) < 8:
        raise ValidationError("Password must be at least 8 characters.")

    salt = bcrypt.gensalt(rounds=12)
    hashed_bytes = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed_bytes.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a candidate password against a stored bcrypt hash using constant-time comparison."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


# =============================================================================
# INPUT VALIDATION
# =============================================================================
_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,29}$")
_EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_username(username: str) -> Tuple[bool, str]:
    """Validate username length and characters (3-30 chars, alphanumeric, dashes, underscores)."""
    if not username:
        return False, "Username is required."
    clean = username.strip()
    if len(clean) < 3 or len(clean) > 30:
        return False, "Username must be between 3 and 30 characters."
    if not _USERNAME_PATTERN.match(clean):
        return False, "Username must start with a letter or number and contain only letters, numbers, hyphens, and underscores."
    return True, clean.lower()


def validate_email(email: str) -> Tuple[bool, str]:
    """Validate email format and return lowercased normalized string."""
    if not email:
        return False, "Email address is required."
    clean = email.strip()
    if not _EMAIL_PATTERN.match(clean) or len(clean) > 254:
        return False, "Please enter a valid email address."
    return True, clean.lower()


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Ensure password meets minimum security standards."""
    if not password:
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if len(password) > 128:
        return False, "Password cannot exceed 128 characters."
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    return True, "Password is valid."


# =============================================================================
# HIGH-LEVEL AUTHENTICATION SERVICES
# =============================================================================
def register_user(
    username: str,
    email: str,
    password: str,
    confirm_password: str,
    full_name: Optional[str] = None,
    db_path: Optional[Any] = None,
) -> UserRecord:
    """Validate credentials and register a new user account."""
    is_valid_user, user_res = validate_username(username)
    if not is_valid_user:
        raise ValidationError(user_res)

    is_valid_email, email_res = validate_email(email)
    if not is_valid_email:
        raise ValidationError(email_res)

    if password != confirm_password:
        raise ValidationError("Passwords do not match.")

    is_valid_pw, pw_res = validate_password_strength(password)
    if not is_valid_pw:
        raise ValidationError(pw_res)

    pwd_hash = hash_password(password)
    return create_user(
        username=user_res,
        email=email_res,
        password_hash=pwd_hash,
        full_name=full_name,
        db_path=db_path
    )


def authenticate(
    identifier: str,
    password: str,
    db_path: Optional[Any] = None,
) -> Optional[UserRecord]:
    """Authenticate a user using their username or email and password."""
    if not identifier or not password:
        raise AuthenticationError("Username/email and password are required.")

    user = get_user_by_identifier(identifier, db_path=db_path)
    if not user:
        raise AuthenticationError("Invalid username/email or password.")

    if not user.is_active:
        raise AuthenticationError("This account is inactive. Please contact support.")

    if not verify_password(password, user.password_hash):
        raise AuthenticationError("Invalid username/email or password.")

    update_user_last_login(user.id, db_path=db_path)
    return user


# =============================================================================
# STREAMLIT SESSION STATE HELPERS
# =============================================================================
def init_auth_session() -> None:
    """Initialize session state keys for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None


def login_session_user(user: UserRecord) -> None:
    """Establish active authenticated session in Streamlit session state."""
    st.session_state.authenticated = True
    st.session_state.current_user = user.to_dict()


def logout_session_user() -> None:
    """Terminate the authenticated session and purge user-specific session caches."""
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.last_prediction = None
    st.session_state.last_inputs = None


def get_current_user_id() -> Optional[str]:
    """Return user_id of currently authenticated user, or None if guest."""
    if st.session_state.get("authenticated") and st.session_state.get("current_user"):
        return st.session_state.current_user.get("id")
    return None

