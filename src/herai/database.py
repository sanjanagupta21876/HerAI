"""Persistent SQLite database manager for HERAI multi-user accounts and records.

Provides:
- Thread-safe connection management with SQLite WAL mode and foreign key enforcement.
- Parameterized SQL queries for all database operations to eliminate injection vulnerabilities.
- Strict user-level data isolation: profile and cycle operations require and verify user_id ownership.
- Comprehensive cascade cleanup on account deletion.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
import uuid

from herai.config import DATABASE_PATH, DATA_DIR


# Custom Exception Hierarchy
class DatabaseError(Exception):
    """Base exception for all HERAI database errors."""
    pass


class UserAlreadyExistsError(DatabaseError):
    """Raised when registering a username or email that is already registered."""
    pass


class UserNotFoundError(DatabaseError):
    """Raised when an operation references a non-existent user."""
    pass


class OwnershipError(DatabaseError):
    """Raised when an operation attempts unauthorized access across user boundaries."""
    pass


class ValidationError(DatabaseError):
    """Raised when database input fails integrity or format checks."""
    pass


class ProfileValidationError(ValidationError):
    """Raised when profile inputs fail clinical bounds or format validation."""
    def __init__(self, message: str, errors: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.errors = errors or {}


@dataclass
class UserProfileRecord:
    """Represents a personalized user profile and clinical baseline record."""
    user_id: str
    preferred_name: Optional[str] = None
    age: Optional[float] = None
    date_of_birth: Optional[str] = None
    track_cycle: bool = True
    last_period_date: Optional[str] = None
    typical_cycle_length: Optional[int] = None
    typical_period_duration: Optional[int] = None
    cycle_regularity: Optional[str] = None
    wellness_interests: List[str] = field(default_factory=list)
    onboarding_completed: bool = False
    bmi: Optional[float] = None
    sleep_hours: Optional[float] = None
    stress_score_baseline: Optional[float] = None
    state: Optional[str] = None
    diet_quality: Optional[str] = None
    exercise_frequency: Optional[str] = None
    caffeine_intake: Optional[float] = None
    water_intake_liters: Optional[float] = None
    alcohol_consumption: Optional[str] = None
    smoking_status: Optional[str] = None
    birth_control_use: Optional[int] = 0
    pcos_diagnosed: Optional[int] = 0
    updated_at: str = ""

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style indexing for backwards compatibility."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-style get() method for backwards compatibility."""
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' membership tests for dict compatibility."""
        return hasattr(self, key)

    def __iter__(self):
        """Allow iteration over keys like a dict."""
        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return number of fields like a dict."""
        return len(self.to_dict())

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a dictionary."""
        return asdict(self)


@dataclass
class UserRecord:
    """Represents an authenticated user account record."""
    id: str
    username: str
    email: str
    password_hash: str
    full_name: Optional[str] = None
    created_at: str = ""
    last_login_at: Optional[str] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to safe dictionary (excluding password_hash for UI exposure)."""
        d = asdict(self)
        d.pop("password_hash", None)
        return d


def get_db_path(custom_path: Optional[Path] = None) -> Path:
    """Resolve database path, allowing environment or argument override."""
    if custom_path is not None:
        return Path(custom_path)
    env_override = os.environ.get("HERAI_DB_PATH")
    if env_override:
        return Path(env_override)
    return DATABASE_PATH


@contextmanager
def get_db_connection(db_path: Optional[Path] = None):
    """Context manager for SQLite database connection.
    
    Guarantees:
    - Foreign keys are enabled (`PRAGMA foreign_keys = ON;`).
    - WAL journaling enabled for reliable concurrent read/write access.
    - Transaction committed on clean exit, rolled back on error.
    - Resources closed cleanly.
    """
    resolved_path = get_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(resolved_path),
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize relational database schemas and indexes."""
    with get_db_connection(db_path) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL COLLATE NOCASE,
            email TEXT NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT NOT NULL,
            last_login_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            CONSTRAINT uq_username UNIQUE (username),
            CONSTRAINT uq_email UNIQUE (email)
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            preferred_name TEXT,
            age REAL,
            date_of_birth TEXT,
            track_cycle INTEGER DEFAULT 1,
            last_period_date TEXT,
            typical_cycle_length INTEGER,
            typical_period_duration INTEGER,
            cycle_regularity TEXT,
            wellness_interests TEXT,
            onboarding_completed INTEGER DEFAULT 0,
            bmi REAL,
            sleep_hours REAL,
            stress_score_baseline REAL,
            state TEXT,
            diet_quality TEXT,
            exercise_frequency TEXT,
            caffeine_intake REAL,
            water_intake_liters REAL,
            alcohol_consumption TEXT,
            smoking_status TEXT,
            birth_control_use INTEGER,
            pcos_diagnosed INTEGER,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)

        # Ensure backwards-compatible column migration for existing database files
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(user_profiles)").fetchall()}
        phase2_cols = [
            ("preferred_name", "TEXT"),
            ("date_of_birth", "TEXT"),
            ("track_cycle", "INTEGER DEFAULT 1"),
            ("last_period_date", "TEXT"),
            ("typical_cycle_length", "INTEGER"),
            ("typical_period_duration", "INTEGER"),
            ("cycle_regularity", "TEXT"),
            ("wellness_interests", "TEXT"),
            ("onboarding_completed", "INTEGER DEFAULT 0"),
        ]
        for col_name, col_type in phase2_cols:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE user_profiles ADD COLUMN {col_name} {col_type};")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_cycle_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            cycle_number INTEGER NOT NULL,
            cycle_length_days REAL,
            cycle_phase TEXT,
            flow_level TEXT,
            pain_level REAL,
            pms_symptoms TEXT,
            mood_score REAL,
            stress_score_cycle REAL,
            sleep_hours_cycle REAL,
            energy_level REAL,
            concentration_score REAL,
            work_hours_lost REAL,
            predicted_next_health_score REAL,
            score_category TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            CONSTRAINT uq_user_cycle UNIQUE (user_id, cycle_number)
        );
        """)

        # Performance and integrity indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cycle_logs_user_cycle ON user_cycle_logs(user_id, cycle_number);")


# =============================================================================
# USER ACCOUNT MANAGEMENT
# =============================================================================
def create_user(
    username: str,
    email: str,
    password_hash: str,
    full_name: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> UserRecord:
    """Create a new user account with parameterized query and unique constraints."""
    clean_username = username.strip().lower()
    clean_email = email.strip().lower()

    if not clean_username or not clean_email or not password_hash:
        raise ValidationError("Username, email, and password hash are required.")

    user_id = f"usr_{uuid.uuid4().hex[:16]}"
    now_utc = datetime.now(timezone.utc).isoformat()

    init_db(db_path)

    try:
        with get_db_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, full_name, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (user_id, clean_username, clean_email, password_hash, full_name.strip() if full_name else None, now_utc)
            )
    except sqlite3.IntegrityError as err:
        err_msg = str(err).lower()
        if "uq_username" in err_msg or "username" in err_msg:
            raise UserAlreadyExistsError(f"Username '{clean_username}' is already registered.")
        if "uq_email" in err_msg or "email" in err_msg:
            raise UserAlreadyExistsError(f"Email '{clean_email}' is already registered.")
        raise DatabaseError(f"Integrity constraint violation: {err}")

    return UserRecord(
        id=user_id,
        username=clean_username,
        email=clean_email,
        password_hash=password_hash,
        full_name=full_name.strip() if full_name else None,
        created_at=now_utc,
        last_login_at=None,
        is_active=True
    )


def get_user_by_id(user_id: str, db_path: Optional[Path] = None) -> Optional[UserRecord]:
    """Retrieve user record by unique user_id using parameterized query."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, username, email, password_hash, full_name, created_at, last_login_at, is_active
            FROM users WHERE id = ?
            """,
            (user_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return UserRecord(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            full_name=row["full_name"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
            is_active=bool(row["is_active"])
        )


def get_user_by_username(username: str, db_path: Optional[Path] = None) -> Optional[UserRecord]:
    """Retrieve user record by username (case-insensitive) using parameterized query."""
    init_db(db_path)
    clean_username = username.strip().lower()
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, username, email, password_hash, full_name, created_at, last_login_at, is_active
            FROM users WHERE username = ?
            """,
            (clean_username,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return UserRecord(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            full_name=row["full_name"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
            is_active=bool(row["is_active"])
        )


def get_user_by_email(email: str, db_path: Optional[Path] = None) -> Optional[UserRecord]:
    """Retrieve user record by email (case-insensitive) using parameterized query."""
    init_db(db_path)
    clean_email = email.strip().lower()
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, username, email, password_hash, full_name, created_at, last_login_at, is_active
            FROM users WHERE email = ?
            """,
            (clean_email,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return UserRecord(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            full_name=row["full_name"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
            is_active=bool(row["is_active"])
        )


def get_user_by_identifier(identifier: str, db_path: Optional[Path] = None) -> Optional[UserRecord]:
    """Find a user by either username or email address using parameterized query."""
    init_db(db_path)
    clean_id = identifier.strip().lower()
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, username, email, password_hash, full_name, created_at, last_login_at, is_active
            FROM users WHERE username = ? OR email = ?
            """,
            (clean_id, clean_id)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return UserRecord(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            full_name=row["full_name"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
            is_active=bool(row["is_active"])
        )


def update_user_last_login(user_id: str, db_path: Optional[Path] = None) -> bool:
    """Record login timestamp for user_id using parameterized query."""
    init_db(db_path)
    now_utc = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (now_utc, user_id)
        )
        return cursor.rowcount > 0


def delete_user_account(user_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete a user account and cascade delete all their profiles and cycle logs."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cursor.rowcount > 0


delete_user = delete_user_account


# =============================================================================
# USER PROFILE VALIDATION & STORAGE (STRICT USER_ID ISOLATION)
# =============================================================================
def validate_profile_inputs(data: Dict[str, Any]) -> Tuple[bool, Dict[str, str], Dict[str, Any]]:
    """Validate user profile inputs against clinical bounds, types, and logic.
    
    Returns:
        Tuple of (is_valid: bool, error_map: Dict[str, str], cleaned_data: Dict[str, Any])
    """
    errors: Dict[str, str] = {}
    cleaned: Dict[str, Any] = dict(data)

    # 1. Preferred Name (optional, max 60 chars)
    if "preferred_name" in data and data["preferred_name"] is not None:
        pname = str(data["preferred_name"]).strip()
        if len(pname) > 60:
            errors["preferred_name"] = "Preferred name cannot exceed 60 characters."
        else:
            cleaned["preferred_name"] = pname if pname else None

    # 2. Age (optional, between 15 and 100)
    if "age" in data and data["age"] is not None and data["age"] != "":
        try:
            age_val = float(data["age"])
            if age_val < 15 or age_val > 100:
                errors["age"] = "Age must be between 15 and 100 years."
            else:
                cleaned["age"] = age_val
        except (ValueError, TypeError):
            errors["age"] = "Please enter a valid numeric age."

    # 3. Track Cycle (boolean)
    if "track_cycle" in data:
        cleaned["track_cycle"] = bool(data["track_cycle"])

    # 4. Last Period Date (optional, YYYY-MM-DD, cannot be in future, within 365 days)
    if "last_period_date" in data and data["last_period_date"]:
        date_raw = data["last_period_date"]
        try:
            if isinstance(date_raw, datetime):
                dt_val = date_raw.date()
            elif isinstance(date_raw, date):
                dt_val = date_raw
            else:
                dt_val = datetime.strptime(str(date_raw).strip(), "%Y-%m-%d").date()

            today = date.today()
            if dt_val > today:
                errors["last_period_date"] = "Last period date cannot be in the future."
            elif (today - dt_val).days > 365:
                errors["last_period_date"] = "Last period date cannot be more than 1 year in the past."
            else:
                cleaned["last_period_date"] = dt_val.isoformat()
        except ValueError:
            errors["last_period_date"] = "Please enter a valid date in YYYY-MM-DD format."

    # 5. Typical Cycle Length (optional, integer between 18 and 60 days)
    if "typical_cycle_length" in data and data["typical_cycle_length"] is not None and data["typical_cycle_length"] != "":
        try:
            clen = int(data["typical_cycle_length"])
            if clen < 18 or clen > 60:
                errors["typical_cycle_length"] = "Typical cycle length must be between 18 and 60 days."
            else:
                cleaned["typical_cycle_length"] = clen
        except (ValueError, TypeError):
            errors["typical_cycle_length"] = "Typical cycle length must be a whole number of days."

    # 6. Typical Period Duration (optional, integer between 1 and 15 days)
    if "typical_period_duration" in data and data["typical_period_duration"] is not None and data["typical_period_duration"] != "":
        try:
            pdur = int(data["typical_period_duration"])
            if pdur < 1 or pdur > 15:
                errors["typical_period_duration"] = "Period duration must be between 1 and 15 days."
            else:
                cleaned["typical_period_duration"] = pdur
        except (ValueError, TypeError):
            errors["typical_period_duration"] = "Period duration must be a whole number of days."

    # 7. Cycle Regularity (optional, specific allowed categories)
    valid_regularities = {"Regular", "Irregular", "Varied / Not Sure", "Prefer not to say"}
    if "cycle_regularity" in data and data["cycle_regularity"]:
        reg_val = str(data["cycle_regularity"]).strip()
        if reg_val not in valid_regularities:
            errors["cycle_regularity"] = f"Please select one of the supported regularity options: {', '.join(valid_regularities)}."
        else:
            cleaned["cycle_regularity"] = reg_val

    # 8. Wellness Interests (list of strings)
    if "wellness_interests" in data and data["wellness_interests"] is not None:
        if isinstance(data["wellness_interests"], list):
            cleaned["wellness_interests"] = [str(item).strip() for item in data["wellness_interests"] if str(item).strip()]
        elif isinstance(data["wellness_interests"], str):
            try:
                parsed = json.loads(data["wellness_interests"])
                if isinstance(parsed, list):
                    cleaned["wellness_interests"] = parsed
                else:
                    cleaned["wellness_interests"] = [data["wellness_interests"].strip()]
            except json.JSONDecodeError:
                cleaned["wellness_interests"] = [item.strip() for item in data["wellness_interests"].split(",") if item.strip()]

    # 9. BMI (optional, between 12 and 60)
    if "bmi" in data and data["bmi"] is not None and data["bmi"] != "":
        try:
            bmi_val = float(data["bmi"])
            if bmi_val < 12.0 or bmi_val > 60.0:
                errors["bmi"] = "BMI must be between 12.0 and 60.0."
            else:
                cleaned["bmi"] = bmi_val
        except (ValueError, TypeError):
            errors["bmi"] = "Please enter a valid numeric BMI."

    # 10. Sleep Hours (optional, between 2.0 and 16.0)
    if "sleep_hours" in data and data["sleep_hours"] is not None and data["sleep_hours"] != "":
        try:
            sleep_val = float(data["sleep_hours"])
            if sleep_val < 2.0 or sleep_val > 16.0:
                errors["sleep_hours"] = "Sleep duration must be between 2.0 and 16.0 hours."
            else:
                cleaned["sleep_hours"] = sleep_val
        except (ValueError, TypeError):
            errors["sleep_hours"] = "Please enter a valid numeric sleep duration."

    # 11. Baseline Stress (optional, between 0.0 and 10.0)
    if "stress_score_baseline" in data and data["stress_score_baseline"] is not None and data["stress_score_baseline"] != "":
        try:
            stress_val = float(data["stress_score_baseline"])
            if stress_val < 0.0 or stress_val > 10.0:
                errors["stress_score_baseline"] = "Stress score must be between 0.0 and 10.0."
            else:
                cleaned["stress_score_baseline"] = stress_val
        except (ValueError, TypeError):
            errors["stress_score_baseline"] = "Please enter a valid numeric stress score."

    return (len(errors) == 0, errors, cleaned)


def save_user_profile(
    user_id: str,
    profile_data: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> bool:
    """Upsert user baseline profile data with strict user_id ownership enforcement and input validation."""
    init_db(db_path)
    if not user_id:
        raise OwnershipError("User ID must be provided to save profile data.")

    # Verify user exists
    user = get_user_by_id(user_id, db_path=db_path)
    if not user:
        raise UserNotFoundError(f"User '{user_id}' does not exist.")

    # Validate inputs
    is_valid, errors, cleaned = validate_profile_inputs(profile_data)
    if not is_valid:
        first_err = next(iter(errors.values()))
        raise ProfileValidationError(f"Invalid profile data: {first_err}", errors=errors)

    # Fetch existing profile to preserve unchanged fields during partial updates
    existing = get_user_profile(user_id, db_path=db_path)
    merged: Dict[str, Any] = {}
    if existing:
        merged.update(existing.to_dict())
    merged.update(cleaned)

    now_utc = datetime.now(timezone.utc).isoformat()
    wellness_list = merged.get("wellness_interests") or []
    if isinstance(wellness_list, str):
        try:
            wellness_list = json.loads(wellness_list)
        except Exception:
            wellness_list = [item.strip() for item in wellness_list.split(",") if item.strip()]
    wellness_json = json.dumps(wellness_list)

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (
                user_id, preferred_name, age, date_of_birth, track_cycle,
                last_period_date, typical_cycle_length, typical_period_duration,
                cycle_regularity, wellness_interests, onboarding_completed,
                bmi, sleep_hours, stress_score_baseline,
                state, diet_quality, exercise_frequency, caffeine_intake,
                water_intake_liters, alcohol_consumption, smoking_status,
                birth_control_use, pcos_diagnosed, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                preferred_name = excluded.preferred_name,
                age = excluded.age,
                date_of_birth = excluded.date_of_birth,
                track_cycle = excluded.track_cycle,
                last_period_date = excluded.last_period_date,
                typical_cycle_length = excluded.typical_cycle_length,
                typical_period_duration = excluded.typical_period_duration,
                cycle_regularity = excluded.cycle_regularity,
                wellness_interests = excluded.wellness_interests,
                onboarding_completed = excluded.onboarding_completed,
                bmi = excluded.bmi,
                sleep_hours = excluded.sleep_hours,
                stress_score_baseline = excluded.stress_score_baseline,
                state = excluded.state,
                diet_quality = excluded.diet_quality,
                exercise_frequency = excluded.exercise_frequency,
                caffeine_intake = excluded.caffeine_intake,
                water_intake_liters = excluded.water_intake_liters,
                alcohol_consumption = excluded.alcohol_consumption,
                smoking_status = excluded.smoking_status,
                birth_control_use = excluded.birth_control_use,
                pcos_diagnosed = excluded.pcos_diagnosed,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                merged.get("preferred_name"),
                merged.get("age"),
                merged.get("date_of_birth"),
                1 if merged.get("track_cycle", True) else 0,
                merged.get("last_period_date"),
                merged.get("typical_cycle_length"),
                merged.get("typical_period_duration"),
                merged.get("cycle_regularity"),
                wellness_json,
                1 if merged.get("onboarding_completed") else 0,
                merged.get("bmi"),
                merged.get("sleep_hours"),
                merged.get("stress_score_baseline"),
                merged.get("state"),
                merged.get("diet_quality"),
                merged.get("exercise_frequency"),
                merged.get("caffeine_intake"),
                merged.get("water_intake_liters"),
                merged.get("alcohol_consumption"),
                merged.get("smoking_status"),
                1 if merged.get("birth_control_use") else 0,
                1 if merged.get("pcos_diagnosed") else 0,
                now_utc,
            )
        )
    return True


def get_user_profile(user_id: str, db_path: Optional[Path] = None) -> Optional[UserProfileRecord]:
    """Retrieve user baseline profile strictly for the given user_id."""
    init_db(db_path)
    if not user_id:
        return None

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT user_id, preferred_name, age, date_of_birth, track_cycle,
                   last_period_date, typical_cycle_length, typical_period_duration,
                   cycle_regularity, wellness_interests, onboarding_completed,
                   bmi, sleep_hours, stress_score_baseline,
                   state, diet_quality, exercise_frequency, caffeine_intake,
                   water_intake_liters, alcohol_consumption, smoking_status,
                   birth_control_use, pcos_diagnosed, updated_at
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        # Parse wellness_interests JSON safely
        raw_interests = row["wellness_interests"]
        interests_list: List[str] = []
        if raw_interests:
            try:
                interests_list = json.loads(raw_interests)
                if not isinstance(interests_list, list):
                    interests_list = [str(raw_interests)]
            except Exception:
                interests_list = [item.strip() for item in str(raw_interests).split(",") if item.strip()]

        return UserProfileRecord(
            user_id=row["user_id"],
            preferred_name=row["preferred_name"],
            age=row["age"],
            date_of_birth=row["date_of_birth"],
            track_cycle=bool(row["track_cycle"]) if row["track_cycle"] is not None else True,
            last_period_date=row["last_period_date"],
            typical_cycle_length=row["typical_cycle_length"],
            typical_period_duration=row["typical_period_duration"],
            cycle_regularity=row["cycle_regularity"],
            wellness_interests=interests_list,
            onboarding_completed=bool(row["onboarding_completed"]) if row["onboarding_completed"] is not None else False,
            bmi=row["bmi"],
            sleep_hours=row["sleep_hours"],
            stress_score_baseline=row["stress_score_baseline"],
            state=row["state"],
            diet_quality=row["diet_quality"],
            exercise_frequency=row["exercise_frequency"],
            caffeine_intake=row["caffeine_intake"],
            water_intake_liters=row["water_intake_liters"],
            alcohol_consumption=row["alcohol_consumption"],
            smoking_status=row["smoking_status"],
            birth_control_use=row["birth_control_use"],
            pcos_diagnosed=row["pcos_diagnosed"],
            updated_at=row["updated_at"] or "",
        )


def has_completed_onboarding(user_id: str, db_path: Optional[Path] = None) -> bool:
    """Check whether a user has successfully finished their initial onboarding."""
    if not user_id:
        return False
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT onboarding_completed FROM user_profiles WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        if not row or row["onboarding_completed"] is None:
            return False
        return bool(row["onboarding_completed"])


# =============================================================================
# USER CYCLE LOGS (STRICT USER_ID ISOLATION)
# =============================================================================
@dataclass
class UserCycleLogRecord:
    """Represents a single historical cycle record for an authenticated user."""
    id: str
    user_id: str
    cycle_number: int
    cycle_length_days: Optional[float] = None
    cycle_phase: Optional[str] = None
    flow_level: Optional[str] = None
    pain_level: Optional[float] = None
    pms_symptoms: Optional[str] = None
    mood_score: Optional[float] = None
    stress_score_cycle: Optional[float] = None
    sleep_hours_cycle: Optional[float] = None
    energy_level: Optional[float] = None
    concentration_score: Optional[float] = None
    work_hours_lost: Optional[float] = None
    predicted_next_health_score: Optional[float] = None
    score_category: Optional[str] = None
    created_at: str = ""

    # Property aliases for seamless compatibility with both UI conventions and DB schemas
    @property
    def predicted_health_score(self) -> Optional[float]:
        return self.predicted_next_health_score

    @property
    def prediction_tier(self) -> Optional[str]:
        return self.score_category

    def __getitem__(self, key: str) -> Any:
        if key == "predicted_health_score":
            return self.predicted_next_health_score
        if key == "prediction_tier":
            return self.score_category
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "predicted_health_score":
            return self.predicted_next_health_score
        if key == "prediction_tier":
            return self.score_category
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in ("predicted_health_score", "prediction_tier")

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["predicted_health_score"] = self.predicted_next_health_score
        d["prediction_tier"] = self.score_category
        return d


def get_next_cycle_number(user_id: str, db_path: Optional[Path] = None) -> int:
    """Calculate the sequential cycle number for the user's next log entry."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT COALESCE(MAX(cycle_number), 0) + 1 AS next_cycle FROM user_cycle_logs WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return row["next_cycle"] if row else 1


def save_user_cycle_log(
    user_id: str,
    log_data: Optional[Dict[str, Any]] = None,
    cycle_data: Optional[Dict[str, Any]] = None,
    predicted_score: Optional[float] = None,
    prediction_tier: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Persist a single cycle record strictly owned by user_id."""
    init_db(db_path)
    if not user_id:
        raise OwnershipError("User ID must be provided to persist cycle records.")

    # Verify user exists
    user = get_user_by_id(user_id, db_path=db_path)
    if not user:
        raise UserNotFoundError(f"User '{user_id}' does not exist.")

    # Unify flexible calling conventions
    merged: Dict[str, Any] = {}
    if log_data:
        merged.update(log_data)
    if cycle_data:
        merged.update(cycle_data)
    if predicted_score is not None:
        merged["predicted_next_health_score"] = float(predicted_score)
    if prediction_tier is not None:
        merged["score_category"] = str(prediction_tier)

    # Normalize aliases if present
    if "predicted_health_score" in merged and "predicted_next_health_score" not in merged:
        merged["predicted_next_health_score"] = merged["predicted_health_score"]
    if "prediction_tier" in merged and "score_category" not in merged:
        merged["score_category"] = merged["prediction_tier"]

    log_id = f"log_{uuid.uuid4().hex[:16]}"
    now_utc = datetime.now(timezone.utc).isoformat()

    cycle_num = merged.get("cycle_number")
    if cycle_num is None:
        cycle_num = get_next_cycle_number(user_id, db_path=db_path)

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_cycle_logs (
                id, user_id, cycle_number, cycle_length_days, cycle_phase,
                flow_level, pain_level, pms_symptoms, mood_score,
                stress_score_cycle, sleep_hours_cycle, energy_level,
                concentration_score, work_hours_lost, predicted_next_health_score,
                score_category, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, cycle_number) DO UPDATE SET
                cycle_length_days = excluded.cycle_length_days,
                cycle_phase = excluded.cycle_phase,
                flow_level = excluded.flow_level,
                pain_level = excluded.pain_level,
                pms_symptoms = excluded.pms_symptoms,
                mood_score = excluded.mood_score,
                stress_score_cycle = excluded.stress_score_cycle,
                sleep_hours_cycle = excluded.sleep_hours_cycle,
                energy_level = excluded.energy_level,
                concentration_score = excluded.concentration_score,
                work_hours_lost = excluded.work_hours_lost,
                predicted_next_health_score = excluded.predicted_next_health_score,
                score_category = excluded.score_category,
                created_at = excluded.created_at
            """,
            (
                log_id,
                user_id,
                int(cycle_num),
                merged.get("cycle_length_days"),
                merged.get("cycle_phase"),
                merged.get("flow_level"),
                merged.get("pain_level"),
                merged.get("pms_symptoms"),
                merged.get("mood_score"),
                merged.get("stress_score_cycle"),
                merged.get("sleep_hours_cycle"),
                merged.get("energy_level"),
                merged.get("concentration_score"),
                merged.get("work_hours_lost"),
                merged.get("predicted_next_health_score"),
                merged.get("score_category"),
                now_utc,
            )
        )
    return log_id


def get_user_cycle_logs(
    user_id: str,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> List[UserCycleLogRecord]:
    """Retrieve all historical cycle logs strictly owned by user_id, ordered chronologically."""
    init_db(db_path)
    if not user_id:
        return []

    with get_db_connection(db_path) as conn:
        query = """
            SELECT id, user_id, cycle_number, cycle_length_days, cycle_phase,
                   flow_level, pain_level, pms_symptoms, mood_score,
                   stress_score_cycle, sleep_hours_cycle, energy_level,
                   concentration_score, work_hours_lost, predicted_next_health_score,
                   score_category, created_at
            FROM user_cycle_logs
            WHERE user_id = ?
            ORDER BY cycle_number ASC
        """
        params: List[Any] = [user_id]
        if limit is not None and isinstance(limit, int) and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [
            UserCycleLogRecord(
                id=row["id"],
                user_id=row["user_id"],
                cycle_number=row["cycle_number"],
                cycle_length_days=row["cycle_length_days"],
                cycle_phase=row["cycle_phase"],
                flow_level=row["flow_level"],
                pain_level=row["pain_level"],
                pms_symptoms=row["pms_symptoms"],
                mood_score=row["mood_score"],
                stress_score_cycle=row["stress_score_cycle"],
                sleep_hours_cycle=row["sleep_hours_cycle"],
                energy_level=row["energy_level"],
                concentration_score=row["concentration_score"],
                work_hours_lost=row["work_hours_lost"],
                predicted_next_health_score=row["predicted_next_health_score"],
                score_category=row["score_category"],
                created_at=row["created_at"] or "",
            )
            for row in rows
        ]


def get_user_dashboard_stats(
    user_id: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calculate personalized aggregated wellness and cycle metrics strictly for user_id."""
    init_db(db_path)
    if not user_id:
        return {
            "total_cycles": 0,
            "avg_cycle_length": None,
            "latest_cycle": None,
            "latest_health_score": None,
            "latest_health_tier": None,
            "avg_sleep": None,
            "avg_stress": None,
            "avg_pain": None,
            "days_since_last_period": None,
            "estimated_cycle_day": None,
            "estimated_phase": None,
            "profile": None,
        }

    profile = get_user_profile(user_id, db_path=db_path)
    logs = get_user_cycle_logs(user_id, db_path=db_path)

    total_cycles = len(logs)
    avg_cycle_length = None
    latest_cycle = logs[-1] if logs else None
    latest_health_score = latest_cycle.predicted_next_health_score if latest_cycle else None
    latest_health_tier = latest_cycle.score_category if latest_cycle else None

    # Calculate averages from logs if present
    valid_lengths = [l.cycle_length_days for l in logs if l.cycle_length_days is not None]
    if valid_lengths:
        avg_cycle_length = round(sum(valid_lengths) / len(valid_lengths), 1)
    elif profile and profile.typical_cycle_length:
        avg_cycle_length = float(profile.typical_cycle_length)

    valid_sleep = [l.sleep_hours_cycle for l in logs if l.sleep_hours_cycle is not None]
    avg_sleep = round(sum(valid_sleep) / len(valid_sleep), 1) if valid_sleep else (profile.sleep_hours if profile else None)

    valid_stress = [l.stress_score_cycle for l in logs if l.stress_score_cycle is not None]
    avg_stress = round(sum(valid_stress) / len(valid_stress), 1) if valid_stress else (profile.stress_score_baseline if profile else None)

    valid_pain = [l.pain_level for l in logs if l.pain_level is not None]
    avg_pain = round(sum(valid_pain) / len(valid_pain), 1) if valid_pain else None

    # Cycle phase estimation based on last_period_date
    days_since_last_period = None
    estimated_cycle_day = None
    estimated_phase = None

    if profile and profile.track_cycle and profile.last_period_date:
        try:
            last_date = datetime.strptime(profile.last_period_date, "%Y-%m-%d").date()
            today = date.today()
            delta = (today - last_date).days
            if 0 <= delta <= 120:
                days_since_last_period = delta
                estimated_cycle_day = delta + 1
                cycle_len = profile.typical_cycle_length or 28
                period_dur = profile.typical_period_duration or 5

                if estimated_cycle_day <= period_dur:
                    estimated_phase = "Menstrual"
                elif estimated_cycle_day <= max(period_dur + 1, int(cycle_len * 0.45)):
                    estimated_phase = "Follicular"
                elif estimated_cycle_day <= max(period_dur + 2, int(cycle_len * 0.55)):
                    estimated_phase = "Ovulatory"
                elif estimated_cycle_day <= cycle_len:
                    estimated_phase = "Luteal"
                else:
                    estimated_phase = "Late / Extended"
        except Exception:
            pass

    return {
        "total_cycles": total_cycles,
        "avg_cycle_length": avg_cycle_length,
        "latest_cycle": latest_cycle,
        "latest_health_score": latest_health_score,
        "latest_health_tier": latest_health_tier,
        "avg_sleep": avg_sleep,
        "avg_stress": avg_stress,
        "avg_pain": avg_pain,
        "days_since_last_period": days_since_last_period,
        "estimated_cycle_day": estimated_cycle_day,
        "estimated_phase": estimated_phase,
        "profile": profile,
    }


