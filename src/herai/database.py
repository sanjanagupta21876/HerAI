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
    log_data: Dict[str, Any],
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

    log_id = f"log_{uuid.uuid4().hex[:16]}"
    now_utc = datetime.now(timezone.utc).isoformat()

    cycle_num = log_data.get("cycle_number")
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
                log_data.get("cycle_length_days"),
                log_data.get("cycle_phase"),
                log_data.get("flow_level"),
                log_data.get("pain_level"),
                log_data.get("pms_symptoms"),
                log_data.get("mood_score"),
                log_data.get("stress_score_cycle"),
                log_data.get("sleep_hours_cycle"),
                log_data.get("energy_level"),
                log_data.get("concentration_score"),
                log_data.get("work_hours_lost"),
                log_data.get("predicted_next_health_score"),
                log_data.get("score_category"),
                now_utc,
            )
        )
    return log_id


def get_user_cycle_logs(
    user_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all historical cycle logs strictly owned by user_id, ordered chronologically."""
    init_db(db_path)
    if not user_id:
        return []

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT id, user_id, cycle_number, cycle_length_days, cycle_phase,
                   flow_level, pain_level, pms_symptoms, mood_score,
                   stress_score_cycle, sleep_hours_cycle, energy_level,
                   concentration_score, work_hours_lost, predicted_next_health_score,
                   score_category, created_at
            FROM user_cycle_logs
            WHERE user_id = ?
            ORDER BY cycle_number ASC
            """,
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

