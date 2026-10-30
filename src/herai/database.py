"""Persistent SQLite database manager for HERAI multi-user accounts and records.

Provides:
- Thread-safe connection management with SQLite WAL mode and foreign key enforcement.
- Parameterized SQL queries for all database operations to eliminate injection vulnerabilities.
- Strict user-level data isolation: profile and cycle operations require and verify user_id ownership.
- Comprehensive cascade cleanup on account deletion.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
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
            age REAL,
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
# USER PROFILE STORAGE & RETRIEVAL (STRICT USER_ID ISOLATION)
# =============================================================================
def save_user_profile(
    user_id: str,
    profile_data: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> bool:
    """Upsert user baseline profile data with strict user_id ownership enforcement."""
    init_db(db_path)
    if not user_id:
        raise OwnershipError("User ID must be provided to save profile data.")

    # Verify user exists
    user = get_user_by_id(user_id, db_path=db_path)
    if not user:
        raise UserNotFoundError(f"User '{user_id}' does not exist.")

    now_utc = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (
                user_id, age, bmi, sleep_hours, stress_score_baseline,
                state, diet_quality, exercise_frequency, caffeine_intake,
                water_intake_liters, alcohol_consumption, smoking_status,
                birth_control_use, pcos_diagnosed, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                age = excluded.age,
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
                profile_data.get("age"),
                profile_data.get("bmi"),
                profile_data.get("sleep_hours"),
                profile_data.get("stress_score_baseline"),
                profile_data.get("state"),
                profile_data.get("diet_quality"),
                profile_data.get("exercise_frequency"),
                profile_data.get("caffeine_intake"),
                profile_data.get("water_intake_liters"),
                profile_data.get("alcohol_consumption"),
                profile_data.get("smoking_status"),
                1 if profile_data.get("birth_control_use") else 0,
                1 if profile_data.get("pcos_diagnosed") else 0,
                now_utc,
            )
        )
    return True


def get_user_profile(user_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user baseline profile strictly for the given user_id."""
    init_db(db_path)
    if not user_id:
        return None

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT user_id, age, bmi, sleep_hours, stress_score_baseline,
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
        return dict(row)


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
