"""SQLite persistence layer for GeoStress AI.

Stores audit trail, model training history, expert RLHF preferences,
model version registry, drift baselines, and failure cases
in a durable SQLite database. Data survives server restarts.

Tables:
  audit_log           - Every analysis action with parameters and result hashes
  model_history       - Every ML model training run with metrics
  expert_preferences  - RLHF regime selections by geomechanics experts
  model_versions      - Model version registry with fingerprints and performance
  drift_baselines     - Feature distribution baselines for drift detection
  failure_cases       - Systematically collected prediction failures for learning
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


# Default database path — next to data/ directory
_DEFAULT_DB = Path(__file__).parent.parent / "data" / "geostress.db"

_local = threading.local()


def _get_conn(db_path: str = None) -> sqlite3.Connection:
    """Get a thread-local SQLite connection (one per thread)."""
    path = db_path or str(_DEFAULT_DB)
    if not hasattr(_local, "conn") or _local.conn is None or _local.db_path != path:
        _local.conn = sqlite3.connect(path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
        _local.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still safe
        _local.db_path = path
    return _local.conn


def init_db(db_path: str = None):
    """Create tables if they don't exist. Safe to call multiple times."""
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            source TEXT DEFAULT 'demo',
            well TEXT,
            parameters TEXT,
            result_hash TEXT,
            result_summary TEXT,
            elapsed_s REAL DEFAULT 0,
            app_version TEXT DEFAULT '3.2.0'
        );

        CREATE TABLE IF NOT EXISTS model_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            accuracy REAL,
            f1 REAL,
            n_samples INTEGER,
            n_features INTEGER,
            source TEXT DEFAULT 'demo',
            params TEXT,
            run_id TEXT UNIQUE
        );

        CREATE TABLE IF NOT EXISTS expert_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            well TEXT NOT NULL,
            depth_m REAL,
            regime TEXT,
            selected_regime TEXT NOT NULL,
            expert_confidence TEXT DEFAULT 'MODERATE',
            rationale TEXT,
            physics_regime TEXT,
            physics_confidence TEXT,
            solution_metrics TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
        CREATE INDEX IF NOT EXISTS idx_audit_well ON audit_log(well);
        CREATE INDEX IF NOT EXISTS idx_model_run_id ON model_history(run_id);
        CREATE INDEX IF NOT EXISTS idx_pref_well ON expert_preferences(well);
        CREATE INDEX IF NOT EXISTS idx_pref_regime ON expert_preferences(selected_regime);

        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            version INTEGER NOT NULL,
            model_type TEXT NOT NULL,
            well TEXT,
            accuracy REAL,
            f1 REAL,
            balanced_accuracy REAL,
            n_samples INTEGER,
            n_features INTEGER,
            data_fingerprint TEXT,
            hyperparams TEXT,
            feature_importances TEXT,
            is_active INTEGER DEFAULT 1,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS drift_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            well TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            mean REAL,
            std REAL,
            min_val REAL,
            max_val REAL,
            q25 REAL,
            q50 REAL,
            q75 REAL,
            n_samples INTEGER,
            histogram TEXT
        );

        CREATE TABLE IF NOT EXISTS failure_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            well TEXT,
            failure_type TEXT NOT NULL,
            description TEXT,
            depth_m REAL,
            azimuth REAL,
            dip REAL,
            predicted TEXT,
            actual TEXT,
            confidence REAL,
            context TEXT,
            root_cause TEXT,
            resolved INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_model_ver_active ON model_versions(is_active);
        CREATE INDEX IF NOT EXISTS idx_model_ver_well ON model_versions(well);
        CREATE INDEX IF NOT EXISTS idx_drift_well ON drift_baselines(well);
        CREATE INDEX IF NOT EXISTS idx_failure_well ON failure_cases(well);
        CREATE INDEX IF NOT EXISTS idx_failure_type ON failure_cases(failure_type);
        CREATE INDEX IF NOT EXISTS idx_failure_resolved ON failure_cases(resolved);
    """)
    conn.commit()


# ── Audit Log ────────────────────────────────────────

def insert_audit(action: str, source: str = "demo", well: str = None,
                 parameters: dict = None, result_hash: str = None,
                 result_summary: dict = None, elapsed_s: float = 0,
                 app_version: str = "3.2.0") -> int:
    """Insert an audit record and return its ID."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO audit_log
           (timestamp, action, source, well, parameters, result_hash,
            result_summary, elapsed_s, app_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            action, source, well,
            json.dumps(parameters, default=str) if parameters else None,
            result_hash,
            json.dumps(result_summary, default=str) if result_summary else None,
            round(elapsed_s, 2),
            app_version,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_audit_log(limit: int = 50, offset: int = 0,
                  well: str = None, action: str = None) -> list[dict]:
    """Retrieve audit records with optional filtering."""
    conn = _get_conn()
    query = "SELECT * FROM audit_log WHERE 1=1"
    params = []
    if well:
        query += " AND well = ?"
        params.append(well)
    if action:
        query += " AND action = ?"
        params.append(action)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        # Parse JSON fields back to dicts
        for field in ("parameters", "result_summary"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(d)
    return results


def count_audit(well: str = None, action: str = None) -> int:
    """Count audit records with optional filtering."""
    conn = _get_conn()
    query = "SELECT COUNT(*) FROM audit_log WHERE 1=1"
    params = []
    if well:
        query += " AND well = ?"
        params.append(well)
    if action:
        query += " AND action = ?"
        params.append(action)
    return conn.execute(query, params).fetchone()[0]


# ── Model History ────────────────────────────────────

def insert_model_history(model: str, accuracy: float, f1: float,
                         n_samples: int, n_features: int,
                         source: str = "demo", params: dict = None,
                         run_id: str = None) -> int:
    """Insert a model training record. Skips duplicates by run_id."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO model_history
               (timestamp, model, accuracy, f1, n_samples, n_features,
                source, params, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                model, round(accuracy, 4), round(f1, 4),
                n_samples, n_features, source,
                json.dumps(params, default=str) if params else None,
                run_id,
            ),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return 0  # Duplicate run_id


def get_model_history(limit: int = 50, model: str = None) -> list[dict]:
    """Retrieve model training history."""
    conn = _get_conn()
    query = "SELECT * FROM model_history WHERE 1=1"
    params = []
    if model:
        query += " AND model = ?"
        params.append(model)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("params"):
            try:
                d["params"] = json.loads(d["params"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


# ── Expert Preferences (RLHF) ───────────────────────

def insert_preference(well: str, selected_regime: str,
                      depth_m: float = None, regime: str = None,
                      expert_confidence: str = "MODERATE",
                      rationale: str = None, physics_regime: str = None,
                      physics_confidence: str = None,
                      solution_metrics: dict = None) -> int:
    """Record an expert regime selection."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO expert_preferences
           (timestamp, well, depth_m, regime, selected_regime,
            expert_confidence, rationale, physics_regime,
            physics_confidence, solution_metrics)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            well, depth_m, regime, selected_regime,
            expert_confidence, rationale, physics_regime,
            physics_confidence,
            json.dumps(solution_metrics, default=str) if solution_metrics else None,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_preferences(well: str = None, limit: int = 200) -> list[dict]:
    """Retrieve expert preferences with optional well filter."""
    conn = _get_conn()
    query = "SELECT * FROM expert_preferences WHERE 1=1"
    params = []
    if well:
        query += " AND well = ?"
        params.append(well)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("solution_metrics"):
            try:
                d["solution_metrics"] = json.loads(d["solution_metrics"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def count_preferences(well: str = None) -> int:
    """Count expert preferences."""
    conn = _get_conn()
    query = "SELECT COUNT(*) FROM expert_preferences WHERE 1=1"
    params = []
    if well:
        query += " AND well = ?"
        params.append(well)
    return conn.execute(query, params).fetchone()[0]


def clear_preferences(well: str = None) -> int:
    """Delete preferences, optionally filtered by well. Returns count deleted."""
    conn = _get_conn()
    if well:
        cur = conn.execute(
            "DELETE FROM expert_preferences WHERE well = ?", (well,)
        )
    else:
        cur = conn.execute("DELETE FROM expert_preferences")
    conn.commit()
    return cur.rowcount


# ── Export / Import ──────────────────────────────────

def export_all() -> dict:
    """Export entire database as JSON-serializable dict for backup."""
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "audit_log": get_audit_log(limit=10000),
        "model_history": get_model_history(limit=10000),
        "expert_preferences": get_preferences(limit=10000),
        "model_versions": get_model_versions(limit=10000),
        "failure_cases": get_failure_cases(limit=10000),
    }


def import_all(data: dict) -> dict:
    """Import records from a backup dict. Returns counts of imported records."""
    counts = {"audit": 0, "models": 0, "preferences": 0}
    conn = _get_conn()

    for rec in data.get("audit_log", []):
        try:
            conn.execute(
                """INSERT OR IGNORE INTO audit_log
                   (timestamp, action, source, well, parameters,
                    result_hash, result_summary, elapsed_s, app_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.get("timestamp", ""),
                    rec.get("action", ""),
                    rec.get("source", "demo"),
                    rec.get("well"),
                    json.dumps(rec.get("parameters"), default=str)
                        if rec.get("parameters") else None,
                    rec.get("result_hash"),
                    json.dumps(rec.get("result_summary"), default=str)
                        if rec.get("result_summary") else None,
                    rec.get("elapsed_s", 0),
                    rec.get("app_version", "3.2.0"),
                ),
            )
            counts["audit"] += 1
        except Exception:
            pass

    for rec in data.get("model_history", []):
        try:
            conn.execute(
                """INSERT OR IGNORE INTO model_history
                   (timestamp, model, accuracy, f1, n_samples,
                    n_features, source, params, run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.get("timestamp", ""),
                    rec.get("model", ""),
                    rec.get("accuracy", 0),
                    rec.get("f1", 0),
                    rec.get("n_samples", 0),
                    rec.get("n_features", 0),
                    rec.get("source", "demo"),
                    json.dumps(rec.get("params"), default=str)
                        if rec.get("params") else None,
                    rec.get("run_id"),
                ),
            )
            counts["models"] += 1
        except Exception:
            pass

    for rec in data.get("expert_preferences", []):
        try:
            conn.execute(
                """INSERT INTO expert_preferences
                   (timestamp, well, depth_m, regime, selected_regime,
                    expert_confidence, rationale, physics_regime,
                    physics_confidence, solution_metrics)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.get("timestamp", ""),
                    rec.get("well", ""),
                    rec.get("depth_m"),
                    rec.get("regime"),
                    rec.get("selected_regime", ""),
                    rec.get("expert_confidence", "MODERATE"),
                    rec.get("rationale"),
                    rec.get("physics_regime"),
                    rec.get("physics_confidence"),
                    json.dumps(rec.get("solution_metrics"), default=str)
                        if rec.get("solution_metrics") else None,
                ),
            )
            counts["preferences"] += 1
        except Exception:
            pass

    conn.commit()
    return counts


def db_stats() -> dict:
    """Return summary statistics about the database."""
    conn = _get_conn()
    return {
        "audit_count": conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0],
        "model_count": conn.execute("SELECT COUNT(*) FROM model_history").fetchone()[0],
        "preference_count": conn.execute("SELECT COUNT(*) FROM expert_preferences").fetchone()[0],
        "version_count": conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0],
        "drift_count": conn.execute("SELECT COUNT(*) FROM drift_baselines").fetchone()[0],
        "failure_count": conn.execute("SELECT COUNT(*) FROM failure_cases").fetchone()[0],
        "db_path": str(_DEFAULT_DB),
        "db_size_kb": round(_DEFAULT_DB.stat().st_size / 1024, 1) if _DEFAULT_DB.exists() else 0,
    }


# ── Model Versions ──────────────────────────────────

def insert_model_version(model_type: str, accuracy: float, f1: float,
                         n_samples: int, n_features: int,
                         well: str = None, balanced_accuracy: float = None,
                         data_fingerprint: str = None,
                         hyperparams: dict = None,
                         feature_importances: dict = None,
                         notes: str = None) -> int:
    """Register a new model version. Auto-increments version number."""
    conn = _get_conn()
    # Get next version number for this model_type + well
    cur = conn.execute(
        "SELECT MAX(version) FROM model_versions WHERE model_type = ? AND (well = ? OR (well IS NULL AND ? IS NULL))",
        (model_type, well, well),
    )
    max_ver = cur.fetchone()[0]
    next_ver = (max_ver or 0) + 1

    # Deactivate previous versions for this model+well
    conn.execute(
        "UPDATE model_versions SET is_active = 0 WHERE model_type = ? AND (well = ? OR (well IS NULL AND ? IS NULL))",
        (model_type, well, well),
    )

    cur = conn.execute(
        """INSERT INTO model_versions
           (timestamp, version, model_type, well, accuracy, f1, balanced_accuracy,
            n_samples, n_features, data_fingerprint, hyperparams,
            feature_importances, is_active, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            next_ver, model_type, well,
            round(accuracy, 4) if accuracy else None,
            round(f1, 4) if f1 else None,
            round(balanced_accuracy, 4) if balanced_accuracy else None,
            n_samples, n_features, data_fingerprint,
            json.dumps(hyperparams, default=str) if hyperparams else None,
            json.dumps(feature_importances, default=str) if feature_importances else None,
            notes,
        ),
    )
    conn.commit()
    return next_ver


def get_model_versions(model_type: str = None, well: str = None,
                       active_only: bool = False, limit: int = 50) -> list[dict]:
    """Retrieve model version history."""
    conn = _get_conn()
    query = "SELECT * FROM model_versions WHERE 1=1"
    params = []
    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)
    if well:
        query += " AND well = ?"
        params.append(well)
    if active_only:
        query += " AND is_active = 1"
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        for field in ("hyperparams", "feature_importances"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(d)
    return results


def rollback_model_version(model_type: str, target_version: int,
                           well: str = None) -> bool:
    """Rollback to a previous model version (set it as active)."""
    conn = _get_conn()
    # Deactivate all versions for this model+well
    conn.execute(
        "UPDATE model_versions SET is_active = 0 WHERE model_type = ? AND (well = ? OR (well IS NULL AND ? IS NULL))",
        (model_type, well, well),
    )
    # Activate the target version
    cur = conn.execute(
        "UPDATE model_versions SET is_active = 1 WHERE model_type = ? AND version = ? AND (well = ? OR (well IS NULL AND ? IS NULL))",
        (model_type, target_version, well, well),
    )
    conn.commit()
    return cur.rowcount > 0


# ── Drift Baselines ─────────────────────────────────

def save_drift_baseline(well: str, feature_stats: list[dict]) -> int:
    """Save feature distribution baseline for drift detection.

    feature_stats: list of dicts with keys:
        feature_name, mean, std, min_val, max_val, q25, q50, q75, n_samples, histogram
    """
    conn = _get_conn()
    # Clear old baseline for this well
    conn.execute("DELETE FROM drift_baselines WHERE well = ?", (well,))
    count = 0
    for fs in feature_stats:
        conn.execute(
            """INSERT INTO drift_baselines
               (timestamp, well, feature_name, mean, std, min_val, max_val,
                q25, q50, q75, n_samples, histogram)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                well, fs["feature_name"],
                fs.get("mean"), fs.get("std"),
                fs.get("min_val"), fs.get("max_val"),
                fs.get("q25"), fs.get("q50"), fs.get("q75"),
                fs.get("n_samples"),
                json.dumps(fs.get("histogram")) if fs.get("histogram") else None,
            ),
        )
        count += 1
    conn.commit()
    return count


def get_drift_baseline(well: str) -> list[dict]:
    """Retrieve stored drift baseline for a well."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM drift_baselines WHERE well = ? ORDER BY feature_name",
        (well,),
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("histogram"):
            try:
                d["histogram"] = json.loads(d["histogram"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


# ── Failure Cases ───────────────────────────────────

def insert_failure_case(failure_type: str, well: str = None,
                        description: str = None, depth_m: float = None,
                        azimuth: float = None, dip: float = None,
                        predicted: str = None, actual: str = None,
                        confidence: float = None, context: dict = None,
                        root_cause: str = None) -> int:
    """Record a prediction failure case for learning."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO failure_cases
           (timestamp, well, failure_type, description, depth_m,
            azimuth, dip, predicted, actual, confidence, context, root_cause)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            well, failure_type, description, depth_m,
            azimuth, dip, predicted, actual, confidence,
            json.dumps(context, default=str) if context else None,
            root_cause,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_failure_cases(well: str = None, failure_type: str = None,
                      resolved: bool = None, limit: int = 200) -> list[dict]:
    """Retrieve failure cases with optional filtering."""
    conn = _get_conn()
    query = "SELECT * FROM failure_cases WHERE 1=1"
    params = []
    if well:
        query += " AND well = ?"
        params.append(well)
    if failure_type:
        query += " AND failure_type = ?"
        params.append(failure_type)
    if resolved is not None:
        query += " AND resolved = ?"
        params.append(1 if resolved else 0)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("context"):
            try:
                d["context"] = json.loads(d["context"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(d)
    return results


def resolve_failure_case(case_id: int, root_cause: str = None) -> bool:
    """Mark a failure case as resolved with optional root cause."""
    conn = _get_conn()
    if root_cause:
        cur = conn.execute(
            "UPDATE failure_cases SET resolved = 1, root_cause = ? WHERE id = ?",
            (root_cause, case_id),
        )
    else:
        cur = conn.execute(
            "UPDATE failure_cases SET resolved = 1 WHERE id = ?",
            (case_id,),
        )
    conn.commit()
    return cur.rowcount > 0


def count_failure_cases(well: str = None, resolved: bool = None) -> int:
    """Count failure cases."""
    conn = _get_conn()
    query = "SELECT COUNT(*) FROM failure_cases WHERE 1=1"
    params = []
    if well:
        query += " AND well = ?"
        params.append(well)
    if resolved is not None:
        query += " AND resolved = ?"
        params.append(1 if resolved else 0)
    return conn.execute(query, params).fetchone()[0]
