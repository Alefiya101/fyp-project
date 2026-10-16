"""Production-grade SQLite Storage & Audit Layer.

Implements high-concurrency WAL mode, foreign key constraints, connection context management,
and schema initialization.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator

from app_service.config import settings
from app_service.storage.crypto import crypto


def get_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    """SQLite Database Manager with WAL mode and transaction support."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode / explicit transactions
        )
        conn.row_factory = sqlite3.Row
        # Enable high-concurrency PRAGMAs
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE;")
        try:
            yield cursor
            cursor.execute("COMMIT;")
        except Exception:
            cursor.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @contextlib.contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes relational schema for sessions, encrypted drafts, audit logs, and sync outbox."""
        with self.transaction() as cur:
            # 1. Sessions table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                clinician_id TEXT NOT NULL,
                patient_identifier TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 2. Transcripts table (Interface 1 encrypted)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                session_id TEXT PRIMARY KEY,
                encrypted_content TEXT NOT NULL,
                language_mix TEXT DEFAULT 'hi-en',
                segment_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """)

            # 3. Clinical Drafts table (Interface 2 & 3 encrypted)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS clinical_drafts (
                session_id TEXT PRIMARY KEY,
                encrypted_extraction TEXT NOT NULL,
                encrypted_bundle TEXT NOT NULL,
                schema_valid INTEGER NOT NULL,
                validation_json TEXT NOT NULL,
                grounding_flags_json TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """)

            # 4. Approved Records table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS approved_records (
                session_id TEXT PRIMARY KEY,
                clinician_id TEXT NOT NULL,
                encrypted_bundle TEXT NOT NULL,
                signature TEXT NOT NULL,
                approved_at TEXT NOT NULL,
                sync_status TEXT DEFAULT 'PENDING',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """)

            # 5. Tamper-evident Audit Logs table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                action TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details_json TEXT,
                signature TEXT NOT NULL
            );
            """)

            # 6. Outbox Sync Queue table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                payload_encrypted TEXT NOT NULL,
                status TEXT DEFAULT 'PENDING',
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 5,
                next_retry_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                synced_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            """)

            # Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sync_status ON sync_queue(status, next_retry_at);")

    # -----------------------------------------------------------------------
    # Audit log writer
    # -----------------------------------------------------------------------
    def log_audit(self, session_id: str | None, action: str, actor_id: str, details: dict[str, Any] | None = None) -> int:
        ts = get_now_iso()
        details_str = json.dumps(details or {}, ensure_ascii=False)
        # Compute tamper-evident signature
        sig_payload = f"{session_id or ''}|{action}|{actor_id}|{ts}|{details_str}"
        sig = crypto.sign_record(sig_payload)

        with self.transaction() as cur:
            cur.execute("""
            INSERT INTO audit_logs (session_id, action, actor_id, timestamp, details_json, signature)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (session_id, action, actor_id, ts, details_str, sig))
            return cur.lastrowid

    def get_audit_trail(self, session_id: str) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute("""
            SELECT id, session_id, action, actor_id, timestamp, details_json, signature
            FROM audit_logs
            WHERE session_id = ?
            ORDER BY id ASC;
            """, (session_id,))
            rows = cur.fetchall()

        results = []
        for r in rows:
            details_str = r["details_json"] or "{}"
            sig_payload = f"{r['session_id'] or ''}|{r['action']}|{r['actor_id']}|{r['timestamp']}|{details_str}"
            is_valid = crypto.verify_signature(sig_payload, r["signature"])
            results.append({
                "id": r["id"],
                "session_id": r["session_id"],
                "action": r["action"],
                "actor_id": r["actor_id"],
                "timestamp": r["timestamp"],
                "details": json.loads(details_str),
                "signature_valid": is_valid,
            })
        return results


# Global singleton
db = Database()
