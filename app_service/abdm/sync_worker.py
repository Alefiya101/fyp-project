"""Offline Outbox Sync Worker.

Polls the persistent sync_queue for approved FHIR bundles, checks network connectivity,
applies exponential backoff on retries, and idempotently pushes to ABDM.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from app_service.abdm.client import abdm_client
from app_service.config import settings
from app_service.storage.crypto import crypto
from app_service.storage.database import db, get_now_iso

logger = logging.getLogger(__name__)


class SyncWorker:
    """Outbox pattern background sync worker with retry backoff."""

    def __init__(self, client=None, poll_interval: float | None = None):
        self.client = client or abdm_client
        self.poll_interval = poll_interval or settings.sync_poll_interval_sec
        self.max_retries = settings.sync_max_retries
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts background worker daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ABDM-Sync-Worker")
        self._thread.start()
        logger.info("ABDM Sync Worker started (poll interval: %.1fs)", self.poll_interval)

    def stop(self) -> None:
        """Stops background worker thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            logger.info("ABDM Sync Worker stopped")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.process_pending_queue()
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in SyncWorker loop: %s", exc)
            self._stop_event.wait(self.poll_interval)

    def process_pending_queue(self) -> int:
        """Processes all currently eligible pending or retry items in sync_queue. Returns processed count."""
        if not self.client.is_online():
            logger.debug("Network offline. Postponing ABDM sync queue execution.")
            return 0

        now_str = get_now_iso()
        with db.cursor() as cur:
            cur.execute("""
            SELECT queue_id, session_id, payload_encrypted, retry_count, max_retries
            FROM sync_queue
            WHERE status = 'PENDING'
               OR (status = 'FAILED' AND retry_count < max_retries AND (next_retry_at IS NULL OR next_retry_at <= ?))
            ORDER BY queue_id ASC
            LIMIT 10;
            """, (now_str,))
            items = cur.fetchall()

        if not items:
            return 0

        processed = 0
        for item in items:
            qid = item["queue_id"]
            sid = item["session_id"]
            retries = item["retry_count"]
            max_r = item["max_retries"]

            # Mark in-flight
            with db.transaction() as cur:
                cur.execute("UPDATE sync_queue SET status = 'IN_FLIGHT' WHERE queue_id = ?", (qid,))

            try:
                # Decrypt bundle
                fhir_bundle = crypto.decrypt_json(item["payload_encrypted"], associated_data=sid)
                # Push to ABDM
                res = self.client.push_op_consult_record(sid, fhir_bundle)

                # Update DB on success
                synced_at = get_now_iso()
                with db.transaction() as cur:
                    cur.execute("""
                    UPDATE sync_queue
                    SET status = 'COMPLETED', synced_at = ?, last_error = NULL
                    WHERE queue_id = ?;
                    """, (synced_at, qid))
                    cur.execute("""
                    UPDATE approved_records
                    SET sync_status = 'SYNCED'
                    WHERE session_id = ?;
                    """, (sid,))
                    cur.execute("""
                    UPDATE sessions
                    SET status = 'SYNCED', updated_at = ?
                    WHERE session_id = ?;
                    """, (synced_at, sid))

                db.log_audit(sid, "ABDM_SYNC_SUCCESS", "sync_worker", res)
                processed += 1
                logger.info("Successfully synced session '%s' to ABDM", sid)

            except Exception as exc:  # noqa: BLE001
                new_retries = retries + 1
                # Exponential backoff: 5s * (2 ^ retries)
                backoff_sec = min(300, 5 * (2 ** new_retries))
                next_retry = (datetime.now(UTC) + timedelta(seconds=backoff_sec)).isoformat()
                new_status = "FAILED" if new_retries >= max_r else "PENDING"

                with db.transaction() as cur:
                    cur.execute("""
                    UPDATE sync_queue
                    SET status = ?, retry_count = ?, next_retry_at = ?, last_error = ?
                    WHERE queue_id = ?;
                    """, (new_status, new_retries, next_retry, str(exc), qid))

                db.log_audit(sid, "ABDM_SYNC_FAILED", "sync_worker", {
                    "error": str(exc),
                    "retry_count": new_retries,
                    "next_retry_at": next_retry,
                })
                logger.warning("ABDM sync failed for '%s' (retry %d/%d): %s", sid, new_retries, max_r, exc)

        return processed

    def get_queue_summary(self) -> dict[str, Any]:
        """Returns statistics of the sync queue."""
        with db.cursor() as cur:
            cur.execute("""
            SELECT status, COUNT(*) as cnt
            FROM sync_queue
            GROUP BY status;
            """)
            rows = cur.fetchall()

        counts = {r["status"]: r["cnt"] for r in rows}
        return {
            "pending": counts.get("PENDING", 0),
            "in_flight": counts.get("IN_FLIGHT", 0),
            "completed": counts.get("COMPLETED", 0),
            "failed": counts.get("FAILED", 0),
            "online": self.client.is_online(),
            "mock_mode": self.client.mock_mode,
        }


# Singleton
sync_worker = SyncWorker()
