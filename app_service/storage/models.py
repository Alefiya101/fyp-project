"""Data Access Objects (DAO) and Pydantic models for Person C Storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from app_service.storage.crypto import crypto
from app_service.storage.database import db, get_now_iso
from clinical_service.schemas.interface1_transcript import Transcript
from clinical_service.schemas.interface2_extraction import ClinicalExtraction
from clinical_service.schemas.interface3_bundle import GroundingFlag, ValidatedBundle, ValidationResult


class SessionRecord(BaseModel):
    session_id: str
    clinician_id: str
    patient_identifier: str | None = None
    status: Literal["CREATED", "RECORDING", "PROCESSING", "DRAFT_READY", "APPROVED", "REJECTED", "SYNCED"]
    created_at: str
    updated_at: str


class SessionDAO:
    """Operations for consultation sessions."""

    @staticmethod
    def create_session(session_id: str, clinician_id: str, patient_id: str | None = None) -> SessionRecord:
        now = get_now_iso()
        with db.transaction() as cur:
            cur.execute("""
            INSERT INTO sessions (session_id, clinician_id, patient_identifier, status, created_at, updated_at)
            VALUES (?, ?, ?, 'CREATED', ?, ?);
            """, (session_id, clinician_id, patient_id, now, now))
        db.log_audit(session_id, "CREATE_SESSION", clinician_id, {"patient_id": patient_id})
        return SessionRecord(
            session_id=session_id,
            clinician_id=clinician_id,
            patient_identifier=patient_id,
            status="CREATED",
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def get_session(session_id: str) -> SessionRecord | None:
        with db.cursor() as cur:
            cur.execute("SELECT session_id, clinician_id, patient_identifier, status, created_at, updated_at FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
        if not row:
            return None
        return SessionRecord(**dict(row))

    @staticmethod
    def update_status(session_id: str, status: str, actor_id: str = "system", details: dict | None = None) -> None:
        now = get_now_iso()
        with db.transaction() as cur:
            cur.execute("UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?", (status, now, session_id))
        db.log_audit(session_id, f"STATUS_CHANGE_{status}", actor_id, details or {})


class TranscriptDAO:
    """Operations for encrypted Interface 1 transcripts."""

    @staticmethod
    def save_transcript(transcript: Transcript, actor_id: str = "system") -> None:
        now = get_now_iso()
        encrypted = crypto.encrypt_json(transcript.model_dump(), associated_data=transcript.session_id)
        confs = [s.confidence for s in transcript.segments if s.confidence is not None]
        avg_conf = sum(confs) / len(confs) if confs else 1.0

        with db.transaction() as cur:
            cur.execute("""
            INSERT INTO transcripts (session_id, encrypted_content, language_mix, segment_count, avg_confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                encrypted_content = excluded.encrypted_content,
                language_mix = excluded.language_mix,
                segment_count = excluded.segment_count,
                avg_confidence = excluded.avg_confidence;
            """, (transcript.session_id, encrypted, transcript.language_mix, len(transcript.segments), avg_conf, now))
        db.log_audit(transcript.session_id, "STORE_TRANSCRIPT", actor_id, {
            "segments": len(transcript.segments),
            "avg_confidence": avg_conf,
        })

    @staticmethod
    def get_transcript(session_id: str) -> Transcript | None:
        with db.cursor() as cur:
            cur.execute("SELECT encrypted_content FROM transcripts WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
        if not row:
            return None
        data = crypto.decrypt_json(row["encrypted_content"], associated_data=session_id)
        return Transcript.model_validate(data)


class DraftDAO:
    """Operations for encrypted clinical drafts (Interface 2 & 3)."""

    @staticmethod
    def save_draft(
        session_id: str,
        extraction: ClinicalExtraction,
        validated_bundle: ValidatedBundle,
        actor_id: str = "system",
    ) -> None:
        now = get_now_iso()
        enc_extract = crypto.encrypt_json(extraction.model_dump(), associated_data=session_id)
        enc_bundle = crypto.encrypt_json(validated_bundle.fhir_bundle, associated_data=session_id)

        val_json = json.dumps(validated_bundle.validation.model_dump(), ensure_ascii=False)
        flags_json = json.dumps([f.model_dump() for f in validated_bundle.grounding_flags], ensure_ascii=False)

        with db.transaction() as cur:
            cur.execute("""
            INSERT INTO clinical_drafts (
                session_id, encrypted_extraction, encrypted_bundle,
                schema_valid, validation_json, grounding_flags_json,
                version, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                encrypted_extraction = excluded.encrypted_extraction,
                encrypted_bundle = excluded.encrypted_bundle,
                schema_valid = excluded.schema_valid,
                validation_json = excluded.validation_json,
                grounding_flags_json = excluded.grounding_flags_json,
                version = version + 1,
                updated_at = excluded.updated_at;
            """, (
                session_id, enc_extract, enc_bundle,
                1 if validated_bundle.validation.schema_valid else 0,
                val_json, flags_json, now, now
            ))
        db.log_audit(session_id, "SAVE_DRAFT", actor_id, {
            "schema_valid": validated_bundle.validation.schema_valid,
            "grounding_flags": len(validated_bundle.grounding_flags),
        })

    @staticmethod
    def get_draft(session_id: str) -> dict[str, Any] | None:
        with db.cursor() as cur:
            cur.execute("""
            SELECT encrypted_extraction, encrypted_bundle, schema_valid,
                   validation_json, grounding_flags_json, version, updated_at
            FROM clinical_drafts WHERE session_id = ?
            """, (session_id,))
            row = cur.fetchone()
        if not row:
            return None

        extraction_data = crypto.decrypt_json(row["encrypted_extraction"], associated_data=session_id)
        bundle_data = crypto.decrypt_json(row["encrypted_bundle"], associated_data=session_id)
        val_data = json.loads(row["validation_json"])
        flags_data = json.loads(row["grounding_flags_json"])

        return {
            "session_id": session_id,
            "extraction": ClinicalExtraction.model_validate(extraction_data),
            "fhir_bundle": bundle_data,
            "validation": ValidationResult.model_validate(val_data),
            "grounding_flags": [GroundingFlag.model_validate(f) for f in flags_data],
            "version": row["version"],
            "updated_at": row["updated_at"],
        }


class ApprovalDAO:
    """Operations for clinician approvals and audit sealing."""

    @staticmethod
    def approve_record(session_id: str, clinician_id: str, final_bundle: dict[str, Any]) -> str:
        now = get_now_iso()
        enc_bundle = crypto.encrypt_json(final_bundle, associated_data=session_id)
        # Cryptographic seal
        serialized = json.dumps(final_bundle, sort_keys=True, ensure_ascii=False)
        sig = crypto.sign_record(f"{session_id}|{clinician_id}|{now}|{serialized}")

        with db.transaction() as cur:
            # 1. Store approved record
            cur.execute("""
            INSERT INTO approved_records (session_id, clinician_id, encrypted_bundle, signature, approved_at, sync_status)
            VALUES (?, ?, ?, ?, ?, 'PENDING')
            ON CONFLICT(session_id) DO UPDATE SET
                clinician_id = excluded.clinician_id,
                encrypted_bundle = excluded.encrypted_bundle,
                signature = excluded.signature,
                approved_at = excluded.approved_at,
                sync_status = 'PENDING';
            """, (session_id, clinician_id, enc_bundle, sig, now))

            # 2. Update session status
            cur.execute("UPDATE sessions SET status = 'APPROVED', updated_at = ? WHERE session_id = ?", (now, session_id))

            # 3. Enqueue to Outbox Sync Queue
            cur.execute("""
            INSERT INTO sync_queue (session_id, payload_encrypted, status, retry_count, created_at)
            VALUES (?, ?, 'PENDING', 0, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                payload_encrypted = excluded.payload_encrypted,
                status = 'PENDING',
                retry_count = 0,
                last_error = NULL,
                synced_at = NULL;
            """, (session_id, enc_bundle, now))

        db.log_audit(session_id, "APPROVE_RECORD", clinician_id, {"signature": sig})
        return sig

    @staticmethod
    def get_approved_bundle(session_id: str) -> dict[str, Any] | None:
        with db.cursor() as cur:
            cur.execute("SELECT encrypted_bundle FROM approved_records WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
        if not row:
            return None
        return crypto.decrypt_json(row["encrypted_bundle"], associated_data=session_id)
