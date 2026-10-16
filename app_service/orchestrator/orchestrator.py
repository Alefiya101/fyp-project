"""End-to-End Clinical Scribe Pipeline Orchestrator (Person C).

Chains together:
  Audio Upload / Ingestion
    -> Person A: SpeechService (Preprocessing, Diarization, Whisper ASR) -> Interface 1
    -> Person B: ClinicalService (Extraction, FHIR Mapping, Validation, Grounding) -> Interface 2 & 3
    -> Encrypted Storage & Audit
    -> Clinician Review & Re-mapping
    -> Cryptographic Approval & ABDM Outbox Enqueue
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app_service.abdm.sync_worker import sync_worker
from app_service.config import settings
from app_service.storage.database import db
from app_service.storage.models import ApprovalDAO, DraftDAO, SessionDAO, TranscriptDAO
from clinical_service.fhir_mapper.mapper import FHIRMapper
from clinical_service.schemas.interface1_transcript import Transcript
from clinical_service.schemas.interface2_extraction import ClinicalExtraction
from clinical_service.schemas.interface3_bundle import ValidatedBundle
from clinical_service.service import ClinicalService
from clinical_service.validation.fhir_validator import FHIRValidator
from clinical_service.validation.grounding_checker import GroundingChecker

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates consultation sessions, models, clinician edits, and approvals."""

    def __init__(self):
        self.clinical_service = ClinicalService(
            extractor_backend=settings.extractor_backend,
            model_path=settings.model_path,
        )
        self.fhir_mapper = FHIRMapper()
        self.fhir_validator = FHIRValidator()
        self.grounding_checker = GroundingChecker()

    def process_transcript_input(
        self,
        session_id: str,
        transcript_data: dict[str, Any] | Transcript,
        actor_id: str = "doctor",
    ) -> ValidatedBundle:
        if isinstance(transcript_data, dict):
            if "segments" in transcript_data and isinstance(transcript_data["segments"], list):
                transcript_data["segments"] = [
                    s for s in transcript_data["segments"]
                    if isinstance(s, dict) and s.get("text", "").strip()
                ]
            transcript = Transcript.model_validate(transcript_data)
        else:
            transcript = transcript_data

        SessionDAO.update_status(session_id, "PROCESSING", actor_id=actor_id)

        # 1. Store encrypted transcript
        TranscriptDAO.save_transcript(transcript, actor_id=actor_id)

        # 2. Run Person B Clinical Service
        extraction = self.clinical_service.extract_entities_only(transcript)
        validated_bundle = self.clinical_service.process_transcript(transcript)

        # 3. Store encrypted draft
        DraftDAO.save_draft(session_id, extraction, validated_bundle, actor_id=actor_id)

        # 4. Update session status
        SessionDAO.update_status(session_id, "DRAFT_READY", actor_id=actor_id)
        return validated_bundle

    def process_audio_file(
        self,
        session_id: str,
        audio_path: str | Path,
        actor_id: str = "doctor",
        person_a_only: bool = False,
    ) -> ValidatedBundle:
        """Runs Person A's SpeechService on audio, then processes transcript through Person B."""
        SessionDAO.update_status(session_id, "PROCESSING", actor_id=actor_id)

        p = Path(audio_path)
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            from speech_service.speech_service import SpeechService
            speech_svc = SpeechService()
            transcript_dict = speech_svc.process(str(p), session_id=session_id, person_a_only=person_a_only)
        except Exception as exc:
            logger.error("Person A speech_service failed for audio %s: %s", p, exc, exc_info=True)
            if settings.allow_mock_fallback:
                logger.warning("ALLOW_MOCK_FALLBACK is enabled. Using mock fallback transcript.")
                transcript_dict = {
                    "session_id": session_id,
                    "language_mix": "hi-en",
                    "segments": [
                        {"start": 0.0, "end": 2.5, "speaker": "doctor", "text": "Namaste, kya takleef hai?", "lang": "hi-en", "confidence": 0.95},
                        {"start": 2.6, "end": 6.5, "speaker": "patient", "text": "Doctor teen din se bukhar hai aur badan me dard hai. Khansi bilkul nahi hai.", "lang": "hi-en", "confidence": 0.92},
                        {"start": 6.6, "end": 9.5, "speaker": "doctor", "text": "Temperature 101.4 F hai.", "lang": "hi-en", "confidence": 0.96},
                        {"start": 9.6, "end": 14.0, "speaker": "doctor", "text": "Yeh viral fever hai. Paracetamol 650 mg TDS lijiye 3 din tak.", "lang": "hi-en", "confidence": 0.94},
                    ],
                }
            else:
                SessionDAO.update_status(session_id, "FAILED", actor_id=actor_id)
                raise RuntimeError(f"Speech transcription failed for audio '{p.name}': {exc}") from exc

        # Auto-delete audio if configured
        if settings.auto_delete_audio:
            try:
                p.unlink(missing_ok=True)
                db.log_audit(session_id, "AUTO_DELETE_AUDIO", actor_id, {"path": str(p)})
            except OSError as err:
                logger.warning("Could not delete audio file %s: %s", p, err)

        return self.process_transcript_input(session_id, transcript_dict, actor_id=actor_id)

    def update_draft(
        self,
        session_id: str,
        updated_extraction_data: dict[str, Any] | ClinicalExtraction,
        actor_id: str = "doctor",
    ) -> ValidatedBundle:
        """Handles clinician edits to Interface 2 extraction, re-maps to FHIR R4, and re-validates."""
        if isinstance(updated_extraction_data, dict):
            extraction = ClinicalExtraction.model_validate(updated_extraction_data)
        else:
            extraction = updated_extraction_data

        transcript = TranscriptDAO.get_transcript(session_id)
        if not transcript:
            raise ValueError(f"No transcript found for session {session_id}")

        # 1. Deterministic FHIR mapping of edited fields
        bundle_dict = self.fhir_mapper.map_to_fhir_bundle(extraction)

        # 2. Re-validate FHIR schema
        val_result = self.fhir_validator.validate_bundle(bundle_dict)

        # 3. Re-verify grounding against original transcript
        grounding_flags = self.grounding_checker.verify(transcript, extraction)

        validated_bundle = ValidatedBundle(
            session_id=session_id,
            fhir_bundle=bundle_dict,
            validation=val_result,
            grounding_flags=grounding_flags,
        )

        # 4. Save updated draft
        DraftDAO.save_draft(session_id, extraction, validated_bundle, actor_id=actor_id)
        db.log_audit(session_id, "EDIT_DRAFT", actor_id, {"entities_count": len(extraction.entities.symptoms) + len(extraction.entities.medications)})
        return validated_bundle

    def approve_session(
        self,
        session_id: str,
        clinician_id: str,
    ) -> dict[str, Any]:
        """Approves consultation draft, seals audit signature, and triggers sync worker."""
        draft_info = DraftDAO.get_draft(session_id)
        if not draft_info:
            raise ValueError(f"No clinical draft found for session {session_id}")

        final_bundle = draft_info["fhir_bundle"]

        # Seal cryptographic approval and enqueue to outbox
        signature = ApprovalDAO.approve_record(session_id, clinician_id, final_bundle)

        # Attempt immediate push if online
        synced_count = sync_worker.process_pending_queue()

        return {
            "session_id": session_id,
            "status": "APPROVED",
            "clinician_id": clinician_id,
            "signature": signature,
            "synced_immediately": synced_count > 0,
        }

    def reject_session(
        self,
        session_id: str,
        clinician_id: str,
        reason: str = "Rejected by clinician",
    ) -> None:
        """Rejects draft and closes session with audit log."""
        SessionDAO.update_status(session_id, "REJECTED", actor_id=clinician_id, details={"reason": reason})
        db.log_audit(session_id, "REJECT_SESSION", clinician_id, {"reason": reason})


# Singleton
orchestrator = Orchestrator()
