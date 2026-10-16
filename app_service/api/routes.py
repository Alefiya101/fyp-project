"""FastAPI Router for Person C Application Endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app_service.abdm.sync_worker import sync_worker
from app_service.config import settings
from app_service.orchestrator.orchestrator import orchestrator
from app_service.storage.database import db
from app_service.storage.models import ApprovalDAO, DraftDAO, SessionDAO, TranscriptDAO
from clinical_service.schemas.interface1_transcript import Transcript
from clinical_service.schemas.interface2_extraction import ClinicalExtraction

router = APIRouter(prefix="/api", tags=["Ambient Clinical Scribe"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Optional custom session ID")
    clinician_id: str = Field(default="dr.sharma@opd", description="Identifier of consulting doctor")
    patient_id: str | None = Field(default=None, description="ABHA ID or OPD registration number")


class StartSessionResponse(BaseModel):
    session_id: str
    clinician_id: str
    patient_id: str | None
    status: str
    created_at: str


class ApproveSessionRequest(BaseModel):
    clinician_id: str = Field(..., description="Clinician ID confirming the approval")


class RejectSessionRequest(BaseModel):
    clinician_id: str = Field(..., description="Clinician ID rejecting the draft")
    reason: str = Field(default="Draft rejected by physician")


# ---------------------------------------------------------------------------
# Session Endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/start", response_model=StartSessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(req: StartSessionRequest):
    """Initializes a new OPD consultation session."""
    sid = req.session_id or f"OPD_{uuid.uuid4().hex[:8].upper()}"
    existing = SessionDAO.get_session(sid)
    if existing:
        raise HTTPException(status_code=400, detail=f"Session '{sid}' already exists")

    session = SessionDAO.create_session(sid, req.clinician_id, req.patient_id)
    return StartSessionResponse(
        session_id=session.session_id,
        clinician_id=session.clinician_id,
        patient_id=session.patient_identifier,
        status=session.status,
        created_at=session.created_at,
    )


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Fetches session metadata and status."""
    session = SessionDAO.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.model_dump()


@router.post("/sessions/{session_id}/upload-audio")
async def upload_audio(session_id: str, file: UploadFile = File(...), actor_id: str = Form("doctor")):
    """Uploads recorded OPD consultation audio and processes it end-to-end."""
    session = SessionDAO.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    dest_path = settings.audio_upload_dir / f"{session_id}_{file.filename}"
    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {exc}")

    try:
        validated_bundle = orchestrator.process_audio_file(session_id, dest_path, actor_id=actor_id)
        return {
            "session_id": session_id,
            "status": "DRAFT_READY",
            "schema_valid": validated_bundle.validation.schema_valid,
            "grounding_flags_count": len(validated_bundle.grounding_flags),
        }
    except Exception as exc:
        SessionDAO.update_status(session_id, "FAILED", actor_id=actor_id, details={"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {exc}")


@router.post("/sessions/{session_id}/transcript")
def upload_transcript(session_id: str, transcript_data: dict[str, Any], actor_id: str = Query("doctor")):
    """Ingests Interface 1 transcript JSON directly and runs clinical extraction & FHIR mapping."""
    session = SessionDAO.get_session(session_id)
    if not session:
        SessionDAO.create_session(session_id, actor_id)

    try:
        validated_bundle = orchestrator.process_transcript_input(session_id, transcript_data, actor_id=actor_id)
        return {
            "session_id": session_id,
            "status": "DRAFT_READY",
            "schema_valid": validated_bundle.validation.schema_valid,
            "grounding_flags_count": len(validated_bundle.grounding_flags),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process transcript: {exc}")


@router.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
    """Retrieves decrypted Interface 1 transcript for the consultation."""
    transcript = TranscriptDAO.get_transcript(session_id)
    if not transcript:
        raise HTTPException(status_code=404, detail=f"No transcript found for session '{session_id}'")
    return transcript.model_dump()


@router.get("/sessions/{session_id}/draft")
def get_draft(session_id: str):
    """Retrieves clinical draft (Interface 2 extraction, Interface 3 bundle, and grounding flags)."""
    draft = DraftDAO.get_draft(session_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"No draft found for session '{session_id}'")

    return {
        "session_id": session_id,
        "extraction": draft["extraction"].model_dump(),
        "fhir_bundle": draft["fhir_bundle"],
        "validation": draft["validation"].model_dump(),
        "grounding_flags": [f.model_dump() for f in draft["grounding_flags"]],
        "version": draft["version"],
        "updated_at": draft["updated_at"],
    }


@router.put("/sessions/{session_id}/draft")
def update_draft(session_id: str, extraction_data: dict[str, Any], actor_id: str = Query("doctor")):
    """Saves clinician edits to Interface 2 extraction, deterministically re-maps to FHIR, and re-validates."""
    try:
        validated_bundle = orchestrator.update_draft(session_id, extraction_data, actor_id=actor_id)
        return {
            "session_id": session_id,
            "message": "Draft updated and FHIR bundle re-mapped successfully",
            "schema_valid": validated_bundle.validation.schema_valid,
            "grounding_flags_count": len(validated_bundle.grounding_flags),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to update draft: {exc}")


@router.post("/sessions/{session_id}/approve")
def approve_session(session_id: str, req: ApproveSessionRequest):
    """Clinician approves consultation draft; seals cryptographic audit trail and enqueues to ABDM."""
    try:
        res = orchestrator.approve_session(session_id, req.clinician_id)
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Approval failed: {exc}")


@router.post("/sessions/{session_id}/reject")
def reject_session(session_id: str, req: RejectSessionRequest):
    """Clinician rejects draft."""
    try:
        orchestrator.reject_session(session_id, req.clinician_id, req.reason)
        return {"session_id": session_id, "status": "REJECTED", "reason": req.reason}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Rejection failed: {exc}")


@router.get("/sessions/{session_id}/audit")
def get_audit(session_id: str):
    """Returns tamper-evident audit trail for the encounter."""
    trail = db.get_audit_trail(session_id)
    return {"session_id": session_id, "audit_trail": trail}


# ---------------------------------------------------------------------------
# ABDM Sync Outbox Endpoints
# ---------------------------------------------------------------------------

@router.get("/sync/queue")
def get_sync_queue():
    """Returns current status of ABDM offline sync queue."""
    return sync_worker.get_queue_summary()


@router.post("/sync/trigger")
def trigger_sync():
    """Manually triggers background sync queue execution."""
    processed = sync_worker.process_pending_queue()
    return {"message": "Sync worker executed", "records_synced": processed}
