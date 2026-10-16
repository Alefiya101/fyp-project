"""Comprehensive Unit and Integration Tests for Person C: Application, Storage & Sync."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app_service.abdm.client import ABDMClient
from app_service.abdm.sync_worker import SyncWorker
from app_service.main import app
from app_service.storage.crypto import CryptoManager
from app_service.storage.database import Database
from app_service.storage.models import ApprovalDAO, DraftDAO, SessionDAO, TranscriptDAO
from clinical_service.schemas.interface1_transcript import Transcript, TranscriptSegment
from clinical_service.schemas.interface2_extraction import (
    ClinicalEntities,
    ClinicalExtraction,
    MedicationEntity,
    SymptomEntity,
    VitalEntity,
)
from clinical_service.schemas.interface3_bundle import ValidatedBundle, ValidationResult


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite database and crypto manager for testing."""
    db_file = tmp_path / "test_ambient.db"
    test_db = Database(db_path=db_file)
    test_crypto = CryptoManager(key=b"12345678901234567890123456789012")
    return test_db, test_crypto


def test_crypto_manager(temp_db):
    _, cm = temp_db
    plaintext = "Paracetamol 650 mg TDS"
    enc = cm.encrypt_text(plaintext, associated_data="S001")
    assert enc != plaintext

    dec = cm.decrypt_text(enc, associated_data="S001")
    assert dec == plaintext

    # Tampered associated data should fail
    with pytest.raises(Exception):
        cm.decrypt_text(enc, associated_data="DIFFERENT_SESSION")

    # JSON round trip
    payload = {"symptom": "fever", "duration": "3 days"}
    enc_json = cm.encrypt_json(payload)
    dec_json = cm.decrypt_json(enc_json)
    assert dec_json == payload

    # Signature verification
    sig = cm.sign_record("OPD_AUDIT_LOG_ENTRY")
    assert cm.verify_signature("OPD_AUDIT_LOG_ENTRY", sig) is True
    assert cm.verify_signature("TAMPERED_ENTRY", sig) is False


def test_database_and_audit(temp_db):
    test_db, _ = temp_db

    # Test audit logging
    log_id = test_db.log_audit("OPD_TEST_1", "START_CONSULTATION", "dr.sharma", {"notes": "test"})
    assert log_id > 0

    trail = test_db.get_audit_trail("OPD_TEST_1")
    assert len(trail) == 1
    assert trail[0]["action"] == "START_CONSULTATION"
    assert trail[0]["signature_valid"] is True


def test_dao_and_draft_lifecycle(temp_db, monkeypatch):
    test_db, cm = temp_db
    monkeypatch.setattr("app_service.storage.models.db", test_db)
    monkeypatch.setattr("app_service.storage.models.crypto", cm)

    sid = "OPD_LIFECYCLE_01"
    # 1. Create session
    session = SessionDAO.create_session(sid, "dr.sharma@opd", "PATIENT_99")
    assert session.session_id == sid
    assert session.status == "CREATED"

    # 2. Save Transcript
    t = Transcript(
        session_id=sid,
        language_mix="hi-en",
        segments=[
            TranscriptSegment(start=0.0, end=3.0, speaker="doctor", text="Bukhar kab se hai?", lang="hi-en", confidence=0.9),
            TranscriptSegment(start=3.1, end=6.0, speaker="patient", text="2 din se hai.", lang="hi-en", confidence=0.88),
        ],
    )
    TranscriptDAO.save_transcript(t)
    loaded_t = TranscriptDAO.get_transcript(sid)
    assert loaded_t is not None
    assert len(loaded_t.segments) == 2
    assert loaded_t.segments[0].text == "Bukhar kab se hai?"

    # 3. Save Draft
    ext = ClinicalExtraction(
        session_id=sid,
        entities=ClinicalEntities(
            symptoms=[SymptomEntity(text="fever", duration="2 days", source_segments=[1])],
            medications=[MedicationEntity(name="Paracetamol", dose="500 mg", frequency="BD", source_segments=[0])],
        ),
    )
    bundle = {
        "resourceType": "Bundle",
        "type": "document",
        "id": f"bundle-{sid}",
        "entry": [],
    }
    val = ValidatedBundle(
        session_id=sid,
        fhir_bundle=bundle,
        validation=ValidationResult(schema_valid=True),
        grounding_flags=[],
    )
    DraftDAO.save_draft(sid, ext, val)

    loaded_draft = DraftDAO.get_draft(sid)
    assert loaded_draft is not None
    assert loaded_draft["extraction"].entities.symptoms[0].text == "fever"
    assert loaded_draft["validation"].schema_valid is True

    # 4. Approve Record
    sig = ApprovalDAO.approve_record(sid, "dr.sharma@opd", bundle)
    assert len(sig) == 64
    approved_bundle = ApprovalDAO.get_approved_bundle(sid)
    assert approved_bundle["resourceType"] == "Bundle"

    s_record = SessionDAO.get_session(sid)
    assert s_record.status == "APPROVED"


def test_abdm_client_and_sync_worker(temp_db, monkeypatch):
    test_db, cm = temp_db
    import app_service.storage.models as models_mod
    import app_service.abdm.sync_worker as sync_mod

    monkeypatch.setattr(models_mod, "db", test_db)
    monkeypatch.setattr(models_mod, "crypto", cm)
    monkeypatch.setattr(sync_mod, "db", test_db)
    monkeypatch.setattr(sync_mod, "crypto", cm)

    sid = "OPD_SYNC_TEST"
    SessionDAO.create_session(sid, "dr.sharma")
    bundle = {"resourceType": "Bundle", "type": "document", "id": f"bundle-{sid}", "entry": []}
    ApprovalDAO.approve_record(sid, "dr.sharma", bundle)

    client = ABDMClient(mock_mode=True)
    worker = SyncWorker(client=client)

    # Process pending queue
    processed = worker.process_pending_queue()
    assert processed == 1

    summary = worker.get_queue_summary()
    assert summary["completed"] == 1
    assert summary["pending"] == 0

    s = SessionDAO.get_session(sid)
    assert s.status == "SYNCED"


def test_fastapi_endpoints(monkeypatch, tmp_path):
    """Integration test of all FastAPI endpoints using TestClient."""
    test_db = Database(db_path=tmp_path / "api_test.db")
    import app_service.storage.database as db_mod
    import app_service.storage.models as models_mod
    import app_service.api.routes as routes_mod
    import app_service.orchestrator.orchestrator as orch_mod
    import app_service.abdm.sync_worker as sync_mod

    monkeypatch.setattr(db_mod, "db", test_db)
    monkeypatch.setattr(models_mod, "db", test_db)
    monkeypatch.setattr(routes_mod, "db", test_db)
    monkeypatch.setattr(orch_mod, "db", test_db)
    monkeypatch.setattr(sync_mod, "db", test_db)

    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. Start session
    sid = "OPD_API_001"
    res_start = client.post("/api/sessions/start", json={"session_id": sid, "clinician_id": "dr.verma"})
    assert res_start.status_code == 201
    assert res_start.json()["session_id"] == sid

    # 3. Post Transcript
    transcript_payload = {
        "session_id": sid,
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "doctor", "text": "Kya takleef hai?", "lang": "hi-en"},
            {"start": 3.1, "end": 7.0, "speaker": "patient", "text": "Teen din se bukhar hai. Khansi nahi hai.", "lang": "hi-en"},
            {"start": 7.1, "end": 10.0, "speaker": "doctor", "text": "Temperature 101.4 F hai.", "lang": "hi-en"},
            {"start": 10.1, "end": 14.0, "speaker": "doctor", "text": "Paracetamol 650 mg TDS lijiye 3 din tak.", "lang": "hi-en"},
        ],
    }
    res_trans = client.post(f"/api/sessions/{sid}/transcript", json=transcript_payload)
    assert res_trans.status_code == 200
    assert res_trans.json()["status"] == "DRAFT_READY"

    # 4. Get Draft
    res_draft = client.get(f"/api/sessions/{sid}/draft")
    assert res_draft.status_code == 200
    draft_data = res_draft.json()
    assert draft_data["validation"]["schema_valid"] is True
    assert len(draft_data["extraction"]["entities"]["symptoms"]) >= 1

    # Verify negation
    syms = draft_data["extraction"]["entities"]["symptoms"]
    cough = next((s for s in syms if s["text"] == "cough"), None)
    if cough:
        assert cough["negated"] is True

    # 5. In-line edit draft (e.g. edit dose to 500 mg)
    edited_extraction = draft_data["extraction"]
    for m in edited_extraction["entities"]["medications"]:
        if "Paracetamol" in m["name"]:
            m["dose"] = "500 mg"

    res_update = client.put(f"/api/sessions/{sid}/draft", json=edited_extraction)
    assert res_update.status_code == 200
    assert res_update.json()["schema_valid"] is True

    # 6. Approve session
    res_approve = client.post(f"/api/sessions/{sid}/approve", json={"clinician_id": "dr.verma"})
    assert res_approve.status_code == 200
    assert res_approve.json()["status"] == "APPROVED"

    # 7. Check Audit Trail
    res_audit = client.get(f"/api/sessions/{sid}/audit")
    assert res_audit.status_code == 200
    trail = res_audit.json()["audit_trail"]
    assert len(trail) >= 4
    assert any(a["action"] == "APPROVE_RECORD" for a in trail)

    # 8. Check Sync Queue
    res_queue = client.get("/api/sync/queue")
    assert res_queue.status_code == 200

    # 9. Test HTML UI Pages (Dashboard & Review)
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "OPD Consultations Dashboard" in res_dash.text

    res_review = client.get(f"/review/{sid}")
    assert res_review.status_code == 200
    assert "Clinician Review" in res_review.text
