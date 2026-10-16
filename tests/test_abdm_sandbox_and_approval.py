"""Tests for Doctor Approval/Rejection Flow and ABDM Sandbox Integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app as studio_app
from app_service.abdm.client import ABDMClient, abdm_client
from clinical_service.fhir_mapper.mapper import FHIRMapper
from clinical_service.schemas.interface2_extraction import (
    ClinicalEntities,
    ClinicalExtraction,
    DiagnosisEntity,
    MedicationEntity,
    SymptomEntity,
)
from clinical_service.validation.fhir_validator import FHIRValidator


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """Provides test client with isolated live_fhir_bundle.json."""
    test_out = tmp_path / "output"
    test_out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.main.OUTPUT_DIR", test_out)

    # Generate an authentic initial bundle
    extraction = ClinicalExtraction(
        session_id="test_consultation_001",
        entities=ClinicalEntities(
            symptoms=[SymptomEntity(text="fever", duration="3 days")],
            diagnosis=[DiagnosisEntity(text="common cold", confidence="high")],
            medications=[MedicationEntity(name="paracetamol", dose="650 mg", frequency="TDS", duration="3 days")],
        ),
    )
    mapper = FHIRMapper()
    bundle_dict = mapper.map_to_fhir_bundle(extraction)
    validator = FHIRValidator()
    val_result = validator.validate_bundle(bundle_dict)

    initial_payload = {
        "session_id": "test_consultation_001",
        "fhir_bundle": bundle_dict,
        "validation": {
            "schema_valid": val_result.schema_valid,
            "errors": val_result.errors,
            "warnings": val_result.warnings,
        },
        "grounding_flags": [],
        "approval": {
            "status": "DRAFT",
            "doctor_name": "Dr. Consulting Physician",
            "doctor_id": "practitioner-001",
            "approved_at": None,
            "signature": None,
            "notes": None,
        },
        "abdm_sync": {
            "status": "PENDING_APPROVAL",
            "mode": "MOCK_SANDBOX",
            "request_id": None,
            "care_context": "OPD-test_consultation_001",
            "synced_at": None,
        },
    }
    bundle_file = test_out / "live_fhir_bundle.json"
    bundle_file.write_text(json.dumps(initial_payload, indent=2), encoding="utf-8")

    return TestClient(studio_app)


def test_abdm_profile_validation():
    """Verifies that FHIRMapper produces bundles strictly conformant with ABDM NRCES profiles."""
    extraction = ClinicalExtraction(
        session_id="abdm_profile_test",
        entities=ClinicalEntities(
            symptoms=[SymptomEntity(text="bukhar", duration="2 days")],
            diagnosis=[DiagnosisEntity(text="viral infection", confidence="high")],
            medications=[MedicationEntity(name="azithromycin", dose="500 mg", frequency="OD", duration="3 days")],
        ),
    )
    mapper = FHIRMapper()
    bundle = mapper.map_to_fhir_bundle(extraction)

    validator = FHIRValidator()
    # 1. Base validation
    res = validator.validate_bundle(bundle)
    assert res.schema_valid is True
    assert len(res.errors) == 0

    # 2. Detailed ABDM NRCES profile audit
    report = validator.validate_abdm_profiles(bundle)
    assert report["conforms"] is True
    assert report["profile_conformance"]["Bundle"]["present"] is True
    assert report["profile_conformance"]["Composition"]["present"] is True
    assert report["profile_conformance"]["Patient"]["present"] is True
    assert report["profile_conformance"]["Practitioner"]["present"] is True
    assert report["profile_conformance"]["Encounter"]["present"] is True
    assert report["profile_conformance"]["Condition"]["present"] is True
    assert report["profile_conformance"]["MedicationRequest"]["present"] is True
    assert "http://snomed.info/sct" in report["coding_systems"]


def test_abdm_sandbox_client():
    """Verifies ABDMClient mock sandbox and real gateway endpoints."""
    client = ABDMClient(mock_mode=True)
    assert client.is_online() is True

    test_conn = client.test_connection()
    assert test_conn["status"] == "ONLINE"
    assert test_conn["authenticated"] is True

    # Test Care Context notification (Milestone 2)
    notify_res = client.notify_care_context("test_session_101", "patient-001")
    assert notify_res["status"] == "ACKNOWLEDGED"
    assert "OPD-test_session_101" in notify_res["careContextReference"]

    # Test Push (Milestone 3)
    bundle = {
        "resourceType": "Bundle",
        "type": "document",
        "id": "bundle-test",
        "meta": {"profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/OPConsultRecord"]},
        "entry": [],
    }
    push_res = client.push_op_consult_record("test_session_101", bundle, "patient-001")
    assert push_res["status"] == "SUCCESS"
    assert push_res["mode"] == "MOCK_SANDBOX"
    assert "requestId" in push_res


def test_doctor_approve_flow(test_client):
    """Doctor approves consultation record, generating digital seal and ABDM Sandbox sync."""
    client = test_client

    # 1. Read latest consultation -> status should be DRAFT
    res_latest = client.get("/api/latest")
    assert res_latest.status_code == 200
    latest_data = res_latest.json()
    assert latest_data["approval"]["status"] == "DRAFT"

    # 2. Doctor approves the record
    approve_payload = {
        "session_id": "test_consultation_001",
        "doctor_name": "Dr. Anita Desai",
        "doctor_id": "practitioner-dr-desai",
        "notes": "Verified clinical details with patient",
    }
    res_approve = client.post("/api/approve", json=approve_payload)
    assert res_approve.status_code == 200
    appr_data = res_approve.json()

    assert appr_data["success"] is True
    assert appr_data["approval"]["status"] == "APPROVED"
    assert appr_data["approval"]["doctor_name"] == "Dr. Anita Desai"
    assert len(appr_data["approval"]["signature"]) == 64
    assert appr_data["abdm_sync"]["status"] == "SUCCESS"
    assert "OPD-test_consultation_001" in appr_data["abdm_sync"]["care_context"]

    # 3. Subsequent call to /api/latest reflects APPROVED status
    res_latest_after = client.get("/api/latest")
    assert res_latest_after.json()["approval"]["status"] == "APPROVED"


def test_doctor_reject_flow(test_client):
    """Doctor rejects consultation record with reason, halting ABDM sync."""
    client = test_client

    reject_payload = {
        "session_id": "test_consultation_001",
        "doctor_name": "Dr. Anita Desai",
        "doctor_id": "practitioner-dr-desai",
        "reason": "Speech recognition missed secondary symptom of headache",
    }
    res_reject = client.post("/api/reject", json=reject_payload)
    assert res_reject.status_code == 200
    rej_data = res_reject.json()

    assert rej_data["success"] is True
    assert rej_data["approval"]["status"] == "REJECTED"
    assert rej_data["approval"]["reason"] == "Speech recognition missed secondary symptom of headache"
    assert rej_data["abdm_sync"]["status"] == "REJECTED"


def test_abdm_status_and_validate_endpoints(test_client):
    """Verifies /api/abdm/status, /api/abdm/test, and /api/validate-bundle endpoints."""
    client = test_client

    # Status
    res_status = client.get("/api/abdm/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "status" in status_data
    assert "gateway_url" in status_data

    # Test connection
    res_test = client.post("/api/abdm/test")
    assert res_test.status_code == 200
    assert "latency_ms" in res_test.json()

    # Validate bundle
    res_val = client.get("/api/validate-bundle")
    assert res_val.status_code == 200
    val_data = res_val.json()
    assert val_data["schema_valid"] is True
    assert val_data["abdm_profile_conformance"]["conforms"] is True
