"""FastAPI Backend for Ambient OPD Clinical Scribing & ABDM FHIR Studio.

Chains Whisper ASR, Qwen2.5-1.5B LLM Refiner, and ABDM FHIR Mapper.
Guarantees NO transcript is returned or displayed in the UI.
Allows clinicians to edit patient/practitioner info and add/remove symptoms, diagnoses, and prescriptions.
"""

from __future__ import annotations

import io
import json
import logging
import re
import threading
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.signal import resample_poly

from clinical_service.fhir_mapper.mapper import FHIRMapper
from clinical_service.schemas.interface2_extraction import (
    ClinicalEntities,
    ClinicalExtraction,
    DiagnosisEntity,
    InvestigationEntity,
    MedicationEntity,
    SymptomEntity,
    VitalEntity,
)
from clinical_service.service import ClinicalService
from clinical_service.validation.fhir_validator import FHIRValidator

logger = logging.getLogger(__name__)

# Workspace paths
ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = ROOT_DIR / "output"
RECORDINGS_DIR = ROOT_DIR / "recordings"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

_service_instance: ClinicalService | None = None
_service_lock = threading.Lock()


def get_clinical_service() -> ClinicalService:
    """Thread-safe singleton ClinicalService to avoid reloading weights repeatedly."""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = ClinicalService(enable_llm_refiner=True)
        return _service_instance


app = FastAPI(
    title="ABDM OPD Ambient Clinical Scribing Studio",
    description="Transforms Indian OPD Doctor-Patient audio into validated ABDM FHIR R4 Bundles using local SLMs.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_fhir_summary(bundle_dict: dict[str, Any]) -> dict[str, Any]:
    """Extracts clinician-friendly EMR views directly from the FHIR R4 Bundle.

    Strictly omits transcripts and raw speech segments.
    """
    entries = bundle_dict.get("entry", [])

    patient_info = {"id": "patient-001", "name": "OPD Patient", "gender": "Unknown"}
    practitioner_info = {"id": "practitioner-001", "name": "Dr. Consulting Physician", "role": "OPD Physician"}
    encounter_info = {"status": "finished", "class": "AMB", "type": "Outpatient Consultation"}

    symptoms = []
    medications = []
    vitals = []
    diagnoses = []
    investigations = []

    consultation_time = "10:02 AM"
    timestamp_str = bundle_dict.get("timestamp")
    if timestamp_str:
        try:
            dt = datetime.fromisoformat(timestamp_str)
            consultation_time = dt.strftime("%I:%M %p")
        except (ValueError, TypeError) as exc:
            logger.debug("Failed to parse consultation timestamp: %s", exc)

    for entry in entries:
        res = entry.get("resource", {})
        rtype = res.get("resourceType")

        if rtype == "Patient":
            patient_info["id"] = res.get("id", "patient-001")
            if res.get("name"):
                text_name = res["name"][0].get("text")
                if text_name:
                    patient_info["name"] = text_name
            patient_info["gender"] = res.get("gender", "unknown")

        elif rtype == "Practitioner":
            practitioner_info["id"] = res.get("id", "practitioner-001")
            if res.get("name"):
                text_name = res["name"][0].get("text")
                if text_name:
                    practitioner_info["name"] = text_name

        elif rtype == "Encounter":
            encounter_info["status"] = res.get("status", "finished")

        elif rtype == "Condition":
            category_codings = res.get("category", [{}])[0].get("coding", [{}])
            code_text = res.get("code", {}).get("text") or "Condition"
            snomed_code = None
            if "coding" in res.get("code", {}) and res["code"]["coding"]:
                snomed_code = res["code"]["coding"][0].get("code")
                code_text = res["code"]["coding"][0].get("display", code_text)

            recorded_date = res.get("recordedDate")

            # Check duration from onset or note
            duration = None
            onset_str = res.get("onsetString")
            if onset_str:
                duration = onset_str
            elif res.get("note"):
                note_str = res["note"][0].get("text", "")
                m = re.search(r"Duration:\s*([^;]+)", note_str)
                if m:
                    duration = m.group(1).strip()

            is_diag = any(c.get("code") == "encounter-diagnosis" for c in category_codings)
            item = {
                "name": code_text,
                "snomed_code": snomed_code or "386661006",
                "duration": duration or "3",
                "confidence": "high conf" if is_diag else None,
                "recorded_date": recorded_date,
                "clinical_status": res.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "active"),
            }
            if is_diag:
                diagnoses.append(item)
            else:
                symptoms.append(item)

        elif rtype == "MedicationRequest":
            med_concept = res.get("medicationCodeableConcept", {})
            med_name = med_concept.get("text") or "Medication"
            rxnorm_code = None
            if med_concept.get("coding"):
                rxnorm_code = med_concept["coding"][0].get("code")
                med_name = med_concept["coding"][0].get("display", med_name)

            dosage_instruction = res.get("dosageInstruction", [{}])[0]
            dosage_text = dosage_instruction.get("text", "")

            # Default values
            dosage = "500 mg"
            frequency = "TDS"
            duration = "3 days"
            status = res.get("status", "new")

            # Extract from note if available
            if res.get("note"):
                note_str = res["note"][0].get("text", "")
                freq_m = re.search(r"Frequency:\s*([^;]+)", note_str)
                if freq_m:
                    frequency = freq_m.group(1).strip()
                dur_m = re.search(r"Duration:\s*([^;]+)", note_str)
                if dur_m:
                    duration = dur_m.group(1).strip()
                stat_m = re.search(r"Status:\s*([^;]+)", note_str)
                if stat_m:
                    status = stat_m.group(1).strip()

            # Dose from dosage_text
            dose_m = re.search(r"(\d+\s*(?:mg|ml|tablet|goli|tab))", dosage_text, re.IGNORECASE)
            if dose_m:
                dosage = dose_m.group(1)

            medications.append({
                "name": med_name,
                "rxnorm_code": rxnorm_code or "RxNorm",
                "dosage": dosage,
                "frequency": frequency,
                "duration": duration,
                "status": status,
                "dosage_instruction": dosage_text,
            })

        elif rtype == "Observation":
            obs_code = res.get("code", {}).get("text") or "Observation"
            loinc_code = None
            if "coding" in res.get("code", {}) and res["code"]["coding"]:
                loinc_code = res["code"]["coding"][0].get("code")
                obs_code = res["code"]["coding"][0].get("display", obs_code)

            val_qty = res.get("valueQuantity", {})
            val_str = f"{val_qty.get('value', '')} {val_qty.get('unit', '')}".strip()
            vitals.append({
                "name": obs_code,
                "loinc_code": loinc_code or "LOINC",
                "value": val_str,
            })

        elif rtype == "ServiceRequest":
            code_text = res.get("code", {}).get("text") or "Laboratory Order"
            investigations.append(code_text)

    # Resource counts
    resource_counts = {}
    for entry in entries:
        rt = entry.get("resource", {}).get("resourceType", "Other")
        resource_counts[rt] = resource_counts.get(rt, 0) + 1

    return {
        "time": consultation_time,
        "patient": patient_info,
        "practitioner": practitioner_info,
        "encounter": encounter_info,
        "symptoms": symptoms,
        "diagnoses": diagnoses,
        "medications": medications,
        "vitals": vitals,
        "investigations": investigations,
        "resource_counts": resource_counts,
    }


def format_validated_bundle_response(bundle_obj_or_dict: Any) -> dict[str, Any]:
    """Formats standard response dictionary for frontend."""
    if hasattr(bundle_obj_or_dict, "model_dump"):
        data = bundle_obj_or_dict.model_dump()
    elif isinstance(bundle_obj_or_dict, dict):
        data = bundle_obj_or_dict
    else:
        raise ValueError("Invalid bundle representation")

    session_id = data.get("session_id", "live_consultation")
    fhir_bundle = data.get("fhir_bundle", {})
    validation = data.get("validation", {"schema_valid": True, "errors": [], "warnings": []})
    grounding_flags = data.get("grounding_flags", [])

    summary = parse_fhir_summary(fhir_bundle)

    return {
        "success": True,
        "session_id": session_id,
        "summary": summary,
        "validation": validation,
        "grounding_flags": grounding_flags,
        "fhir_bundle": fhir_bundle,
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "ABDM Clinical Scribing Studio",
        "llm_refiner": "Qwen2.5-1.5B-Instruct (Active)",
        "abdm_profile": "OPConsultRecord (NRCES FHIR R4)",
    }


@app.get("/api/latest")
def get_latest_consultation():
    """Returns the most recent consultation from output/live_fhir_bundle.json."""
    bundle_path = OUTPUT_DIR / "live_fhir_bundle.json"
    if not bundle_path.is_file():
        raise HTTPException(status_code=404, detail="No consultation recorded yet.")
    try:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        return format_validated_bundle_response(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read consultation: {exc}")


@app.post("/api/process-audio")
async def process_audio_file(file: Annotated[UploadFile, File()]):
    """Receives audio file (.wav, .mp3, etc.), runs Whisper ASR + LLM Refiner + FHIR Mapper."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        # Decode using load_audio (supports wav, mp3, m4a, flac, ogg via soundfile + PyAV)
        ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
        temp_raw = RECORDINGS_DIR / f"upload_raw{ext}"
        temp_raw.write_bytes(content)

        try:
            from speech_service.preprocessing.denoise import load_audio
            audio_data, sr = load_audio(str(temp_raw))
        except Exception:
            audio_data, sr = sf.read(str(temp_raw), dtype="float32", always_2d=False)

        # Convert to mono if multi-channel
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz mono PCM_16 WAV
        if sr != 16000:
            g = gcd(sr, 16000)
            audio_data = resample_poly(audio_data, 16000 // g, sr // g).astype(np.float32)

        temp_wav = RECORDINGS_DIR / "live_consultation.wav"
        sf.write(str(temp_wav), audio_data, 16000, subtype="PCM_16")

        service = get_clinical_service()
        validated_bundle = service.process_audio(temp_wav, session_id="live_consultation")

        # Persist to live_fhir_bundle.json
        out_path = OUTPUT_DIR / "live_fhir_bundle.json"
        out_path.write_text(validated_bundle.model_dump_json(indent=2), encoding="utf-8")

        return format_validated_bundle_response(validated_bundle)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Audio processing failed")
        raise HTTPException(status_code=500, detail=f"Audio processing error: {exc}")


class LiveMicRecorder:
    """Manages start/stop microphone stream recording for consultations of arbitrary length."""

    def __init__(self) -> None:
        self._stream: Any = None
        self._frames: list[np.ndarray] = []
        self._is_recording = False
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def start(self) -> None:
        import sounddevice as sd

        with self._lock:
            if self._is_recording:
                return
            self._frames = []
            self._is_recording = True

            def _callback(indata: np.ndarray, _frames: int, _time_info: Any, _status: Any) -> None:
                if self._is_recording:
                    self._frames.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                callback=_callback,
            )
            self._stream.start()

    def stop(self) -> np.ndarray:
        with self._lock:
            if not self._is_recording:
                return np.empty((0, 1), dtype=np.float32)
            self._is_recording = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("Error stopping microphone stream: %s", e)

        if self._frames:
            return np.concatenate(self._frames, axis=0)
        return np.empty((0, 1), dtype=np.float32)


live_recorder = LiveMicRecorder()


@app.post("/api/record/start")
def start_live_recording():
    """Starts recording ambient doctor-patient consultation from the microphone."""
    try:
        live_recorder.start()
        return {"success": True, "status": "recording"}
    except Exception as exc:
        logger.exception("Failed to start live recording")
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {exc}")


@app.post("/api/record/stop")
def stop_live_recording():
    """Stops consultation recording and processes audio with Whisper + Qwen2.5 LLM Refiner + FHIR Mapper."""
    try:
        import soundfile as sf

        audio_data = live_recorder.stop()
        if len(audio_data) < 1600:
            raise HTTPException(
                status_code=400,
                detail="Consultation recording too short or no audio detected.",
            )

        temp_wav = RECORDINGS_DIR / "live_consultation.wav"
        sf.write(str(temp_wav), audio_data.flatten(), 16000)

        service = get_clinical_service()
        validated_bundle = service.process_audio(temp_wav, session_id="live_consultation")

        out_path = OUTPUT_DIR / "live_fhir_bundle.json"
        out_path.write_text(validated_bundle.model_dump_json(indent=2), encoding="utf-8")

        return format_validated_bundle_response(validated_bundle)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Audio processing failed after stopping recording")
        raise HTTPException(status_code=500, detail=f"Microphone recording error: {exc}")


class SymptomItem(BaseModel):
    name: str
    duration: str | None = None
    status: str = "active"


class DiagnosisItem(BaseModel):
    name: str
    confidence: str | None = "high"
    status: str = "provisional"


class MedicationItem(BaseModel):
    name: str
    dosage: str = "500 mg"
    frequency: str = "TDS"
    duration: str = "3 days"
    status: str = "new"


class VitalItem(BaseModel):
    name: str
    value: str


class UpdateConsultationRequest(BaseModel):
    session_id: str = "live_consultation"
    patient_name: str = "OPD Patient"
    practitioner_name: str = "Dr. Consulting Physician"
    symptoms: list[SymptomItem] = []
    diagnoses: list[DiagnosisItem] = []
    medications: list[MedicationItem] = []
    vitals: list[VitalItem] = []
    investigations: list[str] = []


@app.post("/api/update-consultation")
def update_consultation_entities(req: UpdateConsultationRequest):
    """Allows doctor to add, edit, or delete clinical entities and update patient/doctor names.

    Regenerates a fully compliant ABDM FHIR R4 OPConsultRecord Bundle in real time.
    """
    try:
        symptom_entities = [
            SymptomEntity(
                text=s.name,
                duration=s.duration or "3 days",
                clinical_status=s.status or "active",
            )
            for s in req.symptoms
        ]
        diag_entities = [
            DiagnosisEntity(
                text=d.name,
                confidence="high",
                status="provisional",
            )
            for d in req.diagnoses
        ]
        valid_med_statuses = {"new", "existing", "discontinued", "modified", "changed", "self_taken"}
        med_entities = [
            MedicationEntity(
                name=m.name,
                dose=m.dosage,
                frequency=m.frequency,
                duration=m.duration,
                status=m.status.lower() if m.status and m.status.lower() in valid_med_statuses else "new",
            )
            for m in req.medications
        ]
        vital_entities = [
            VitalEntity(type=v.name, value=v.value)
            for v in req.vitals
        ]
        inv_entities = [
            InvestigationEntity(test_name=inv)
            for inv in req.investigations
        ]

        entities = ClinicalEntities(
            symptoms=symptom_entities,
            diagnosis=diag_entities,
            medications=med_entities,
            vitals=vital_entities,
            investigations=inv_entities,
        )
        extraction = ClinicalExtraction(session_id=req.session_id, entities=entities)

        mapper = FHIRMapper()
        bundle_dict = mapper.map_to_fhir_bundle(extraction)

        # Update patient & practitioner display names
        for entry in bundle_dict.get("entry", []):
            res = entry.get("resource", {})
            if res.get("resourceType") == "Patient":
                res["name"] = [{"text": req.patient_name}]
            elif res.get("resourceType") == "Practitioner":
                res["name"] = [{"text": req.practitioner_name}]

        validator = FHIRValidator()
        val_result = validator.validate_bundle(bundle_dict)

        # Read previous grounding flags if present
        existing_flags = []
        bundle_path = OUTPUT_DIR / "live_fhir_bundle.json"
        if bundle_path.is_file():
            try:
                prev_data = json.loads(bundle_path.read_text(encoding="utf-8"))
                existing_flags = prev_data.get("grounding_flags", [])
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("Could not read previous grounding flags: %s", exc)

        updated_payload = {
            "session_id": req.session_id,
            "fhir_bundle": bundle_dict,
            "validation": {
                "schema_valid": val_result.schema_valid,
                "errors": val_result.errors,
                "warnings": val_result.warnings,
            },
            "grounding_flags": existing_flags,
        }

        # Save to live_fhir_bundle.json
        bundle_path.write_text(json.dumps(updated_payload, indent=2, default=str), encoding="utf-8")

        return format_validated_bundle_response(updated_payload)
    except Exception as exc:
        logger.exception("Update consultation failed")
        raise HTTPException(status_code=500, detail=f"Failed to update consultation: {exc}")


# Serve static web frontend
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return JSONResponse({"message": "Frontend UI initializing..."})
    return FileResponse(str(index_file))
