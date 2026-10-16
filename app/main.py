"""FastAPI Backend for Ambient OPD Clinical Scribing & ABDM FHIR Studio.

Chains Whisper ASR, Qwen2.5-1.5B LLM Refiner, and ABDM FHIR Mapper.
Guarantees NO transcript is returned or displayed in the UI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clinical_service.service import ClinicalService

logger = logging.getLogger(__name__)

# Workspace paths
ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = ROOT_DIR / "output"
RECORDINGS_DIR = ROOT_DIR / "recordings"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

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
    
    patient_info = {"id": "Unknown", "name": "Patient", "gender": "Unknown"}
    practitioner_info = {"id": "Unknown", "name": "Dr. Sharma", "role": "OPD Physician"}
    encounter_info = {"status": "finished", "class": "AMB", "type": "Outpatient Consultation"}
    
    symptoms = []
    medications = []
    vitals = []
    diagnoses = []
    
    for entry in entries:
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        
        if rtype == "Patient":
            patient_info["id"] = res.get("id", "patient-001")
            if "name" in res and res["name"]:
                text_name = res["name"][0].get("text")
                if text_name:
                    patient_info["name"] = text_name
            patient_info["gender"] = res.get("gender", "unknown")
            
        elif rtype == "Practitioner":
            practitioner_info["id"] = res.get("id", "practitioner-001")
            if "name" in res and res["name"]:
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
            
            # Check if this is symptom or diagnosis
            is_diag = any(c.get("code") == "encounter-diagnosis" for c in category_codings)
            item = {
                "name": code_text,
                "snomed_code": snomed_code,
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
            if "coding" in med_concept and med_concept["coding"]:
                rxnorm_code = med_concept["coding"][0].get("code")
                med_name = med_concept["coding"][0].get("display", med_name)
                
            dosage_instruction = res.get("dosageInstruction", [{}])[0]
            dosage_text = dosage_instruction.get("text", "")
            
            # Parse note if available
            note_text = ""
            if "note" in res and res["note"]:
                note_text = res["note"][0].get("text", "")
                
            medications.append({
                "name": med_name,
                "rxnorm_code": rxnorm_code,
                "dosage_instruction": dosage_text,
                "status": res.get("status", "active"),
                "note": note_text,
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
                "loinc_code": loinc_code,
                "value": val_str,
            })
            
    # Resource counts
    resource_counts = {}
    for entry in entries:
        rt = entry.get("resource", {}).get("resourceType", "Other")
        resource_counts[rt] = resource_counts.get(rt, 0) + 1

    return {
        "patient": patient_info,
        "practitioner": practitioner_info,
        "encounter": encounter_info,
        "symptoms": symptoms,
        "diagnoses": diagnoses,
        "medications": medications,
        "vitals": vitals,
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
        
    session_id = data.get("session_id", "SESSION_01")
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read consultation: {exc}")


@app.post("/api/process-audio")
async def process_audio_file(file: UploadFile = File(...)):
    """Receives audio file (.wav, .mp3), runs Whisper ASR + LLM Refiner + FHIR Mapper."""
    try:
        temp_wav = RECORDINGS_DIR / "live_consultation.wav"
        content = await file.read()
        temp_wav.write_bytes(content)
        
        service = ClinicalService(enable_llm_refiner=True)
        validated_bundle = service.process_audio(temp_wav, session_id="WEB_SESSION_01")
        
        # Persist to live_fhir_bundle.json
        out_path = OUTPUT_DIR / "live_fhir_bundle.json"
        out_path.write_text(validated_bundle.model_dump_json(indent=2), encoding="utf-8")
        
        return format_validated_bundle_response(validated_bundle)
    except Exception as exc:
        logger.exception("Audio processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Audio processing error: {exc}")


class RecordMicRequest(BaseModel):
    duration: float = 12.0


@app.post("/api/record-mic")
def record_from_laptop_mic(req: RecordMicRequest):
    """Directly records audio using the laptop's microphone via sounddevice."""
    try:
        import sounddevice as sd
        import soundfile as sf
        
        sample_rate = 16000
        frames = int(req.duration * sample_rate)
        
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        
        temp_wav = RECORDINGS_DIR / "live_consultation.wav"
        sf.write(str(temp_wav), audio.flatten(), sample_rate)
        
        service = ClinicalService(enable_llm_refiner=True)
        validated_bundle = service.process_audio(temp_wav, session_id="MIC_SESSION_01")
        
        out_path = OUTPUT_DIR / "live_fhir_bundle.json"
        out_path.write_text(validated_bundle.model_dump_json(indent=2), encoding="utf-8")
        
        return format_validated_bundle_response(validated_bundle)
    except Exception as exc:
        logger.exception("Mic recording failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Microphone recording error: {exc}")


# Serve static web frontend
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return JSONResponse({"message": "Frontend UI initializing..."})
    return FileResponse(str(index_file))
