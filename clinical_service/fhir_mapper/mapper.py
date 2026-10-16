"""Deterministic FHIR R4 Mapper for ABDM OPConsultRecord.

Transforms Interface 2 ClinicalExtraction JSON into a valid FHIR R4 Document Bundle
strictly compliant with Ayushman Bharat Digital Mission (ABDM) specifications.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from fhir.resources.bundle import Bundle
from fhir.resources.composition import Composition
from fhir.resources.condition import Condition
from fhir.resources.encounter import Encounter
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from fhir.resources.practitioner import Practitioner
from fhir.resources.servicerequest import ServiceRequest

from clinical_service.fhir_mapper import abdm_profile as ap
from clinical_service.schemas.interface2_extraction import ClinicalExtraction
from clinical_service.terminology.lookup import get_terminology_lookup


def sanitize_id(raw_id: str) -> str:
    r"""Sanitizes strings to match FHIR id regex: [A-Za-z0-9\-.]{1,64}."""
    return re.sub(r"[^A-Za-z0-9\-.]", "-", str(raw_id))[:64]


class FHIRMapper:
    """Deterministic mapper from Interface 2 JSON to ABDM OPConsultRecord FHIR R4 Bundle."""

    def __init__(self):
        self.lookup = get_terminology_lookup()

    def map_to_fhir_bundle(
        self,
        extraction: ClinicalExtraction,
        patient_id: str | None = None,
        practitioner_id: str | None = None,
        encounter_id: str | None = None,
        consultation_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Maps an Interface 2 ClinicalExtraction into a validated FHIR R4 document bundle."""
        session_id = extraction.session_id
        now = consultation_time or datetime.now(UTC)
        iso_now = now.isoformat()

        # IDs sanitized for FHIR [A-Za-z0-9\-.]{1,64}
        bundle_id = sanitize_id(f"bundle-{session_id}")
        comp_id = sanitize_id(f"comp-{session_id}")
        pat_id = sanitize_id(patient_id or f"patient-{session_id}")
        doc_id = sanitize_id(practitioner_id or "practitioner-dr-sharma")
        enc_id = sanitize_id(encounter_id or f"encounter-{session_id}")

        # 1. Base Shared Resources
        patient_res = Patient.model_validate({
            "id": pat_id,
            "meta": {"profile": [ap.ABDM_PATIENT_PROFILE]},
            "name": [{"text": "OPD Patient", "use": "official"}],
            "gender": "unknown",
        })

        practitioner_res = Practitioner.model_validate({
            "id": doc_id,
            "meta": {"profile": [ap.ABDM_PRACTITIONER_PROFILE]},
            "name": [{"text": "Dr. Consulting Physician", "prefix": ["Dr."], "use": "official"}],
        })

        encounter_res = Encounter.model_validate({
            "id": enc_id,
            "meta": {"profile": [ap.ABDM_ENCOUNTER_PROFILE]},
            "status": "finished",
            "class": [{
                "coding": [{
                    "system": ap.SYSTEM_ACT_CODE,
                    "code": "AMB",
                    "display": "ambulatory",
                }]
            }],
            "subject": {"reference": f"Patient/{pat_id}"},
        })

        entries: list[dict[str, Any]] = []
        sections: list[dict[str, Any]] = []

        # 2. Symptoms -> Condition resources
        symptom_refs: list[dict[str, str]] = []
        for idx, sym in enumerate(extraction.entities.symptoms):
            cond_id = sanitize_id(f"condition-sym-{session_id}-{idx + 1}")
            snomed = self.lookup.lookup_snomed(sym.text)

            coding_list = []
            if snomed["code"]:
                coding_list.append({
                    "system": ap.SYSTEM_SNOMED,
                    "code": snomed["code"],
                    "display": snomed["display"],
                })

            ver_status_code = "refuted" if sym.negated else "confirmed"

            cond_dict = {
                "id": cond_id,
                "meta": {"profile": [ap.ABDM_CONDITION_PROFILE]},
                "clinicalStatus": {
                    "coding": [{"system": ap.SYSTEM_CONDITION_CLINICAL, "code": "active"}]
                },
                "verificationStatus": {
                    "coding": [{"system": ap.SYSTEM_CONDITION_VER_STATUS, "code": ver_status_code}]
                },
                "category": [{
                    "coding": [{
                        "system": ap.SYSTEM_SNOMED,
                        "code": "418799008",
                        "display": "Symptom",
                    }]
                }],
                "code": {
                    "coding": coding_list,
                    "text": sym.text + (" (Reported Absent)" if sym.negated else ""),
                },
                "subject": {"reference": f"Patient/{pat_id}"},
                "note": [{
                    "text": f"Duration: {sym.duration or 'Not specified'}; Negated: {sym.negated}; SourceSegments: {sym.source_segments}"
                }],
            }
            cond_res = Condition.model_validate(cond_dict)
            entries.append({"resource": cond_res.model_dump(exclude_none=True)})
            symptom_refs.append({"reference": f"Condition/{cond_id}"})

        if symptom_refs:
            sections.append({
                "title": ap.SECTION_CHIEF_COMPLAINTS_TITLE,
                "code": {
                    "coding": [{"system": ap.SYSTEM_SNOMED, "code": ap.SECTION_CHIEF_COMPLAINTS_CODE}]
                },
                "entry": symptom_refs,
            })

        # 3. Diagnoses -> Condition resources
        diag_refs: list[dict[str, str]] = []
        for idx, diag in enumerate(extraction.entities.diagnosis):
            cond_id = sanitize_id(f"condition-diag-{session_id}-{idx + 1}")
            snomed = self.lookup.lookup_snomed(diag.text)

            coding_list = []
            if snomed["code"]:
                coding_list.append({
                    "system": ap.SYSTEM_SNOMED,
                    "code": snomed["code"],
                    "display": snomed["display"],
                })

            ver_code = "confirmed" if diag.confidence == "high" else "provisional"

            cond_dict = {
                "id": cond_id,
                "meta": {"profile": [ap.ABDM_CONDITION_PROFILE]},
                "clinicalStatus": {
                    "coding": [{"system": ap.SYSTEM_CONDITION_CLINICAL, "code": "active"}]
                },
                "verificationStatus": {
                    "coding": [{"system": ap.SYSTEM_CONDITION_VER_STATUS, "code": ver_code}]
                },
                "category": [{
                    "coding": [{
                        "system": ap.SYSTEM_CONDITION_CATEGORY,
                        "code": "encounter-diagnosis",
                        "display": "Encounter Diagnosis",
                    }]
                }],
                "code": {
                    "coding": coding_list,
                    "text": diag.text,
                },
                "subject": {"reference": f"Patient/{pat_id}"},
                "note": [{
                    "text": f"Status: {diag.status}; Confidence: {diag.confidence}; SourceSegments: {diag.source_segments}"
                }],
            }
            diag_res = Condition.model_validate(cond_dict)
            entries.append({"resource": diag_res.model_dump(exclude_none=True)})
            diag_refs.append({"reference": f"Condition/{cond_id}"})

        if diag_refs:
            sections.append({
                "title": ap.SECTION_DIAGNOSIS_TITLE,
                "code": {
                    "coding": [{"system": ap.SYSTEM_SNOMED, "code": ap.SECTION_DIAGNOSIS_CODE}]
                },
                "entry": diag_refs,
            })

        # 4. Vitals -> Observation resources
        vital_refs: list[dict[str, str]] = []
        for idx, vit in enumerate(extraction.entities.vitals):
            obs_id = sanitize_id(f"obs-vital-{session_id}-{idx + 1}")
            loinc = self.lookup.lookup_loinc(vit.type)

            coding_list = []
            if loinc["code"]:
                coding_list.append({
                    "system": ap.SYSTEM_LOINC,
                    "code": loinc["code"],
                    "display": loinc["display"],
                })

            try:
                numeric_val = float(vit.value)
            except (ValueError, TypeError):
                numeric_val = None

            obs_dict: dict[str, Any] = {
                "id": obs_id,
                "meta": {"profile": [ap.ABDM_OBSERVATION_PROFILE]},
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": ap.SYSTEM_OBSERVATION_CATEGORY,
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }]
                }],
                "code": {
                    "coding": coding_list,
                    "text": vit.type.replace("_", " ").title(),
                },
                "subject": {"reference": f"Patient/{pat_id}"},
                "effectiveDateTime": iso_now,
                "note": [{"text": f"SourceSegments: {vit.source_segments}"}],
            }

            if numeric_val is not None:
                obs_dict["valueQuantity"] = {
                    "value": numeric_val,
                    "unit": vit.unit,
                    "system": ap.SYSTEM_UCUM,
                    "code": vit.unit,
                }
            else:
                obs_dict["valueString"] = str(vit.value)

            obs_res = Observation.model_validate(obs_dict)
            entries.append({"resource": obs_res.model_dump(exclude_none=True)})
            vital_refs.append({"reference": f"Observation/{obs_id}"})

        if vital_refs:
            sections.append({
                "title": ap.SECTION_VITALS_TITLE,
                "code": {
                    "coding": [{"system": ap.SYSTEM_SNOMED, "code": ap.SECTION_VITALS_CODE}]
                },
                "entry": vital_refs,
            })

        # 5. Medications -> MedicationRequest resources
        med_refs: list[dict[str, str]] = []
        for idx, med in enumerate(extraction.entities.medications):
            med_req_id = sanitize_id(f"med-req-{session_id}-{idx + 1}")
            rxnorm = self.lookup.lookup_rxnorm(med.name)

            coding_list = []
            if rxnorm["code"]:
                coding_list.append({
                    "system": ap.SYSTEM_RXNORM,
                    "code": rxnorm["code"],
                    "display": rxnorm["display"],
                })

            dosage_text = f"{med.name} {med.dose or ''} {med.frequency or ''} {med.duration or ''} {med.instructions or ''}".strip()

            med_dict = {
                "id": med_req_id,
                "meta": {"profile": [ap.ABDM_MEDICATION_REQUEST_PROFILE]},
                "status": "active",
                "intent": "order",
                "medication": {
                    "concept": {
                        "coding": coding_list,
                        "text": med.name,
                    }
                },
                "subject": {"reference": f"Patient/{pat_id}"},
                "authoredOn": iso_now,
                "dosageInstruction": [{
                    "text": dosage_text,
                    "patientInstruction": med.instructions or None,
                }],
                "note": [{
                    "text": f"Status: {med.status}; Frequency: {med.frequency}; Duration: {med.duration}; SourceSegments: {med.source_segments}"
                }],
            }
            med_res = MedicationRequest.model_validate(med_dict)
            entries.append({"resource": med_res.model_dump(exclude_none=True)})
            med_refs.append({"reference": f"MedicationRequest/{med_req_id}"})

        if med_refs:
            sections.append({
                "title": ap.SECTION_MEDICATIONS_TITLE,
                "code": {
                    "coding": [{"system": ap.SYSTEM_SNOMED, "code": ap.SECTION_MEDICATIONS_CODE}]
                },
                "entry": med_refs,
            })

        # 6. Investigations -> ServiceRequest resources
        inv_refs: list[dict[str, str]] = []
        for idx, inv in enumerate(extraction.entities.investigations):
            serv_id = sanitize_id(f"service-req-{session_id}-{idx + 1}")
            loinc = self.lookup.lookup_loinc(inv.name)

            coding_list = []
            if loinc["code"]:
                coding_list.append({
                    "system": ap.SYSTEM_LOINC,
                    "code": loinc["code"],
                    "display": loinc["display"],
                })

            serv_dict = {
                "id": serv_id,
                "meta": {"profile": [ap.ABDM_SERVICE_REQUEST_PROFILE]},
                "status": "active",
                "intent": "order",
                "code": {
                    "concept": {
                        "coding": coding_list,
                        "text": inv.name,
                    }
                },
                "subject": {"reference": f"Patient/{pat_id}"},
                "authoredOn": iso_now,
                "note": [{
                    "text": f"Status: {inv.status}; Result: {inv.result or 'Pending'}; SourceSegments: {inv.source_segments}"
                }],
            }
            serv_res = ServiceRequest.model_validate(serv_dict)
            entries.append({"resource": serv_res.model_dump(exclude_none=True)})
            inv_refs.append({"reference": f"ServiceRequest/{serv_id}"})

        if inv_refs:
            sections.append({
                "title": ap.SECTION_INVESTIGATIONS_TITLE,
                "code": {
                    "coding": [{"system": ap.SYSTEM_SNOMED, "code": ap.SECTION_INVESTIGATIONS_CODE}]
                },
                "entry": inv_refs,
            })

        # 7. Composition (Header of document bundle)
        comp_dict = {
            "id": comp_id,
            "meta": {"profile": [ap.ABDM_OP_CONSULT_PROFILE]},
            "status": "final",
            "type": {
                "coding": [{
                    "system": ap.SYSTEM_SNOMED,
                    "code": ap.OP_CONSULT_TYPE_CODE,
                    "display": ap.OP_CONSULT_TYPE_DISPLAY,
                }],
                "text": "Outpatient Consultation Note",
            },
            "category": [{
                "coding": [{
                    "system": ap.SYSTEM_SNOMED,
                    "code": ap.OP_CONSULT_TYPE_CODE,
                    "display": ap.OP_CONSULT_TYPE_DISPLAY,
                }]
            }],
            "subject": [{"reference": f"Patient/{pat_id}"}],
            "encounter": {"reference": f"Encounter/{enc_id}"},
            "date": iso_now,
            "author": [{"reference": f"Practitioner/{doc_id}"}],
            "title": "Outpatient Consultation Record",
            "section": sections,
        }
        comp_res = Composition.model_validate(comp_dict)

        # Assemble Full Document Bundle (First entry MUST be Composition)
        bundle_entries: list[dict[str, Any]] = [
            {"fullUrl": f"Composition/{comp_id}", "resource": comp_res.model_dump(exclude_none=True)},
            {"fullUrl": f"Patient/{pat_id}", "resource": patient_res.model_dump(exclude_none=True)},
            {"fullUrl": f"Practitioner/{doc_id}", "resource": practitioner_res.model_dump(exclude_none=True)},
            {"fullUrl": f"Encounter/{enc_id}", "resource": encounter_res.model_dump(exclude_none=True)},
        ]

        for item in entries:
            res_type = item["resource"].get("resourceType") or item["resource"]["id"].split("-")[0].capitalize()
            res_id = item["resource"]["id"]
            bundle_entries.append({
                "fullUrl": f"{res_type}/{res_id}",
                "resource": item["resource"],
            })

        bundle_dict = {
            "id": bundle_id,
            "resourceType": "Bundle",
            "type": "document",
            "timestamp": iso_now,
            "entry": bundle_entries,
        }

        # Validate with fhir.resources and export JSON-serializable dictionary
        bundle_obj = Bundle.model_validate(bundle_dict)
        return bundle_obj.model_dump(mode="json", exclude_none=True)
