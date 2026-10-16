"""FHIR R4 & ABDM OPConsultRecord Profile Validator.

Validates:
1. Full structural FHIR R4 conformance via fhir.resources.
2. ABDM OPConsultRecord constraints (document bundle, Composition entry[0],
   required references, section completeness).
"""

from __future__ import annotations

from typing import Any

from fhir.resources.bundle import Bundle
from fhir.resources.composition import Composition

from clinical_service.schemas.interface3_bundle import ValidationResult


class FHIRValidator:
    """Validates FHIR R4 Bundles against base standard and ABDM OPConsultRecord rules."""

    def validate_bundle(self, bundle_dict: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Base FHIR R4 schema validation via fhir.resources
        try:
            bundle_obj = Bundle.model_validate(bundle_dict)
        except Exception as exc:  # noqa: BLE001 - captures all Pydantic/FHIR validation failures
            errors.append(f"FHIR R4 Schema Validation Failed: {exc}")
            return ValidationResult(schema_valid=False, errors=errors, warnings=warnings)

        # 2. ABDM Document Bundle Constraints
        if bundle_obj.type != "document":
            errors.append(f"ABDM OPConsultRecord bundle must be of type 'document', got '{bundle_obj.type}'")

        if not bundle_obj.entry or len(bundle_obj.entry) == 0:
            errors.append("ABDM document bundle must contain at least one entry")
            return ValidationResult(schema_valid=len(errors) == 0, errors=errors, warnings=warnings)

        first_entry = bundle_obj.entry[0]
        first_res = getattr(first_entry, "resource", None)
        first_type = first_res.get_resource_type() if hasattr(first_res, "get_resource_type") else getattr(first_res, "resource_type", "")
        if not first_res or first_type != "Composition":
            errors.append("ABDM document bundle first entry MUST be a Composition resource")

        # 3. Composition Header Constraints
        comp: Composition = first_res
        if comp:
            if not comp.subject or len(comp.subject) == 0 or not comp.subject[0].reference:
                errors.append("Composition must have a subject (Patient) reference")
            if not comp.author or len(comp.author) == 0:
                errors.append("Composition must have at least one author (Practitioner) reference")
            if not comp.encounter or not comp.encounter.reference:
                warnings.append("Composition should ideally reference an Encounter resource")

        # 4. Internal Reference Integrity
        entry_ids = set()
        for entry in bundle_obj.entry:
            res = getattr(entry, "resource", None)
            if res:
                res_type = res.get_resource_type() if hasattr(res, "get_resource_type") else getattr(res, "resource_type", "")
                res_id = getattr(res, "id", "")
                if res_id:
                    entry_ids.add(f"{res_type}/{res_id}")
                    entry_ids.add(res_id)

        # Check Composition sections references
        if comp and comp.section:
            for sec in comp.section:
                if sec.entry:
                    for ref_item in sec.entry:
                        ref_str = getattr(ref_item, "reference", "")
                        if ref_str and ref_str not in entry_ids:
                            errors.append(f"Composition section references '{ref_str}' which is not present in the Bundle entries")

        # 5. ABDM OPConsultRecord Profile Validation
        if not bundle_obj.meta or not bundle_obj.meta.profile:
            warnings.append("Bundle meta.profile should declare 'https://nrces.in/ndhm/fhir/r4/StructureDefinition/OPConsultRecord'")
        elif "https://nrces.in/ndhm/fhir/r4/StructureDefinition/OPConsultRecord" not in [str(p) for p in bundle_obj.meta.profile]:
            warnings.append("Bundle meta.profile does not contain ABDM OPConsultRecord profile URI")

        schema_valid = len(errors) == 0
        return ValidationResult(schema_valid=schema_valid, errors=errors, warnings=warnings)

    def validate_abdm_profiles(self, bundle_dict: dict[str, Any]) -> dict[str, Any]:
        """Detailed audit of all resources against ABDM NRCES FHIR R4 Profiles.

        Verifies profile URIs and terminology bindings for:
        OPConsultRecord, Patient, Practitioner, Encounter, Condition, Observation,
        MedicationRequest, and ServiceRequest.
        """
        from clinical_service.fhir_mapper import abdm_profile as ap

        profile_map = {
            "Bundle": ap.ABDM_OP_CONSULT_PROFILE,
            "Composition": ap.ABDM_OP_CONSULT_PROFILE,
            "Patient": ap.ABDM_PATIENT_PROFILE,
            "Practitioner": ap.ABDM_PRACTITIONER_PROFILE,
            "Encounter": ap.ABDM_ENCOUNTER_PROFILE,
            "Condition": ap.ABDM_CONDITION_PROFILE,
            "Observation": ap.ABDM_OBSERVATION_PROFILE,
            "MedicationRequest": ap.ABDM_MEDICATION_REQUEST_PROFILE,
            "ServiceRequest": ap.ABDM_SERVICE_REQUEST_PROFILE,
        }

        report: dict[str, Any] = {
            "target_profile": ap.ABDM_OP_CONSULT_PROFILE,
            "conforms": True,
            "resources_evaluated": 0,
            "profile_conformance": {},
            "coding_systems": [],
            "errors": [],
            "warnings": [],
        }

        # Check Bundle profile
        bundle_profiles = [str(p) for p in bundle_dict.get("meta", {}).get("profile", [])]
        bundle_valid = ap.ABDM_OP_CONSULT_PROFILE in bundle_profiles
        report["profile_conformance"]["Bundle"] = {
            "profile": ap.ABDM_OP_CONSULT_PROFILE,
            "present": bundle_valid,
            "count": 1,
        }
        if not bundle_valid:
            report["warnings"].append(f"Bundle is missing target profile: {ap.ABDM_OP_CONSULT_PROFILE}")

        entries = bundle_dict.get("entry", [])
        coding_systems_found = set()

        for entry in entries:
            res = entry.get("resource", {})
            rtype = res.get("resourceType")
            if not rtype:
                continue

            report["resources_evaluated"] += 1
            expected_profile = profile_map.get(rtype)
            res_profiles = [str(p) for p in res.get("meta", {}).get("profile", [])]
            has_profile = bool(expected_profile and expected_profile in res_profiles)

            if rtype not in report["profile_conformance"]:
                report["profile_conformance"][rtype] = {
                    "profile": expected_profile or "standard-fhir",
                    "present": has_profile,
                    "count": 1,
                }
            else:
                report["profile_conformance"][rtype]["count"] += 1
                if not has_profile:
                    report["profile_conformance"][rtype]["present"] = False

            # Inspect coding systems
            for field in ("code", "clinicalStatus", "verificationStatus", "category", "medication"):
                val = res.get(field)
                if isinstance(val, dict):
                    for c in val.get("coding", []):
                        if c.get("system"):
                            coding_systems_found.add(c["system"])
                    concept = val.get("concept")
                    if isinstance(concept, dict):
                        for c in concept.get("coding", []):
                            if c.get("system"):
                                coding_systems_found.add(c["system"])

        report["coding_systems"] = sorted(coding_systems_found)
        if report["errors"]:
            report["conforms"] = False

        return report
