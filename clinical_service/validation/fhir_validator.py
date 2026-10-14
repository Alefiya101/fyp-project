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

        schema_valid = len(errors) == 0
        return ValidationResult(schema_valid=schema_valid, errors=errors, warnings=warnings)
