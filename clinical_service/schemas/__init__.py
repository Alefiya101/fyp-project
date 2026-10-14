"""Schemas module for Clinical NLP and FHIR service."""

from __future__ import annotations

import json
from pathlib import Path

from clinical_service.schemas.interface1_transcript import Transcript, TranscriptSegment
from clinical_service.schemas.interface2_extraction import (
    ClinicalEntities,
    ClinicalExtraction,
    DiagnosisEntity,
    InvestigationEntity,
    MedicationEntity,
    SymptomEntity,
    VitalEntity,
)
from clinical_service.schemas.interface3_bundle import (
    GroundingFlag,
    ValidatedBundle,
    ValidationResult,
)

__all__ = [
    "ClinicalEntities",
    "ClinicalExtraction",
    "DiagnosisEntity",
    "GroundingFlag",
    "InvestigationEntity",
    "MedicationEntity",
    "SymptomEntity",
    "Transcript",
    "TranscriptSegment",
    "ValidatedBundle",
    "ValidationResult",
    "VitalEntity",
    "export_json_schemas",
]


def export_json_schemas(output_dir: str | Path) -> None:
    """Exports raw JSON Schema files for Interfaces 1, 2, and 3."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "interface1_transcript.schema.json").write_text(
        json.dumps(Transcript.model_json_schema(), indent=2), encoding="utf-8"
    )
    (out / "interface2_extraction.schema.json").write_text(
        json.dumps(ClinicalExtraction.model_json_schema(), indent=2), encoding="utf-8"
    )
    (out / "interface3_bundle.schema.json").write_text(
        json.dumps(ValidatedBundle.model_json_schema(), indent=2), encoding="utf-8"
    )
