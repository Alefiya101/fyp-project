"""Interface 3 - Validated Bundle JSON Schema.

Represents the contract between Person B (Clinical NLP & FHIR) and Person C
(Application, Integration & Evaluation).
Contains the generated FHIR R4 Bundle, schema validation results, and grounding flags.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GroundingFlag(BaseModel):
    entity: str = Field(
        ...,
        description="Path or identifier of the extracted entity (e.g. 'medications[0]', 'vitals[1]')."
    )
    issue: str = Field(
        ...,
        description="Explanation of the grounding violation or hallucination detected."
    )
    severity: str = Field(
        default="warning",
        description="Severity of the flag ('warning', 'error', 'info')."
    )
    offending_value: str | None = Field(
        default=None,
        description="The specific value that failed verification (e.g. '650 mg')."
    )
    source_segments: list[int] = Field(
        default_factory=list,
        description="Referenced segment indices checked during validation."
    )


class ValidationResult(BaseModel):
    schema_valid: bool = Field(
        ...,
        description="True if the FHIR bundle strictly conforms to FHIR R4 and ABDM profile."
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of schema validation errors or rule violations if any."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or recommendations."
    )


class ValidatedBundle(BaseModel):
    session_id: str = Field(..., min_length=1, description="Associated session identifier.")
    fhir_bundle: dict[str, Any] = Field(
        ...,
        description="The generated FHIR R4 document bundle adhering to ABDM OPConsultRecord profile."
    )
    validation: ValidationResult = Field(
        ...,
        description="Outcome of the FHIR schema and profile validation."
    )
    grounding_flags: list[GroundingFlag] = Field(
        default_factory=list,
        description="Hallucination detection and transcript-grounding warning flags."
    )
