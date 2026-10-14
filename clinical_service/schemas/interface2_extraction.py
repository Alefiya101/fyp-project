"""Interface 2 - Extraction JSON Schema.

Represents the contract for structured clinical extraction output produced by
Person B's SLM / extraction pipeline. This is read internally by Person B's FHIR mapper,
and also consumed by Person C (Application) for UI review and highlighting.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SymptomEntity(BaseModel):
    text: str = Field(..., min_length=1, description="Symptom name or clinical complaint.")
    duration: str | None = Field(
        default=None,
        description="Duration of the symptom (e.g. '3 days', '1 week', '2 months')."
    )
    severity: str | None = Field(
        default=None,
        description="Reported severity (e.g. 'mild', 'moderate', 'severe', 'high-grade')."
    )
    negated: bool = Field(
        default=False,
        description="True if symptom is explicitly ruled out (e.g. 'no cough', 'chhati me dard nahi hai')."
    )
    source_segments: list[int] = Field(
        default_factory=list,
        description="List of transcript segment indices that justify this extraction."
    )


class DiagnosisEntity(BaseModel):
    text: str = Field(..., min_length=1, description="Diagnosed clinical condition or provisional impression.")
    confidence: Literal["high", "medium", "low", "provisional", "differential"] = Field(
        default="medium",
        description="Clinical confidence level expressed by doctor."
    )
    status: Literal["active", "provisional", "resolved", "recurrence"] = Field(
        default="provisional",
        description="Clinical status of the condition."
    )
    source_segments: list[int] = Field(
        default_factory=list,
        description="Transcript segment indices supporting this diagnosis."
    )


class MedicationEntity(BaseModel):
    name: str = Field(..., min_length=1, description="Medication brand or generic name.")
    dose: str | None = Field(
        default=None,
        description="Dose amount with unit (e.g. '500 mg', '650 mg', '1 tablet', '2 puffs')."
    )
    frequency: str | None = Field(
        default=None,
        description="Administration schedule normalized or as stated (e.g. 'OD', 'BD', 'TDS', 'QID', 'twice a day', 'SOS')."
    )
    duration: str | None = Field(
        default=None,
        description="Prescribed duration (e.g. '3 days', '5 days', '1 month', 'continuous')."
    )
    status: Literal["new", "existing", "discontinued", "modified"] = Field(
        default="new",
        description="Whether this is a newly prescribed medication or ongoing/existing home medication."
    )
    instructions: str | None = Field(
        default=None,
        description="Specific timing instructions (e.g. 'after food', 'before food', 'at bedtime')."
    )
    source_segments: list[int] = Field(
        default_factory=list,
        description="Transcript segment indices where medication was mentioned."
    )


class VitalEntity(BaseModel):
    type: str = Field(
        ...,
        description="Type of vital sign: 'temperature', 'pulse', 'blood_pressure_sys', 'blood_pressure_dia', 'spo2', 'respiratory_rate', 'weight'."
    )
    value: float | int | str = Field(
        ...,
        description="Recorded numerical or string value (e.g. 38.5, 98.6, 120, 80)."
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (e.g. 'Cel', 'degF', 'mm[Hg]', '/min', '%', 'kg')."
    )
    source_segments: list[int] = Field(
        default_factory=list,
        description="Transcript segment indices where vital sign was recorded."
    )


class InvestigationEntity(BaseModel):
    name: str = Field(..., min_length=1, description="Diagnostic test or lab investigation ordered/discussed.")
    status: Literal["ordered", "reviewed", "recommended", "completed"] = Field(
        default="ordered",
        description="Status of the diagnostic investigation."
    )
    result: str | None = Field(
        default=None,
        description="Reported test result if discussed during consultation."
    )
    source_segments: list[int] = Field(
        default_factory=list,
        description="Transcript segment indices where investigation was ordered or discussed."
    )


class ClinicalEntities(BaseModel):
    symptoms: list[SymptomEntity] = Field(default_factory=list)
    diagnosis: list[DiagnosisEntity] = Field(default_factory=list)
    medications: list[MedicationEntity] = Field(default_factory=list)
    vitals: list[VitalEntity] = Field(default_factory=list)
    investigations: list[InvestigationEntity] = Field(default_factory=list)


class ClinicalExtraction(BaseModel):
    session_id: str = Field(..., min_length=1, description="Associated session identifier.")
    entities: ClinicalEntities = Field(
        default_factory=ClinicalEntities,
        description="Extracted structured clinical entities."
    )
