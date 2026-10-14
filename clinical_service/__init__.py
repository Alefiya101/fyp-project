"""Clinical NLP and FHIR Service Package (Person B Work Package).

Ambient clinical scribe processing pipeline:
- Interface 1: Transcript JSON
- Interface 2: Clinical Extraction JSON
- Interface 3: Validated FHIR R4 Bundle + Grounding Flags
"""

from clinical_service.fhir_mapper.mapper import FHIRMapper
from clinical_service.schemas.interface1_transcript import Transcript, TranscriptSegment
from clinical_service.schemas.interface2_extraction import (
    ClinicalEntities,
    ClinicalExtraction,
)
from clinical_service.schemas.interface3_bundle import (
    GroundingFlag,
    ValidatedBundle,
    ValidationResult,
)
from clinical_service.service import ClinicalService
from clinical_service.validation.fhir_validator import FHIRValidator
from clinical_service.validation.grounding_checker import GroundingChecker

__all__ = [
    "ClinicalEntities",
    "ClinicalExtraction",
    "ClinicalService",
    "FHIRMapper",
    "FHIRValidator",
    "GroundingChecker",
    "GroundingFlag",
    "Transcript",
    "TranscriptSegment",
    "ValidatedBundle",
    "ValidationResult",
]
