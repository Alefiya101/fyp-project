"""Clinical Service Orchestrator (Person B Work Package).

Chains together:
  Interface 1 Transcript JSON
      -> SLM / Clinical Extraction (Interface 2)
      -> Deterministic FHIR R4 Mapping (ABDM OPConsultRecord)
      -> Grounding Check (Hallucination Detection)
      -> Schema Validation
      -> Interface 3 Validated Bundle JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinical_service.extraction.extractor import ClinicalExtractor
from clinical_service.extraction.llm_refiner import refine_transcript
from clinical_service.extraction.normalizer import normalize_transcript
from clinical_service.fhir_mapper.mapper import FHIRMapper
from clinical_service.schemas.interface1_transcript import Transcript
from clinical_service.schemas.interface2_extraction import ClinicalExtraction
from clinical_service.schemas.interface3_bundle import ValidatedBundle
from clinical_service.validation.fhir_validator import FHIRValidator
from clinical_service.validation.grounding_checker import GroundingChecker


class ClinicalService:
    """End-to-end Clinical NLP and FHIR Service for Indian OPD Ambient Scribing."""

    def __init__(
        self,
        extractor_backend: str = "rule_based",
        model_path: str | None = None,
        strict_grounding: bool = False,
        enable_llm_refiner: bool = True,
    ):
        self.extractor = ClinicalExtractor(backend=extractor_backend, model_path=model_path)
        self.mapper = FHIRMapper()
        self.grounding_checker = GroundingChecker(strict_mode=strict_grounding)
        self.fhir_validator = FHIRValidator()
        self.enable_llm_refiner = enable_llm_refiner

    def process_transcript(
        self,
        transcript_input: dict[str, Any] | str | Path | Transcript,
    ) -> ValidatedBundle:
        """Processes an Interface 1 Transcript and produces an Interface 3 Validated Bundle."""
        # 1. Parse and validate Interface 1 input
        if isinstance(transcript_input, Transcript):
            transcript = transcript_input
        else:
            if isinstance(transcript_input, (str, Path)):
                p = Path(transcript_input)
                if p.is_file():
                    raw_data = json.loads(p.read_text(encoding="utf-8"))
                else:
                    raw_data = json.loads(str(transcript_input))
            elif isinstance(transcript_input, dict):
                raw_data = transcript_input
            else:
                raise TypeError(f"Unsupported transcript input type: {type(transcript_input)}")

            # Filter out silent/empty segments from ASR output
            if isinstance(raw_data, dict) and "segments" in raw_data:
                cleaned_segments = [
                    s for s in raw_data["segments"]
                    if (s.get("text") if isinstance(s, dict) else getattr(s, "text", "") or "").strip()
                ]
                raw_data = {**raw_data, "segments": cleaned_segments}

            transcript = Transcript.model_validate(raw_data)

        # 1.4. Reconstruct noisy ASR dialogue using Local LLM Refiner (Qwen2.5-1.5B)
        if self.enable_llm_refiner:
            transcript = refine_transcript(transcript)

        # 1.5. Normalize and restore noisy ASR speech
        transcript = normalize_transcript(transcript)

        # 2. Extract Structured Clinical Entities (Interface 2)
        extraction: ClinicalExtraction = self.extractor.extract(transcript)

        # 3. Deterministic Mapping to FHIR R4 Bundle (ABDM OPConsultRecord)
        bundle_dict: dict[str, Any] = self.mapper.map_to_fhir_bundle(extraction)

        # 4. Grounding & Hallucination Check
        grounding_flags = self.grounding_checker.verify(transcript, extraction)

        # 5. FHIR Schema and Profile Validation
        validation_result = self.fhir_validator.validate_bundle(bundle_dict)

        # 6. Assemble Interface 3 Validated Bundle
        validated_bundle = ValidatedBundle(
            session_id=transcript.session_id,
            fhir_bundle=bundle_dict,
            validation=validation_result,
            grounding_flags=grounding_flags,
        )

        return validated_bundle

    def extract_entities_only(
        self,
        transcript_input: dict[str, Any] | Transcript,
    ) -> ClinicalExtraction:
        """Returns only the Interface 2 Extraction JSON (for UI display or debugging)."""
        if isinstance(transcript_input, dict):
            transcript = Transcript.model_validate(transcript_input)
        else:
            transcript = transcript_input
        if self.enable_llm_refiner:
            transcript = refine_transcript(transcript)
        transcript = normalize_transcript(transcript)
        return self.extractor.extract(transcript)

    def process_audio(
        self,
        audio_path: str | Path,
        session_id: str | None = None,
        person_a_only: bool = False,
    ) -> ValidatedBundle:
        """End-to-end ambient scribing: audio (.wav) -> Interface 1 Transcript -> Interface 3 Validated Bundle.

        Chains Person A's SpeechService (preprocessing, diarization, Whisper ASR)
        directly with Person B's clinical extraction and ABDM FHIR mapping.
        """
        try:
            import importlib

            speech_mod = importlib.import_module("speech_service.speech_service")
            speech_cls = speech_mod.SpeechService
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "speech_service and its audio dependencies (soundfile, transformers, torch) "
                "are required to process raw audio files directly. Install them from requirements.txt."
            ) from exc

        speech_svc = speech_cls()
        transcript_dict = speech_svc.process(
            str(audio_path),
            session_id=session_id,
            person_a_only=person_a_only,
        )
        return self.process_transcript(transcript_dict)
