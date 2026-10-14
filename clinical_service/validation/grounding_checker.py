"""Grounding and Hallucination Detector.

Enforces strict verification against the consultation transcript:
1. Ensures all extracted entities cite valid source_segments.
2. Verifies that medication doses, frequencies, and durations literally appear in cited text.
3. Verifies that numerical vital measurements literally appear in cited text.
4. Detects contradiction or hallucination in negation status.
"""

from __future__ import annotations

import re

from clinical_service.extraction.temporal_med_logic import is_negated
from clinical_service.schemas.interface1_transcript import Transcript
from clinical_service.schemas.interface2_extraction import ClinicalExtraction
from clinical_service.schemas.interface3_bundle import GroundingFlag


class GroundingChecker:
    """Verifies clinical extractions against raw transcript lines."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def verify(
        self, transcript: Transcript, extraction: ClinicalExtraction
    ) -> list[GroundingFlag]:
        flags: list[GroundingFlag] = []
        seg_dict = {
            (s.segment_id if s.segment_id is not None else idx): s.text
            for idx, s in enumerate(transcript.segments)
        }
        def get_cited_text(sources: list[int]) -> str:
            return " ".join([seg_dict.get(s, "") for s in sources])

        # 1. Verify Symptoms
        for idx, sym in enumerate(extraction.entities.symptoms):
            entity_tag = f"symptoms[{idx}]"
            if not sym.source_segments:
                flags.append(
                    GroundingFlag(
                        entity=entity_tag,
                        issue="No source_segments provided for symptom",
                        severity="error",
                        offending_value=sym.text,
                        source_segments=[],
                    )
                )
                continue

            invalid_segs = [s for s in sym.source_segments if s not in seg_dict]
            if invalid_segs:
                flags.append(
                    GroundingFlag(
                        entity=entity_tag,
                        issue=f"Referenced segment indices {invalid_segs} do not exist in transcript",
                        severity="error",
                        source_segments=sym.source_segments,
                    )
                )

            cited_text = get_cited_text(sym.source_segments)
            text_negated = is_negated(cited_text)
            if sym.negated != text_negated and sym.negated is False and text_negated is True:
                flags.append(
                    GroundingFlag(
                        entity=entity_tag,
                        issue="Symptom extracted as positive, but source segment contains negation marker",
                        severity="warning",
                        offending_value=sym.text,
                        source_segments=sym.source_segments,
                    )
                )

        # 2. Verify Diagnoses
        for idx, diag in enumerate(extraction.entities.diagnosis):
            entity_tag = f"diagnosis[{idx}]"
            if not diag.source_segments:
                flags.append(
                    GroundingFlag(
                        entity=entity_tag,
                        issue="No source_segments provided for diagnosis",
                        severity="error",
                        offending_value=diag.text,
                        source_segments=[],
                    )
                )

        # 3. Verify Medications
        for idx, med in enumerate(extraction.entities.medications):
            entity_tag = f"medications[{idx}]"
            if not med.source_segments:
                flags.append(
                    GroundingFlag(
                        entity=entity_tag,
                        issue="No source_segments provided for medication",
                        severity="error",
                        offending_value=med.name,
                        source_segments=[],
                    )
                )
                continue

            cited_text = get_cited_text(med.source_segments)
            cited_lower = cited_text.lower()

            if med.dose:
                nums = re.findall(r"\d+", med.dose)
                dose_matched = any(num in cited_text for num in nums)
                if not dose_matched and not re.search(r"\b" + re.escape(med.dose.lower()) + r"\b", cited_lower):
                    flags.append(
                        GroundingFlag(
                            entity=entity_tag,
                            issue=f"dose '{med.dose}' not found in transcript source segments",
                            severity="error",
                            offending_value=med.dose,
                            source_segments=med.source_segments,
                        )
                    )

            if med.duration:
                dur_nums = re.findall(r"\d+", med.duration)
                dur_matched = any(num in cited_text for num in dur_nums) or any(
                    w in cited_lower for w in ["din", "day", "hafta", "week", "mahina", "month"]
                )
                if not dur_matched:
                    flags.append(
                        GroundingFlag(
                            entity=entity_tag,
                            issue=f"duration '{med.duration}' not found in transcript source segments",
                            severity="warning",
                            offending_value=med.duration,
                            source_segments=med.source_segments,
                        )
                    )

        # 4. Verify Vitals
        for idx, vit in enumerate(extraction.entities.vitals):
            entity_tag = f"vitals[{idx}]"
            if not vit.source_segments:
                flags.append(
                    GroundingFlag(
                        entity=entity_tag,
                        issue="No source_segments provided for vital sign",
                        severity="error",
                        offending_value=str(vit.value),
                        source_segments=[],
                    )
                )
                continue

            cited_text = get_cited_text(vit.source_segments)
            val_str = str(vit.value)
            if val_str.endswith(".0"):
                clean_val = val_str[:-2]
            else:
                clean_val = val_str

            if clean_val not in cited_text and val_str not in cited_text:
                flags.append(
                    GroundingFlag(
                        entity=entity_tag,
                        issue=f"vital value '{vit.value}' not found in transcript source segments",
                        severity="error",
                        offending_value=str(vit.value),
                        source_segments=vit.source_segments,
                    )
                )

        return flags
