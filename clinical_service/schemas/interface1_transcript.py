"""Interface 1 - Transcript JSON Schema.

Represents the contract between Person A (Speech & Data) and Person B (Clinical NLP & FHIR).
Contains speaker-attributed, timestamped, language-tagged transcript segments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TranscriptSegment(BaseModel):
    segment_id: int | None = Field(
        default=None,
        description="Optional unique integer index for the segment (0-based or 1-based)."
    )
    start: float = Field(..., ge=0.0, description="Start timestamp in seconds.")
    end: float = Field(..., ge=0.0, description="End timestamp in seconds.")
    speaker: Literal["doctor", "patient", "attendant", "unknown"] = Field(
        ...,
        description="Speaker role assigned by diarization and role heuristic."
    )
    text: str = Field(..., min_length=1, description="Transcribed utterance text.")
    lang: str = Field(
        default="hi-en",
        description="Language tag, e.g. 'hi-en', 'hi', 'en', 'en-IN'."
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ASR transcription confidence score."
    )

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"end timestamp ({v}) cannot be before start timestamp ({start})")
        return v


class Transcript(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique session/encounter identifier.")
    language_mix: str = Field(
        default="hi-en",
        description="Overall session language mix (e.g. 'hi-en', 'hi', 'en')."
    )
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Chronological list of transcript segments."
    )

    def full_text(self) -> str:
        """Returns the full concatenated dialogue text with speaker tags."""
        lines = []
        for i, seg in enumerate(self.segments):
            seg_idx = seg.segment_id if seg.segment_id is not None else i
            lines.append(f"[{seg_idx}] {seg.speaker.upper()}: {seg.text}")
        return "\n".join(lines)

    def get_segment(self, index: int) -> TranscriptSegment | None:
        """Finds segment by explicit segment_id or position in array."""
        for seg in self.segments:
            if seg.segment_id == index:
                return seg
        if 0 <= index < len(self.segments):
            return self.segments[index]
        return None
