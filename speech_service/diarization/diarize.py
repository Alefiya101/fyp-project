# -*- coding: utf-8 -*-
"""diarize.py

Speaker diarization for doctor-patient OPD recordings.

Uses ``pyannote/speaker-diarization-3.x`` when an ``HF_TOKEN`` environment
variable is set.  Falls back to a simple energy-based two-speaker segmenter
when pyannote is unavailable (missing token, model not downloaded, etc.).

Public API
----------
DiarizationSegment  -- frozen dataclass (start, end, speaker)
diarize             -- run diarization on a mono 16 kHz waveform
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiarizationSegment:
    """A speaker-labelled speech segment."""
    start: float   # seconds
    end: float     # seconds
    speaker: str   # "doctor" | "patient" | "SPEAKER_00" | …


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Energy-based fallback diarizer
# ---------------------------------------------------------------------------

def _energy_diarize(
    wav: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    hop_ms: int = 10,
) -> List[DiarizationSegment]:
    """Extremely simple energy-based two-speaker splitter.

    Splits the audio into short frames, computes RMS energy, and
    thresholds at the median to produce two pseudo-speaker labels.
    This is NOT a real diarizer — it exists solely as a CI/offline
    fallback so the pipeline can run without a Hugging Face token.
    """
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if frame < 1 or hop < 1:
        return [DiarizationSegment(0.0, len(wav) / sr, "SPEAKER_00")]

    energies: list[float] = []
    positions: list[float] = []
    i = 0
    while i + frame <= len(wav):
        chunk = wav[i: i + frame]
        energies.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))
        positions.append(i / sr)
        i += hop

    if not energies:
        return [DiarizationSegment(0.0, len(wav) / sr, "SPEAKER_00")]

    median_e = float(np.median(energies))
    duration = len(wav) / sr

    segments: List[DiarizationSegment] = []
    prev_label: Optional[str] = None
    seg_start = 0.0

    for pos, energy in zip(positions, energies):
        label = "SPEAKER_01" if energy > median_e else "SPEAKER_00"
        if label != prev_label:
            if prev_label is not None and pos > seg_start:
                segments.append(DiarizationSegment(seg_start, pos, prev_label))
            seg_start = pos
            prev_label = label

    if prev_label is not None and duration > seg_start:
        segments.append(DiarizationSegment(seg_start, duration, prev_label))

    return segments or [DiarizationSegment(0.0, duration, "SPEAKER_00")]


# ---------------------------------------------------------------------------
# Speaker-role assignment heuristic
# ---------------------------------------------------------------------------

def _assign_roles(segments: List[DiarizationSegment]) -> List[DiarizationSegment]:
    """Map pyannote/energy speaker labels to "doctor" / "patient".

    Heuristic: the speaker whose *first turn* appears earliest in the
    session is labelled "doctor"; the other speaker is labelled "patient".
    Additional (>2) speakers are all labelled "patient".
    """
    if not segments:
        return segments

    # Find the two most common speakers by total duration
    from collections import defaultdict  # noqa: PLC0415
    duration_by_spk: dict[str, float] = defaultdict(float)
    for seg in segments:
        duration_by_spk[seg.speaker] += seg.end - seg.start

    all_speakers = sorted(duration_by_spk, key=lambda s: -duration_by_spk[s])
    top_two = set(all_speakers[:2])

    # Among top-two, the first to speak is the doctor
    first_speaker = min(
        (seg.speaker for seg in segments if seg.speaker in top_two),
        key=lambda spk: next(s.start for s in segments if s.speaker == spk),
    )

    role_map: dict[str, str] = {}
    for spk in all_speakers:
        if spk == first_speaker:
            role_map[spk] = "doctor"
        elif spk in top_two:
            role_map[spk] = "patient"
        else:
            role_map[spk] = "patient"

    return [
        DiarizationSegment(s.start, s.end, role_map.get(s.speaker, "patient"))
        for s in segments
    ]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def diarize(
    wav: np.ndarray,
    sr: int = 16_000,
    num_speakers: Optional[int] = 2,
    seed: int = 42,
) -> List[DiarizationSegment]:
    """Diarize *wav* into speaker-labelled segments.

    Parameters
    ----------
    wav : np.ndarray
        Mono float32 waveform at *sr* Hz.
    sr : int
        Sample rate (must be 16 000 Hz for pyannote).
    num_speakers : int or None
        Expected number of speakers.  Pass ``None`` to auto-detect.
    seed : int
        Random seed for deterministic output.

    Returns
    -------
    List[DiarizationSegment]
        Speaker-labelled segments sorted by start time.

    Raises
    ------
    ValueError
        If *wav* is ``None`` or *sr* < 1.
    """
    if wav is None:
        raise ValueError("wav must not be None")
    if sr < 1:
        raise ValueError(f"sr must be >= 1, got {sr}")

    _set_seeds(seed)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if hf_token:
        try:
            return _pyannote_diarize(wav, sr, num_speakers, hf_token)
        except Exception as exc:
            logger.warning(
                "pyannote diarization failed (%s); falling back to energy-based splitter.", exc
            )

    logger.info("Using energy-based fallback diarizer (no HF_TOKEN or pyannote unavailable).")
    raw_segments = _energy_diarize(wav, sr)
    return _assign_roles(raw_segments)


# ---------------------------------------------------------------------------
# pyannote diarizer
# ---------------------------------------------------------------------------

def _pyannote_diarize(
    wav: np.ndarray,
    sr: int,
    num_speakers: Optional[int],
    hf_token: str,
) -> List[DiarizationSegment]:
    import torch  # noqa: PLC0415
    from pyannote.audio import Pipeline  # noqa: PLC0415

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline = pipeline.to(torch.device("cpu"))

    tensor = torch.from_numpy(wav).unsqueeze(0)  # (1, samples)
    audio_input = {"waveform": tensor, "sample_rate": sr}

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(audio_input, **kwargs)

    raw: List[DiarizationSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw.append(DiarizationSegment(
            start=round(turn.start, 3),
            end=round(turn.end, 3),
            speaker=speaker,
        ))

    raw.sort(key=lambda s: s.start)
    return _assign_roles(raw)
