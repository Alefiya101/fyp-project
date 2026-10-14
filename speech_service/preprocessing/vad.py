# -*- coding: utf-8 -*-
"""vad.py

Voice Activity Detection using Silero-VAD (loaded via torch.hub).

Public API
----------
SpeechSegment       -- frozen dataclass (start: float, end: float) in seconds
get_speech_segments -- detect speech boundaries in a 16 kHz mono waveform
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton so the model is loaded only once per process.
_vad_model: Optional[object] = None
_vad_utils: Optional[tuple] = None

_MIN_DURATION_S = 0.1  # clips shorter than this skip VAD entirely


@dataclass(frozen=True)
class SpeechSegment:
    """A half-open interval ``[start, end)`` of detected speech."""
    start: float  # seconds
    end: float    # seconds


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_silero_vad():
    """Load the Silero-VAD model from torch.hub (cached after first call)."""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model, _vad_utils

    import torch  # noqa: PLC0415
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        verbose=False,
    )
    _vad_model = model
    _vad_utils = utils
    return model, utils


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def get_speech_segments(
    wav: np.ndarray,
    sr: int = 16_000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> List[SpeechSegment]:
    """Detect speech segments in *wav* using Silero-VAD.

    Parameters
    ----------
    wav : np.ndarray
        Mono float32 waveform at 16 000 Hz.
    sr : int
        Sample rate — must be 16 000 Hz (Silero-VAD requirement).
    threshold : float
        VAD probability threshold (0–1).  Higher = stricter.
    min_speech_duration_ms : int
        Minimum speech segment length in milliseconds.
    min_silence_duration_ms : int
        Minimum silence gap before splitting a segment.

    Returns
    -------
    List[SpeechSegment]
        Sorted, non-overlapping speech segments.  Returns an empty list if
        the audio is shorter than 0.1 s, completely silent, or no speech
        is detected.
    """
    if wav is None or wav.size == 0:
        return []

    duration = len(wav) / max(sr, 1)
    if duration < _MIN_DURATION_S:
        return []

    # Completely silent → skip model inference
    if float(np.max(np.abs(wav))) < 1e-7:
        return []

    try:
        import torch  # noqa: PLC0415
        model, utils = _load_silero_vad()
        (get_speech_ts,) = (utils[0],)  # first util is get_speech_timestamps

        tensor = torch.from_numpy(wav)
        speech_ts = get_speech_ts(
            tensor,
            model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            sampling_rate=sr,
            return_seconds=False,  # returns sample indices
        )
    except Exception as exc:
        logger.error("Silero-VAD failed (%s); returning empty segment list.", exc)
        return []

    segments: List[SpeechSegment] = []
    for ts in speech_ts:
        start_s = ts["start"] / sr
        end_s = ts["end"] / sr
        # Clamp to valid range
        start_s = max(0.0, min(start_s, duration))
        end_s = max(start_s, min(end_s, duration))
        if end_s > start_s:
            segments.append(SpeechSegment(start=start_s, end=end_s))

    return sorted(segments, key=lambda s: s.start)
