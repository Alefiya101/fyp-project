# -*- coding: utf-8 -*-
"""denoise.py

Audio loading and spectral-gating noise reduction for the speech pipeline.

Public API
----------
AudioLoadError  -- raised when a file cannot be loaded or decoded
load_audio      -- load any audio file as mono float32
resample_to_16k -- polyphase resampling to 16 000 Hz
denoise         -- apply stationary spectral-gating noise reduction
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AudioLoadError(RuntimeError):
    """Raised when audio cannot be loaded or decoded."""


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV (or any soundfile-readable) file as mono float32.

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    wav : np.ndarray
        Mono float32 waveform.
    sr : int
        Original sample rate.

    Raises
    ------
    AudioLoadError
        If the file does not exist or cannot be decoded.
    """
    p = Path(path)
    if not p.exists():
        raise AudioLoadError(f"Audio file not found: {path}")
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioLoadError(f"Failed to decode audio '{path}': {exc}") from exc

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = data.mean(axis=1)

    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample *wav* from *sr* Hz to 16 000 Hz using a polyphase filter.

    Parameters
    ----------
    wav : np.ndarray
        Mono float32 waveform at sample rate *sr*.
    sr : int
        Source sample rate.

    Returns
    -------
    np.ndarray
        Mono float32 waveform at 16 000 Hz.
    """
    if sr == 16_000:
        return wav
    g = gcd(sr, 16_000)
    resampled = resample_poly(wav, 16_000 // g, sr // g)
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def denoise(wav: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """Apply stationary spectral-gating noise reduction.

    Uses ``noisereduce.reduce_noise`` with ``stationary=True``.  If the input
    is silent (RMS ≈ 0) or if ``noisereduce`` raises a runtime error the
    original array is returned unchanged.

    Parameters
    ----------
    wav : np.ndarray
        Mono float32 waveform.
    sr : int
        Sample rate (default 16 000 Hz).

    Returns
    -------
    np.ndarray
        Denoised mono float32 waveform of the *same length* as *wav*, with
        all sample values in ``[-1.0, 1.0]``.
    """
    if wav.size == 0:
        return wav.copy()

    # Skip denoising on silent audio (avoids division-by-zero inside noisereduce)
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return wav.copy()

    try:
        import noisereduce as nr  # lazy import — not always installed during tests
        cleaned = nr.reduce_noise(y=wav, sr=sr, stationary=True)
        cleaned = np.clip(cleaned, -1.0, 1.0).astype(np.float32)
        # noisereduce may occasionally return a slightly different length
        if len(cleaned) != len(wav):
            logger.warning(
                "noisereduce returned length %d (input %d); padding/trimming.",
                len(cleaned), len(wav),
            )
            if len(cleaned) < len(wav):
                cleaned = np.pad(cleaned, (0, len(wav) - len(cleaned)))
            else:
                cleaned = cleaned[: len(wav)]
        return cleaned
    except Exception as exc:
        logger.error("noisereduce failed (%s); returning original audio.", exc)
        return wav.copy()
