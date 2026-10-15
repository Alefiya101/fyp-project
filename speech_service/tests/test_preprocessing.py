# -*- coding: utf-8 -*-
"""test_preprocessing.py

Unit and property-based tests for:
  - speech_service.preprocessing.denoise  (AudioLoadError, load_audio, denoise)
  - speech_service.preprocessing.vad      (get_speech_segments)

Property tests use the hypothesis library.
All tests run on CPU, no network access, no model downloads.
# Feature: person-a-speech-pipeline
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from speech_service.preprocessing.denoise import (
    AudioLoadError,
    denoise,
    load_audio,
    resample_to_16k,
)
from speech_service.preprocessing.vad import SpeechSegment, get_speech_segments

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 16_000


def _sine(freq_hz: float, duration_s: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    """Generate a mono float32 sine wave."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


@pytest.fixture()
def tmp_wav(tmp_path):
    """Factory that writes a numpy array to a temp WAV file and returns the path."""
    created = []

    def _make(wav: np.ndarray, sr: int = SR) -> str:
        p = tmp_path / f"test_{len(created)}.wav"
        sf.write(str(p), wav, sr, subtype="PCM_16")
        created.append(p)
        return str(p)

    yield _make

    for p in created:
        try:
            p.unlink()
        except OSError:
            pass


# ===========================================================================
# denoise.py — unit tests
# ===========================================================================

class TestLoadAudio:
    def test_raises_on_missing_file(self):
        with pytest.raises(AudioLoadError, match="not found"):
            load_audio("/no/such/file.wav")

    def test_loads_mono_float32(self, tmp_wav):
        wav = _sine(440, 1.0)
        path = tmp_wav(wav)
        out, sr = load_audio(path)
        assert out.dtype == np.float32
        assert out.ndim == 1
        assert sr == SR

    def test_converts_stereo_to_mono(self, tmp_wav):
        stereo = np.column_stack([_sine(440, 0.5), _sine(880, 0.5)])
        p = tmp_wav(stereo)
        out, sr = load_audio(p)
        assert out.ndim == 1

    def test_resample_to_16k_passthrough(self):
        wav = _sine(440, 0.5, sr=SR)
        out = resample_to_16k(wav, SR)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(wav, out)

    def test_resample_to_16k_from_8k(self):
        wav8k = _sine(440, 1.0, sr=8_000)
        out = resample_to_16k(wav8k, 8_000)
        assert out.dtype == np.float32
        assert len(out) == pytest.approx(2 * len(wav8k), abs=10)


class TestDenoise:
    def test_output_same_length_and_dtype(self):
        wav = _sine(440, 1.0) + np.random.default_rng(0).normal(0, 0.05, SR).astype(np.float32)
        out = denoise(wav, sr=SR)
        assert out.dtype == np.float32
        assert len(out) == len(wav)

    def test_output_clipped_to_unit_range(self):
        wav = _sine(440, 1.0) + np.random.default_rng(1).normal(0, 0.1, SR).astype(np.float32)
        out = denoise(wav, sr=SR)
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6

    def test_silent_input_returns_copy(self):
        wav = _silence(1.0)
        out = denoise(wav, sr=SR)
        np.testing.assert_array_equal(out, wav)
        assert out is not wav  # should be a copy

    def test_empty_array_returns_empty(self):
        wav = np.array([], dtype=np.float32)
        out = denoise(wav, sr=SR)
        assert len(out) == 0


# ===========================================================================
# vad.py — unit tests
# ===========================================================================

class TestGetSpeechSegments:
    def test_empty_array_returns_empty(self):
        segs = get_speech_segments(np.array([], dtype=np.float32))
        assert segs == []

    def test_too_short_returns_empty(self):
        wav = _sine(440, 0.05)  # 50 ms < 100 ms threshold
        segs = get_speech_segments(wav)
        assert segs == []

    def test_silent_returns_empty(self):
        wav = _silence(1.0)
        segs = get_speech_segments(wav)
        assert segs == []

    def test_segments_sorted_and_non_overlapping(self):
        """VAD on a 3-second tone should return segments sorted by start."""
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            pytest.skip("torch not installed")

        wav = _sine(1_000, 3.0, amplitude=0.8)
        segs = get_speech_segments(wav)
        if not segs:
            pytest.skip("VAD returned no segments (model may not be cached)")

        for i, seg in enumerate(segs):
            assert seg.start < seg.end, f"Segment {i}: start >= end"
            assert seg.start >= 0.0
            assert seg.end <= len(wav) / SR + 1e-3

        for i in range(len(segs) - 1):
            assert segs[i].end <= segs[i + 1].start + 1e-3, "Overlapping segments"

    def test_segment_is_frozen_dataclass(self):
        seg = SpeechSegment(0.0, 1.0)
        with pytest.raises((AttributeError, TypeError)):
            seg.start = 0.5  # type: ignore[misc]


# ===========================================================================
# Property-based tests (hypothesis)
# ===========================================================================

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    _HYPOTHESIS = True
except ImportError:
    _HYPOTHESIS = False


def _random_float32_array(draw, min_len=1, max_len=16_000):
    length = draw(st.integers(min_value=min_len, max_value=max_len))
    values = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=length, max_size=length,
    ))
    return np.array(values, dtype=np.float32)


if _HYPOTHESIS:
    # Feature: person-a-speech-pipeline, Property 1
    @given(wav=st.builds(
        lambda n, seed: np.random.default_rng(seed).uniform(-1, 1, n).astype(np.float32),
        n=st.integers(min_value=100, max_value=8_000),
        seed=st.integers(min_value=0, max_value=2**31),
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_denoise_preserves_shape_and_dtype(wav):
        """Property 1: denoise output has same length, float32 dtype, values in [-1, 1]."""
        out = denoise(wav, sr=SR)
        assert out.dtype == np.float32
        assert len(out) == len(wav)
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6

    # Feature: person-a-speech-pipeline, Property 2 (structural check without live model)
    def test_vad_segments_structural_invariants_on_long_audio():
        """Property 2 (example-based): VAD segments are bounded and non-overlapping."""
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            pytest.skip("torch not installed")

        rng = np.random.default_rng(42)
        # 2-second bandlimited noise burst
        wav = rng.uniform(-0.5, 0.5, 2 * SR).astype(np.float32)
        segs = get_speech_segments(wav)
        duration = len(wav) / SR
        for seg in segs:
            assert seg.start >= 0.0
            assert seg.end <= duration + 1e-3
            assert seg.start < seg.end
        for i in range(len(segs) - 1):
            assert segs[i].end <= segs[i + 1].start + 1e-3
