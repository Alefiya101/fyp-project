# -*- coding: utf-8 -*-
"""test_speech_service.py

Integration tests for SpeechService, transcribe_batch, and transcribe_stream.

All heavy models (ASR, VAD, diarizer) are mocked — no network or GPU required.
# Feature: person-a-speech-pipeline
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from speech_service.diarization.diarize import DiarizationSegment
from speech_service.preprocessing.vad import SpeechSegment
from speech_service.speech_service import (
    SpeechService,
    TranscriptSegment,
    transcribe_batch,
    transcribe_stream,
    _build_transcript_json,
)

SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(tmp_path: Path, duration_s: float = 1.0) -> str:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "test.wav"
    sf.write(str(p), wav, SR, subtype="PCM_16")
    return str(p)


def _make_service_with_mocks(
    vad_return=None,
    dia_return=None,
    asr_text="hello world",
    asr_confidence=0.9,
) -> SpeechService:
    """Build a SpeechService with mocked VAD, diarizer and ASR."""
    svc = SpeechService()

    if vad_return is None:
        vad_return = [SpeechSegment(0.0, 1.0)]
    if dia_return is None:
        dia_return = [DiarizationSegment(0.0, 1.0, "doctor")]

    # Mock ASR model
    asr_seg = MagicMock()
    asr_seg.text = asr_text
    asr_seg.avg_logprob = math.log(max(asr_confidence, 1e-9))
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([asr_seg], MagicMock())
    svc._asr_model = mock_model

    return svc, vad_return, dia_return


# ===========================================================================
# SpeechService.process — unit tests
# ===========================================================================

class TestSpeechServiceProcess:
    def test_lazy_loading_asr_model(self, tmp_path):
        """Model should be None before first process() call."""
        svc = SpeechService()
        assert svc._asr_model is None

    def test_raises_file_not_found(self):
        svc = SpeechService()
        with pytest.raises(FileNotFoundError):
            svc.process("/no/such/file.wav")

    def test_returns_empty_on_silent_audio(self, tmp_path):
        silent_path = str(tmp_path / "silent.wav")
        sf.write(silent_path, np.zeros(SR, dtype=np.float32), SR, subtype="PCM_16")

        svc = SpeechService()
        with patch("speech_service.speech_service.get_speech_segments", return_value=[]):
            result = svc.process(silent_path)

        assert result["segments"] == []
        assert "session_id" in result
        assert "language_mix" in result

    def test_process_returns_interface1_schema(self, tmp_path):
        wav_path = _make_wav(tmp_path)
        svc, vad_ret, dia_ret = _make_service_with_mocks()

        with patch("speech_service.speech_service.get_speech_segments", return_value=vad_ret), \
             patch("speech_service.speech_service.diarize", return_value=dia_ret):
            result = svc.process(wav_path, session_id="S001")

        assert svc._asr_model.transcribe.call_args.kwargs == {
            "beam_size": 1,
            "language": None,
        }

        # Top-level keys
        assert result["session_id"] == "S001"
        assert "language_mix" in result
        assert isinstance(result["segments"], list)

        # Segment schema
        for seg in result["segments"]:
            assert "start" in seg
            assert "end" in seg
            assert "speaker" in seg
            assert "text" in seg
            assert "lang" in seg
            assert "confidence" in seg

    def test_segment_with_empty_asr_text_is_included(self, tmp_path):
        wav_path = _make_wav(tmp_path)
        svc, vad_ret, dia_ret = _make_service_with_mocks(asr_text="")

        with patch("speech_service.speech_service.get_speech_segments", return_value=vad_ret), \
             patch("speech_service.speech_service.diarize", return_value=dia_ret):
            result = svc.process(wav_path)

        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == ""

    def test_person_a_only_filters_to_highest_turn_speaker(self, tmp_path):
        """person_a_only=True keeps only the speaker with the most turns."""
        wav_path = _make_wav(tmp_path, duration_s=3.0)
        vad_ret = [
            SpeechSegment(0.0, 0.5),
            SpeechSegment(0.6, 1.0),
            SpeechSegment(1.1, 1.5),
            SpeechSegment(1.6, 2.0),
        ]
        dia_ret = [
            DiarizationSegment(0.0, 0.5, "doctor"),
            DiarizationSegment(0.6, 1.0, "patient"),
            DiarizationSegment(1.1, 1.5, "doctor"),
            DiarizationSegment(1.6, 2.0, "doctor"),
        ]
        svc, _, _ = _make_service_with_mocks(vad_return=vad_ret, dia_return=dia_ret)

        with patch("speech_service.speech_service.get_speech_segments", return_value=vad_ret), \
             patch("speech_service.speech_service.diarize", return_value=dia_ret):
            result = svc.process(wav_path, person_a_only=True)

        speakers = {s["speaker"] for s in result["segments"]}
        # doctor has 3 turns vs patient 1 turn — should keep only doctor
        assert speakers == {"doctor"} or len(speakers) == 1

    def test_start_less_than_end_in_all_segments(self, tmp_path):
        """Property 8 (structural): start < end for every segment."""
        wav_path = _make_wav(tmp_path, duration_s=2.0)
        vad_ret = [SpeechSegment(0.0, 1.0), SpeechSegment(1.1, 2.0)]
        dia_ret = [DiarizationSegment(0.0, 2.0, "doctor")]
        svc, _, _ = _make_service_with_mocks(vad_return=vad_ret, dia_return=dia_ret)

        with patch("speech_service.speech_service.get_speech_segments", return_value=vad_ret), \
             patch("speech_service.speech_service.diarize", return_value=dia_ret):
            result = svc.process(wav_path)

        for seg in result["segments"]:
            assert seg["start"] < seg["end"]


# ===========================================================================
# transcribe_stream — unit tests
# ===========================================================================

class TestTranscribeStream:
    def test_raises_file_not_found_before_yielding(self):
        with pytest.raises(FileNotFoundError):
            list(transcribe_stream("/no/such.wav", "S000"))

    def test_yields_sentinel_at_end(self, tmp_path):
        wav_path = _make_wav(tmp_path)
        svc, vad_ret, dia_ret = _make_service_with_mocks()

        with patch("speech_service.speech_service._default_service", svc), \
             patch("speech_service.speech_service.get_speech_segments", return_value=vad_ret), \
             patch("speech_service.speech_service.diarize", return_value=dia_ret):
            yielded = list(transcribe_stream(wav_path, "S001"))

        assert yielded[-1] == {"done": True, "session_id": "S001"}

    def test_segment_dicts_have_required_keys(self, tmp_path):
        wav_path = _make_wav(tmp_path)
        svc, vad_ret, dia_ret = _make_service_with_mocks()

        with patch("speech_service.speech_service._default_service", svc), \
             patch("speech_service.speech_service.get_speech_segments", return_value=vad_ret), \
             patch("speech_service.speech_service.diarize", return_value=dia_ret):
            yielded = list(transcribe_stream(wav_path, "S002"))

        non_sentinel = [y for y in yielded if "done" not in y]
        for seg in non_sentinel:
            assert "start" in seg
            assert "end" in seg
            assert "speaker" in seg
            assert "text" in seg
            assert "lang" in seg
            assert "confidence" in seg

    def test_empty_segments_yields_only_sentinel(self, tmp_path):
        wav_path = _make_wav(tmp_path)

        with patch("speech_service.speech_service.get_speech_segments", return_value=[]):
            svc = SpeechService()
            with patch("speech_service.speech_service._default_service", svc):
                yielded = list(transcribe_stream(wav_path, "S003"))

        assert len(yielded) == 1
        assert yielded[0]["done"] is True


# ===========================================================================
# transcribe_batch convenience function
# ===========================================================================

class TestTranscribeBatch:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            transcribe_batch("/nonexistent.wav", "S999")

    def test_returns_dict_with_interface1_keys(self, tmp_path):
        wav_path = _make_wav(tmp_path)
        svc, vad_ret, dia_ret = _make_service_with_mocks()

        with patch("speech_service.speech_service._default_service", svc), \
             patch("speech_service.speech_service.get_speech_segments", return_value=vad_ret), \
             patch("speech_service.speech_service.diarize", return_value=dia_ret):
            result = transcribe_batch(wav_path, "S010")

        assert set(result.keys()) >= {"session_id", "language_mix", "segments"}
        assert result["session_id"] == "S010"


# ===========================================================================
# _build_transcript_json helper
# ===========================================================================

class TestBuildTranscriptJson:
    def test_empty_segments(self):
        result = _build_transcript_json("S000", [])
        assert result["session_id"] == "S000"
        assert result["segments"] == []
        assert "language_mix" in result

    def test_language_mix_is_most_frequent_lang(self):
        segs = [
            TranscriptSegment(0.0, 1.0, "doctor", "hello", lang="hi-en"),
            TranscriptSegment(1.0, 2.0, "patient", "namaste", lang="hi"),
            TranscriptSegment(2.0, 3.0, "doctor", "ok", lang="hi-en"),
        ]
        result = _build_transcript_json("S001", segs)
        assert result["language_mix"] == "hi-en"


# ===========================================================================
# Property-based tests
# ===========================================================================

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    _HYPOTHESIS = True
except ImportError:
    _HYPOTHESIS = False

if _HYPOTHESIS:
    # Feature: person-a-speech-pipeline, Property 8
    @given(
        n_doc=st.integers(min_value=2, max_value=8),
        n_pat=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_person_a_filter_returns_highest_turn_count_speaker(n_doc, n_pat):
        """Property 8 (filter part): person_a_only keeps only the majority-turn speaker."""
        segs = []
        t = 0.0
        for _ in range(n_doc):
            segs.append(TranscriptSegment(t, t + 0.5, "doctor", "text", "hi-en"))
            t += 0.6
        for _ in range(n_pat):
            segs.append(TranscriptSegment(t, t + 0.5, "patient", "text", "hi-en"))
            t += 0.6

        doctor = SpeechService._identify_person_a(segs)
        # With n_doc > n_pat, doctor should be "doctor"
        if n_doc > n_pat:
            assert doctor == "doctor"

    # Feature: person-a-speech-pipeline, Property 8 (structural)
    @given(
        starts=st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_transcript_segment_start_less_than_end(starts):
        """Property 8 (structural): all TranscriptSegments have start < end."""
        segs = [
            TranscriptSegment(s, s + 0.5, "doctor", "text", "hi-en")
            for s in starts
        ]
        result = _build_transcript_json("S_prop", segs)
        for seg in result["segments"]:
            assert seg["start"] < seg["end"]
