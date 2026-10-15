from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from speech_service.live_microphone import (
    _parse_args,
    record_microphone,
    run_live_pipeline,
)


def test_record_microphone_saves_recorded_audio(tmp_path, monkeypatch):
    fake_sounddevice = ModuleType("sounddevice")
    fake_sounddevice.query_devices = MagicMock(
        return_value={"name": "Test microphone", "default_samplerate": 48_000}
    )
    fake_sounddevice.rec = MagicMock(
        return_value=np.zeros((48_000, 1), dtype=np.float32)
    )
    fake_sounddevice.wait = MagicMock()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    output_path = tmp_path / "recordings" / "test.wav"
    result = record_microphone(1.0, output_path)

    assert result == output_path
    audio, sample_rate = sf.read(output_path)
    assert sample_rate == 48_000
    assert audio.shape == (48_000,)
    fake_sounddevice.rec.assert_called_once_with(
        48_000,
        samplerate=48_000,
        channels=1,
        dtype="float32",
        device=None,
    )


def test_record_microphone_rejects_invalid_duration(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="greater than zero"):
        record_microphone(0, tmp_path / "invalid.wav")


def test_cli_defaults_to_bounded_recording():
    args = _parse_args([])

    assert args.duration_seconds == 30.0
    assert args.audio_output.endswith("live_consultation.wav")
    assert args.transcript_output.endswith("live_transcript.json")
    assert args.bundle_output.endswith("live_fhir_bundle.json")


def test_run_live_pipeline_writes_transcript_and_bundle(tmp_path, monkeypatch):
    transcript = {
        "session_id": "session-1",
        "language_mix": "hi-en",
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "doctor", "text": "hello"}],
    }
    speech_module = ModuleType("speech_service.speech_service")
    speech_module.SpeechService = MagicMock()
    speech_module.SpeechService.return_value.process.return_value = transcript

    validated_bundle = SimpleNamespace(
        validation=SimpleNamespace(schema_valid=True),
        grounding_flags=[],
        model_dump_json=MagicMock(return_value='{"resourceType":"Bundle"}'),
    )
    clinical_module = ModuleType("clinical_service.service")
    clinical_module.ClinicalService = MagicMock()
    clinical_module.ClinicalService.return_value.process_transcript.return_value = validated_bundle

    clinical_package = ModuleType("clinical_service")
    clinical_package.__path__ = []
    monkeypatch.setitem(sys.modules, "clinical_service", clinical_package)
    monkeypatch.setitem(sys.modules, "clinical_service.service", clinical_module)
    monkeypatch.setitem(sys.modules, "speech_service.speech_service", speech_module)

    audio_path = tmp_path / "consultation.wav"
    audio_path.touch()
    transcript_path = tmp_path / "output" / "transcript.json"
    bundle_path = tmp_path / "output" / "bundle.json"

    run_live_pipeline(audio_path, transcript_path, bundle_path, session_id="session-1")

    assert '"session_id": "session-1"' in transcript_path.read_text(encoding="utf-8")
    assert bundle_path.read_text(encoding="utf-8") == '{"resourceType":"Bundle"}'
    speech_module.SpeechService.return_value.process.assert_called_once()
    clinical_module.ClinicalService.return_value.process_transcript.assert_called_once_with(
        transcript
    )
