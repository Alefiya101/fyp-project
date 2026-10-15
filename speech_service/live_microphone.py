"""Record a live microphone utterance and run speech-to-FHIR processing.

The recording is captured for a fixed duration. ASR, diarization, clinical
extraction, and FHIR mapping run after recording finishes.

Usage:
    python -m speech_service.live_microphone --duration-seconds 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _resolve_device(device: str | None) -> int | str | None:
    if device is None:
        return None
    return int(device) if device.isdecimal() else device


def record_microphone(
    duration_seconds: float,
    output_path: str | Path,
    device: str | None = None,
) -> Path:
    """Record mono audio from the selected/default microphone to a WAV file."""
    if duration_seconds <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    import sounddevice as sd

    selected_device = _resolve_device(device)
    input_device = sd.query_devices(selected_device, kind="input")
    sample_rate = int(input_device["default_samplerate"])
    frames = int(duration_seconds * sample_rate)
    if frames <= 0:
        raise ValueError("Recording duration is too short for the microphone sample rate.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Recording from '{input_device['name']}' for {duration_seconds:g} seconds. "
        "Please speak after recording starts."
    )
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=selected_device,
    )
    sd.wait()
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise RuntimeError("Microphone recording returned empty or invalid audio.")

    sf.write(str(destination), audio, sample_rate, subtype="PCM_16")
    print(f"Recording saved to: {destination}")
    return destination


def run_live_pipeline(
    audio_path: str | Path,
    transcript_path: str | Path,
    bundle_path: str | Path,
    session_id: str | None = None,
    person_a_only: bool = False,
) -> None:
    """Transcribe a recording and map it through clinical extraction to FHIR."""
    from clinical_service.service import ClinicalService
    from speech_service.speech_service import SpeechService

    audio_file = Path(audio_path)
    transcript_file = Path(transcript_path)
    bundle_file = Path(bundle_path)
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    bundle_file.parent.mkdir(parents=True, exist_ok=True)

    speech_service = SpeechService()
    transcript = speech_service.process(
        str(audio_file),
        session_id=session_id or audio_file.stem,
        person_a_only=person_a_only,
    )
    transcript_file.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = ClinicalService().process_transcript(transcript)
    bundle_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print(f"Transcript saved to: {transcript_file}")
    print(f"FHIR bundle saved to: {bundle_file}")
    print(f"FHIR schema valid: {result.validation.schema_valid}")
    print(f"Grounding flags: {len(result.grounding_flags)}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a microphone utterance and run the speech-to-FHIR pipeline."
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=30.0,
        help="Fixed microphone recording duration (default: 30 seconds).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Input device ID or name. Defaults to the system's input device.",
    )
    parser.add_argument(
        "--audio-output",
        default="recordings/live_consultation.wav",
        help="Path to save the microphone recording.",
    )
    parser.add_argument(
        "--transcript-output",
        default="output/live_transcript.json",
        help="Path to save the Interface 1 transcript.",
    )
    parser.add_argument(
        "--bundle-output",
        default="output/live_fhir_bundle.json",
        help="Path to save the validated FHIR bundle.",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--person-a-only",
        action="store_true",
        help="Keep only the speaker with the most detected turns.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    audio_path = record_microphone(
        duration_seconds=args.duration_seconds,
        output_path=args.audio_output,
        device=args.device,
    )
    run_live_pipeline(
        audio_path=audio_path,
        transcript_path=args.transcript_output,
        bundle_path=args.bundle_output,
        session_id=args.session_id,
        person_a_only=args.person_a_only,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
