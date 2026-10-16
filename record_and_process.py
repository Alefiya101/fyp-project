"""record_and_process.py

Live Audio Scribing & ABDM FHIR Pipeline Runner.

Allows recording live speech directly from the microphone (or using an existing WAV file),
transcribing via Whisper ASR, post-correcting with Qwen2.5-1.5B LLM Refiner,
extracting clinical entities, and generating an ABDM-compliant FHIR R4 Bundle.

Usage:
    # 1. Record 15 seconds from your microphone and process:
    python record_and_process.py --duration 15

    # 2. Record interactively (press Enter to start, Enter to stop):
    python record_and_process.py

    # 3. Process an existing recording:
    python record_and_process.py --file recordings/live_consultation.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure root workspace is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import sounddevice as sd
import soundfile as sf

from clinical_service.service import ClinicalService

SAMPLE_RATE = 16000  # 16 kHz mono for optimal Whisper & Silero VAD accuracy
DEFAULT_OUTPUT_DIR = _ROOT / "output"
DEFAULT_RECORDINGS_DIR = _ROOT / "recordings"


def record_timed(duration_seconds: float) -> np.ndarray:
    """Records audio from default microphone for a fixed duration."""
    print(f"\n[MIC] Recording for {duration_seconds:.1f} seconds... Speak now!")
    frames = int(duration_seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")

    for remaining in range(int(duration_seconds), 0, -1):
        sys.stdout.write(f"\r  Recording: {remaining}s remaining... ")
        sys.stdout.flush()
        time.sleep(1.0)
    sd.wait()
    sys.stdout.write("\r[MIC] Recording complete!              \n")
    sys.stdout.flush()
    return audio.flatten()


def record_interactive() -> np.ndarray:
    """Records audio until the user presses Enter."""
    input("\n[MIC] Press ENTER to START recording...")
    print("[MIC] Recording in progress... Press ENTER to STOP recording!")

    recorded_chunks = []

    def callback(indata, frames, time_info, status):
        recorded_chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()

    print("[MIC] Recording stopped!")
    if not recorded_chunks:
        return np.zeros(SAMPLE_RATE, dtype="float32")
    return np.concatenate(recorded_chunks, axis=0).flatten()


def main():
    parser = argparse.ArgumentParser(description="Live Ambient Clinical Scribing CLI")
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Recording duration in seconds (e.g. --duration 15). If omitted, interactive mode is used.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to an existing .wav audio file to process directly without recording.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "live_fhir_bundle.json"),
        help="Output path for the generated FHIR R4 Bundle JSON.",
    )
    args = parser.parse_args()

    DEFAULT_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.file:
        wav_path = Path(args.file)
        if not wav_path.is_file():
            sys.exit(f"[ERROR] Specified file not found: {wav_path}")
        print(f"\n[1/3] Using existing audio file: {wav_path}")
    else:
        wav_path = DEFAULT_RECORDINGS_DIR / "live_consultation.wav"
        if args.duration is not None and args.duration > 0:
            audio_data = record_timed(args.duration)
        else:
            audio_data = record_interactive()

        # Save 16kHz mono WAV
        sf.write(str(wav_path), audio_data, SAMPLE_RATE)
        duration_s = len(audio_data) / SAMPLE_RATE
        print(f"[1/3] Saved {duration_s:.1f}s audio to: {wav_path}")

    # Process audio through full end-to-end pipeline:
    # ASR -> Qwen2.5 LLM Refiner -> Normalizer -> Entity Extraction -> ABDM FHIR Mapper -> Validator
    print("\n[2/3] Processing audio through Ambient Scribing Pipeline...")
    print("      -> Speech Service: Silero VAD + Whisper ASR")
    print("      -> Refinement Layer: Qwen2.5-1.5B Clinical LLM Refiner")
    print("      -> Extraction & Mapping: ABDM OPConsultRecord FHIR R4")

    service = ClinicalService(enable_llm_refiner=True)
    validated_bundle = service.process_audio(wav_path, session_id="LIVE_SESSION_01")

    # Save FHIR Bundle
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(validated_bundle.model_dump_json(indent=2), encoding="utf-8")

    print("\n[3/3] Scribing Completed Successfully!")
    print("=======================================================")
    print(f"  FHIR Bundle Saved : {out_file}")
    print(f"  ABDM Schema Valid : {validated_bundle.validation.schema_valid}")
    print(f"  Validation Errors : {len(validated_bundle.validation.errors)}")
    print(f"  Grounding Flags   : {len(validated_bundle.grounding_flags)}")

    entries = validated_bundle.fhir_bundle.get("entry", [])
    resource_types = [e.get("resource", {}).get("resourceType") for e in entries if isinstance(e, dict)]
    print(f"  FHIR Resources    : {', '.join(set(resource_types))}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
