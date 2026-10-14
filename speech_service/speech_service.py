# -*- coding: utf-8 -*-
"""speech_service.py

Top-level orchestrator: audio file → Interface 1 Transcript JSON.

Interface 1 schema
------------------
{
  "session_id": "S001",
  "language_mix": "hi-en",
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "speaker": "doctor",
      "text": "Aapko kab se fever hai?",
      "lang": "hi-en",
      "confidence": 0.91
    }
  ]
}

Public API
----------
TranscriptSegment   -- dataclass for one speaker turn
SpeechService       -- main class; call process() or transcribe_stream()
transcribe_batch    -- convenience function (single call → dict)
transcribe_stream   -- convenience generator (yields segments one-by-one)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Union

import numpy as np

from speech_service.diarization.diarize import DiarizationSegment, diarize
from speech_service.preprocessing.denoise import AudioLoadError, denoise, load_audio, resample_to_16k
from speech_service.preprocessing.vad import SpeechSegment, get_speech_segments

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """One speaker turn in Interface 1 format."""
    start: float
    end: float
    speaker: str
    text: str
    lang: str = "hi-en"
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# SpeechService
# ---------------------------------------------------------------------------

class SpeechService:
    """Chain preprocessing → diarization → ASR → Interface 1 JSON.

    Models are loaded lazily on the first call to ``process()`` or
    ``transcribe_stream()`` to avoid startup overhead when the class is
    imported but not yet used.
    """

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        device: str = "cpu",
        num_speakers: int = 2,
        vad_threshold: float = 0.5,
        lexicon_path: Optional[str] = None,
    ) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._device = device
        self._num_speakers = num_speakers
        self._vad_threshold = vad_threshold
        self._lexicon_path = lexicon_path
        self._asr_model = None  # lazy

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _get_model(self):
        """Load the ASR model on first access."""
        if self._asr_model is None:
            from faster_whisper import WhisperModel  # noqa: PLC0415
            logger.info(
                "Loading faster-whisper model '%s' (%s, %s)…",
                self._model_size, self._device, self._compute_type,
            )
            self._asr_model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._asr_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """Load audio, resample to 16 kHz, denoise."""
        wav, sr = load_audio(path)
        wav = resample_to_16k(wav, sr)
        wav = denoise(wav, sr=16_000)
        return wav

    @staticmethod
    def _identify_person_a(segments: List[TranscriptSegment]) -> str:
        """Return the speaker label with the most turns (proxy for doctor)."""
        if not segments:
            return ""
        counts: Counter = Counter(s.speaker for s in segments)
        return counts.most_common(1)[0][0]

    @staticmethod
    def _detect_lang(text: str) -> str:
        """Heuristic: detect language tag from script content."""
        try:
            import regex  # noqa: PLC0415
            has_deva = bool(regex.search(r"\p{Devanagari}", text))
        except ImportError:
            has_deva = any("\u0900" <= ch <= "\u097F" for ch in text)

        import re  # noqa: PLC0415
        has_latin = bool(re.search(r"[A-Za-z]", text))
        if has_deva and has_latin:
            return "hi-en"
        if has_deva:
            return "hi"
        return "en-IN"

    def _assign_diarization(
        self,
        vad_segments: List[SpeechSegment],
        dia_segments: List[DiarizationSegment],
    ) -> List[dict]:
        """Map each VAD segment to a speaker from the diarization output."""
        assigned = []
        for vseg in vad_segments:
            mid = (vseg.start + vseg.end) / 2.0
            speaker = "unknown"
            for dseg in dia_segments:
                if dseg.start <= mid <= dseg.end:
                    speaker = dseg.speaker
                    break
            # fallback: nearest diarization segment
            if speaker == "unknown" and dia_segments:
                nearest = min(dia_segments, key=lambda d: min(abs(d.start - mid), abs(d.end - mid)))
                speaker = nearest.speaker
            assigned.append({"start": vseg.start, "end": vseg.end, "speaker": speaker})
        return assigned

    def _transcribe_segments(
        self, wav: np.ndarray, segments: List[dict]
    ) -> List[TranscriptSegment]:
        """Run ASR on each VAD segment and return TranscriptSegment list."""
        model = self._get_model()
        results: List[TranscriptSegment] = []

        for seg in segments:
            start_s, end_s = seg["start"], seg["end"]
            start_i = int(start_s * 16_000)
            end_i = int(end_s * 16_000)
            chunk = wav[start_i:end_i]

            if chunk.size == 0:
                results.append(TranscriptSegment(
                    start=start_s, end=end_s,
                    speaker=seg["speaker"], text="",
                    lang="hi-en", confidence=0.0,
                ))
                continue

            # Pass numpy array directly — avoids soundfile metadata_errors
            # incompatibility when passing a file path to faster-whisper
            try:
                asr_segs, _ = model.transcribe(chunk, beam_size=1, language=None)
                text_parts = []
                confidences = []
                for a in asr_segs:
                    text_parts.append(a.text.strip())
                    if hasattr(a, "avg_logprob"):
                        confidences.append(float(np.exp(a.avg_logprob)))
                text = " ".join(text_parts).strip()
                confidence = float(np.mean(confidences)) if confidences else 1.0
            except Exception as exc:
                logger.warning("ASR failed for segment [%.2f, %.2f]: %s", start_s, end_s, exc)
                text, confidence = "", 0.0

            lang = self._detect_lang(text)
            results.append(TranscriptSegment(
                start=start_s, end=end_s,
                speaker=seg["speaker"], text=text,
                lang=lang, confidence=confidence,
            ))

        return results

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def process(
        self,
        path: str,
        session_id: Optional[str] = None,
        person_a_only: bool = False,
    ) -> dict:
        """Full pipeline: audio file → Interface 1 Transcript JSON dict.

        Parameters
        ----------
        path : str
            Path to the input WAV file.
        session_id : str or None
            Session identifier.  Defaults to the stem of *path*.
        person_a_only : bool
            If True, keep only the speaker with the most turns (doctor).

        Returns
        -------
        dict
            Interface 1 Transcript JSON.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        AudioLoadError
            If the file cannot be decoded.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        sid = session_id or p.stem

        # 1. Load + preprocess
        wav = self._load_and_preprocess(path)

        # 2. VAD
        vad_segs = get_speech_segments(wav, sr=16_000, threshold=self._vad_threshold)
        if not vad_segs:
            logger.info("No speech detected in '%s'.", path)
            return _build_transcript_json(sid, [])

        # 3. Diarization
        dia_segs = diarize(wav, sr=16_000, num_speakers=self._num_speakers)

        # 4. Assign speakers to VAD segments
        assigned = self._assign_diarization(vad_segs, dia_segs)

        # 5. ASR per segment
        transcript_segs = self._transcribe_segments(wav, assigned)

        # 6. Optional person-A filter
        if person_a_only:
            doctor = self._identify_person_a(transcript_segs)
            transcript_segs = [s for s in transcript_segs if s.speaker == doctor]

        return _build_transcript_json(sid, transcript_segs)

    def transcribe_stream(
        self,
        path: str,
        session_id: Optional[str] = None,
    ) -> Iterator[dict]:
        """Yield Interface 1 segment dicts one at a time.

        Preprocessing and diarization are applied to the full file first;
        then ASR results are yielded segment by segment.  A sentinel dict
        ``{"done": True, "session_id": sid}`` is yielded at the end.

        Raises
        ------
        FileNotFoundError
            Before yielding any segments if *path* does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        sid = session_id or p.stem

        wav = self._load_and_preprocess(path)
        vad_segs = get_speech_segments(wav, sr=16_000, threshold=self._vad_threshold)

        if not vad_segs:
            yield {"done": True, "session_id": sid}
            return

        dia_segs = diarize(wav, sr=16_000, num_speakers=self._num_speakers)
        assigned = self._assign_diarization(vad_segs, dia_segs)

        # Yield one segment at a time
        for seg_dict in assigned:
            segs = self._transcribe_segments(wav, [seg_dict])
            for ts in segs:
                yield _segment_to_dict(ts)

        yield {"done": True, "session_id": sid}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _segment_to_dict(ts: TranscriptSegment) -> dict:
    return {
        "start": ts.start,
        "end": ts.end,
        "speaker": ts.speaker,
        "text": ts.text,
        "lang": ts.lang,
        "confidence": ts.confidence,
    }


def _build_transcript_json(session_id: str, segments: List[TranscriptSegment]) -> dict:
    seg_dicts = [_segment_to_dict(s) for s in segments]

    if seg_dicts:
        lang_counts: Counter = Counter(s["lang"] for s in seg_dicts)
        language_mix = lang_counts.most_common(1)[0][0]
    else:
        language_mix = "hi-en"

    return {
        "session_id": session_id,
        "language_mix": language_mix,
        "segments": seg_dicts,
    }


# ---------------------------------------------------------------------------
# Module-level convenience functions (lazy singleton)
# ---------------------------------------------------------------------------

_default_service: Optional[SpeechService] = None


def _get_default_service() -> SpeechService:
    global _default_service
    if _default_service is None:
        _default_service = SpeechService()
    return _default_service


def transcribe_batch(audio_path: str, session_id: str) -> dict:
    """Convenience wrapper: audio_file → Interface 1 Transcript JSON dict."""
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return _get_default_service().process(str(p), session_id=session_id)


def transcribe_stream(audio_path: str, session_id: str) -> Iterator[dict]:
    """Convenience wrapper: yields Interface 1 segment dicts + sentinel."""
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    yield from _get_default_service().transcribe_stream(str(p), session_id=session_id)


# ---------------------------------------------------------------------------
# CLI (quick smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys  # noqa: E401

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Run the full speech pipeline on a WAV file")
    ap.add_argument("audio", help="Path to WAV file")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--stream", action="store_true", help="Use streaming mode")
    ap.add_argument("--person-a-only", action="store_true")
    args = ap.parse_args()

    svc = SpeechService()
    if args.stream:
        for seg in svc.transcribe_stream(args.audio, session_id=args.session_id):
            print(json.dumps(seg, ensure_ascii=False))
    else:
        result = svc.process(args.audio, session_id=args.session_id, person_a_only=args.person_a_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
