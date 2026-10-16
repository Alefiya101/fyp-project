"""Medical Transcript Normalization and Restoration Engine.

Implements the fallback specified in Project Plan Section 10:
'Hinglish WER too high for clean extraction -> adds text normalisation
and fuzzy medical-term correction.'

Chains:
1. Unicode Artifact Filter: strips hallucinated non-Latin/non-Devanagari characters.
2. Phonetic & Slang Normalizer: standardizes spoken Hinglish OPD utterances.
3. Medical Fuzzy Term Matcher: recovers garbled drug, symptom, and investigation names.
4. Transcript-level orchestrator: cleans Interface 1 transcripts before entity extraction.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from clinical_service.schemas.interface1_transcript import Transcript, TranscriptSegment
from clinical_service.terminology.lookup import get_terminology_lookup

# Common phonetic Hinglish phrase corrections
PHONETIC_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # Doctor / Patient greetings & questions
    (re.compile(r"\b(?:dapu\s*sa|dr\.?\s*sa+b|doctor\s*sahb)\b", re.IGNORECASE), "doctor sahab"),
    (re.compile(r"\b(?:gi[aà]\s*hoa\s*hay|kya\s*hoa\s*hai|kya\s*hua\s*hay)\b", re.IGNORECASE), "kya hua hai"),
    (re.compile(r"\b(?:app\s*c[oố]|app\s*ko|aap\s*co)\b", re.IGNORECASE), "aapko"),
    (re.compile(r"\b(?:dolohyv|doyouhave|do\s*u\s*have)\b", re.IGNORECASE), "do you have"),
    (re.compile(r"\b(?:mujee|mujhe\s*bhi|muji)\b", re.IGNORECASE), "mujhe"),
    # Symptoms & findings
    (re.compile(r"\b(?:khansi[,\s]*huihe|khansi\s*hui\s*he|khasiyar)\b", re.IGNORECASE), "khansi hui hai"),
    (re.compile(r"\b(?:sardi\s*or\s*khansi|santi\s*or\s*khansi)\b", re.IGNORECASE), "sardi aur khansi"),
    (re.compile(r"\b(?:badan\s*me\s*dart|badan\s*dart)\b", re.IGNORECASE), "badan dard"),
    (re.compile(r"\b(?:sar\s*me\s*dart|sir\s*dart)\b", re.IGNORECASE), "sar dard"),
    (re.compile(r"\b(?:saans\s*fulna|saas\s*phulna)\b", re.IGNORECASE), "saans phoolna"),
    (re.compile(r"\b(?:gala\s*khrab|gale\s*me\s*dart)\b", re.IGNORECASE), "gala kharab"),
    # Timings & directions
    (re.compile(r"\b(?:sube\s*or\s*sham|subah\s*sham)\b", re.IGNORECASE), "subah aur sham"),
    (re.compile(r"\b(?:khani\s*ke\s*baal|khane\s*k\s*baad|khana\s*khake)\b", re.IGNORECASE), "khane ke baad"),
    (re.compile(r"\b(?:khali\s*pet|khali\s*pet\s*lena)\b", re.IGNORECASE), "khali pet"),
    (re.compile(r"\b(?:din\s*me\s*dova|din\s*me\s*do\s*bar)\b", re.IGNORECASE), "din me 2 baar"),
    (re.compile(r"\b(?:din\s*me\s*tin\s*bar|din\s*me\s*3\s*bar)\b", re.IGNORECASE), "din me 3 baar"),
    (re.compile(r"\b(?:sarurak\s*padi\s*to|zaroorat\s*pada\s*toh)\b", re.IGNORECASE), "zaroorat padi toh"),
    (re.compile(r"\b(?:liktetahum|likh\s*deta\s*hum)\b", re.IGNORECASE), "likh deta hoon"),
    (re.compile(r"\b(?:karwal\s*lehnge|karwa\s*lenge)\b", re.IGNORECASE), "karwa lenge"),
]

# Strips stray non-speech non-Devanagari scripts (e.g. Hebrew/Arabic/Korean/Vietnamese artifacts)
STRAY_SCRIPT_RE = re.compile(r"[\u0590-\u05FF\u0600-\u06FF\uAC00-\uD7AF\u1EA0-\u1EF9]+")


class MedicalTranscriptNormalizer:
    """Restores and normalizes noisy ASR transcripts for OPD consultations."""

    def __init__(self, lexicon_path: str | Path | None = None):
        self.lookup = get_terminology_lookup()
        self.canonical_terms: list[str] = []

        # 1. Load canonical medical terms from terminology lookup
        self.canonical_terms.extend(self.lookup.snomed.keys())
        self.canonical_terms.extend(self.lookup.rxnorm.keys())
        self.canonical_terms.extend(self.lookup.loinc.keys())

        # 2. Load custom terms from medical_lexicon.txt if available
        if lexicon_path and Path(lexicon_path).is_file():
            try:
                lines = Path(lexicon_path).read_text(encoding="utf-8").splitlines()
                for line in lines:
                    term = line.strip().lower()
                    if term and not term.startswith("#") and term not in self.canonical_terms:
                        self.canonical_terms.append(term)
            except Exception:  # noqa: BLE001, S110
                pass

    def clean_text(self, text: str) -> str:
        """Removes stray Unicode hallucinations and normalizes punctuation."""
        if not text:
            return ""

        # Remove stray non-Latin/non-Devanagari scripts
        cleaned = STRAY_SCRIPT_RE.sub(" ", text)

        # Remove leading/trailing commas and stray colons
        cleaned = re.sub(r"^[,\s.:;]+", "", cleaned)
        cleaned = re.sub(r"[,\s.:;]+$", "", cleaned)

        # Normalize multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def correct_phonetics(self, text: str) -> str:
        """Applies regex-based phonetic corrections for spoken Hinglish."""
        result = text
        for pat, replacement in PHONETIC_CORRECTIONS:
            result = pat.sub(replacement, result)
        return result

    def correct_fuzzy_medical_terms(self, text: str) -> str:
        """Fuzzy matches broken or misspelled medical terms against the clinical lexicon."""
        words = text.split()
        corrected_words: list[str] = []

        i = 0
        while i < len(words):
            word = words[i].lower().strip(".,;:?!")
            match_found = False

            # Check 2-word phrase first (e.g. 'sar dard', 'blood pressure')
            if i + 1 < len(words):
                two_word = f"{word} {words[i + 1].lower().strip('.,;:?!')}"
                matches = difflib.get_close_matches(two_word, self.canonical_terms, n=1, cutoff=0.85)
                if matches:
                    corrected_words.append(matches[0])
                    i += 2
                    match_found = True
                    continue

            # Check 1-word match for words with length >= 4
            if not match_found and len(word) >= 4:
                matches = difflib.get_close_matches(word, self.canonical_terms, n=1, cutoff=0.82)
                if matches:
                    corrected_words.append(matches[0])
                    i += 1
                    match_found = True
                    continue

            corrected_words.append(words[i])
            i += 1

        return " ".join(corrected_words)

    def normalize_sentence(self, text: str) -> str:
        """Runs the full normalization pipeline on a single utterance."""
        t = self.clean_text(text)
        t = self.correct_phonetics(t)
        t = self.correct_fuzzy_medical_terms(t)
        return t

    def normalize_transcript(self, transcript: Transcript | dict) -> Transcript | dict:
        """Normalizes all segments inside an Interface 1 Transcript (model or dict)."""
        if isinstance(transcript, dict):
            normalized_segments = []
            for seg in transcript.get("segments", []):
                raw_text = seg.get("text", "") if isinstance(seg, dict) else seg.text
                cleaned_text = self.normalize_sentence(raw_text)
                new_seg = dict(seg) if isinstance(seg, dict) else seg.model_dump()
                new_seg["text"] = cleaned_text
                normalized_segments.append(new_seg)
            res = dict(transcript)
            res["segments"] = normalized_segments
            return res

        normalized_model_segments: list[TranscriptSegment] = []
        for seg in transcript.segments:
            cleaned_text = self.normalize_sentence(seg.text)
            if not cleaned_text:
                continue
            normalized_seg = seg.model_copy(update={"text": cleaned_text})
            normalized_model_segments.append(normalized_seg)

        return transcript.model_copy(update={"segments": normalized_model_segments})


_GLOBAL_NORMALIZER: MedicalTranscriptNormalizer | None = None


def get_medical_normalizer() -> MedicalTranscriptNormalizer:
    """Returns the singleton instance of the medical transcript normalizer."""
    global _GLOBAL_NORMALIZER
    if _GLOBAL_NORMALIZER is None:
        lexicon_candidates = [
            Path("data/medical_lexicon.txt"),
            Path(__file__).resolve().parent.parent.parent / "data" / "medical_lexicon.txt",
        ]
        lex_path = next((p for p in lexicon_candidates if p.is_file()), None)
        _GLOBAL_NORMALIZER = MedicalTranscriptNormalizer(lexicon_path=lex_path)
    return _GLOBAL_NORMALIZER


def normalize_transcript(transcript: Transcript) -> Transcript:
    """Public helper to normalize an Interface 1 Transcript."""
    normalizer = get_medical_normalizer()
    return normalizer.normalize_transcript(transcript)
