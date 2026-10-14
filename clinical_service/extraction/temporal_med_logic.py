"""Temporal and Medication Logic for Clinical Extraction.

Handles:
1. Status classification: 'new' vs 'existing' vs 'discontinued' vs 'modified'.
2. Dosage, frequency, and duration pattern extraction and normalization.
3. Food/timing instruction parsing.
4. Robust Hindi/English negation detection for clinical symptoms.
"""

from __future__ import annotations

import re

# Negation keywords & patterns
NEGATION_PATTERNS = [
    re.compile(r"\b(?:nahi|nhi|na|mat)\b\s+(?:hai|tha|thi|the)?", re.IGNORECASE),
    re.compile(r"\b(?:no|not|denies|denied|without|ruled out|negative for)\b", re.IGNORECASE),
    re.compile(r"\bkoi\s+(?:bhi\s+)?(?:dikkat|problem|takleef)\s+nahi\b", re.IGNORECASE),
    re.compile(r"\bbilkul\s+(?:bhi\s+)?nahi\b", re.IGNORECASE),
]

# Temporal medication indicators
EXISTING_MED_CUES = [
    re.compile(r"\b(?:pehle\s+se|chal\s+rahi\s+hai|regular\s+le|daily\s+leta|roz\s+leta|purani\s+dawai)\b", re.IGNORECASE),
    re.compile(r"\b(?:already\s+taking|currently\s+on|taking\s+since|past\s+medication|home\s+medication)\b", re.IGNORECASE),
    re.compile(r"\b(?:saal\s+se|mahine\s+se)\s+(?:chal|le)\b", re.IGNORECASE),
]

DISCONTINUED_MED_CUES = [
    re.compile(r"\b(?:band\s+kar\s+do|rok\s+do|mat\s+lena\s+ab|ab\s+nahi\s+lena)\b", re.IGNORECASE),
    re.compile(r"\b(?:stop\s+taking|discontinue|cease|stop\s+this)\b", re.IGNORECASE),
]

MODIFIED_MED_CUES = [
    re.compile(r"\b(?:dose\s+badha|dose\s+kam|badha\s+dete|kam\s+kar)\b", re.IGNORECASE),
    re.compile(r"\b(?:increase\s+dose|decrease\s+dose|change\s+to|switch\s+to)\b", re.IGNORECASE),
]

NEW_MED_CUES = [
    re.compile(r"\b(?:likh\s+raha\s+hoon|likh\s+diya|yeh\s+shuru\s+karo|aaj\s+se\s+lena|din\s+tak\s+lena)\b", re.IGNORECASE),
    re.compile(r"\b(?:prescribing|start\s+taking|take\s+for|take\s+this)\b", re.IGNORECASE),
]

# Frequency normalization map
FREQ_PATTERNS = [
    (re.compile(r"\b(?:thrice\s+a\s+day|three\s+times\s+(?:a\s+)?day|3\s+times|din\s+me\s+3\s+baar|subah\s+dopahar\s+(?:aur\s+)?sha?am|tds)\b", re.IGNORECASE), "TDS"),
    (re.compile(r"\b(?:twice\s+a\s+day|two\s+times\s+(?:a\s+)?day|2\s+times|din\s+me\s+2\s+baar|subah\s+sha?am|subah\s+aur\s+raat|bd)\b", re.IGNORECASE), "BD"),
    (re.compile(r"\b(?:once\s+a\s+day|one\s+time\s+(?:a\s+)?day|1\s+time|din\s+me\s+1\s+baar|rozana\s+ek|daily\s+once|od)\b", re.IGNORECASE), "OD"),
    (re.compile(r"\b(?:four\s+times\s+(?:a\s+)?day|4\s+times|din\s+me\s+4\s+baar|qid)\b", re.IGNORECASE), "QID"),
    (re.compile(r"\b(?:as\s+needed|when\s+needed|sos|jab\s+zaroorat\s+ho|dard\s+hone\s+par|bukhar\s+aane\s+par)\b", re.IGNORECASE), "SOS"),
]

# Dose patterns
DOSE_REGEX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|milligram|gm|gram|ml|mcg|microgram|tablet|tab|capsule|cap|puff|puffs|drop|drops|unit|units|iu)\b",
    re.IGNORECASE,
)

# Duration patterns
DURATION_REGEX = re.compile(
    r"\b(\d+|ek|do|teen|char|paanch|one|two|three|four|five)\s*(days?|din|weeks?|hafte?|hafta|months?|mahina|mahine)\b",
    re.IGNORECASE,
)

WORD_NUMBERS = {
    "ek": "1", "do": "2", "teen": "3", "char": "4", "paanch": "5",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5"
}

# Instruction patterns
INSTRUCTION_PATTERNS = [
    (re.compile(r"\b(?:after\s+food|after\s+meals|khana\s+khane\s+ke\s+baad|khane\s+ke\s+baad)\b", re.IGNORECASE), "after food"),
    (re.compile(r"\b(?:before\s+food|empty\s+stomach|khana\s+khane\s+se\s+pehle|khali\s+pet|subah\s+khali\s+pet)\b", re.IGNORECASE), "empty stomach / before food"),
    (re.compile(r"\b(?:at\s+bedtime|before\s+sleeping|raat\s+ko\s+sote\s+samay|sone\s+se\s+pehle)\b", re.IGNORECASE), "at bedtime"),
]


def is_negated(text: str, symptom_span: tuple[int, int] | None = None) -> bool:
    """Detects whether a symptom mention is negated in Hindi, English, or Hinglish."""
    if symptom_span:
        start, end = symptom_span
        window_start = max(0, start - 40)
        window_end = min(len(text), end + 40)
        snippet = text[window_start:window_end]
    else:
        snippet = text

    for pat in NEGATION_PATTERNS:
        if pat.search(snippet):
            return True
    return False


def classify_medication_status(context_text: str) -> str:
    """Classifies medication status into 'new', 'existing', 'discontinued', or 'modified'."""
    for cue in DISCONTINUED_MED_CUES:
        if cue.search(context_text):
            return "discontinued"
    for cue in MODIFIED_MED_CUES:
        if cue.search(context_text):
            return "modified"
    for cue in EXISTING_MED_CUES:
        if cue.search(context_text):
            return "existing"
    return "new"


def extract_dose(text: str) -> str | None:
    """Extracts dosage from text (e.g. '650 mg', '1 tablet')."""
    m = DOSE_REGEX.search(text)
    if m:
        val, unit = m.group(1), m.group(2).lower()
        if unit in ["tab", "tablet"]:
            return f"{val} tablet"
        if unit in ["cap", "capsule"]:
            return f"{val} capsule"
        if unit in ["mg", "milligram"]:
            return f"{val} mg"
        return f"{val} {unit}"
    return None


def extract_frequency(text: str) -> str | None:
    """Extracts and normalizes administration frequency."""
    for pat, freq_code in FREQ_PATTERNS:
        if pat.search(text):
            return freq_code
    return None


def extract_duration(text: str) -> str | None:
    """Extracts duration string (e.g. '3 days', '1 week')."""
    m = DURATION_REGEX.search(text)
    if m:
        val_str, unit_str = m.group(1).lower(), m.group(2).lower()
        val = WORD_NUMBERS.get(val_str, val_str)
        if "din" in unit_str or "day" in unit_str:
            unit = "days"
        elif "haft" in unit_str or "week" in unit_str:
            unit = "weeks"
        elif "mahin" in unit_str or "month" in unit_str:
            unit = "months"
        else:
            unit = unit_str
        return f"{val} {unit}"
    return None


def extract_instruction(text: str) -> str | None:
    """Extracts food/timing instructions."""
    for pat, inst in INSTRUCTION_PATTERNS:
        if pat.search(text):
            return inst
    return None
