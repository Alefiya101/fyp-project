"""Local Terminology Lookup for SNOMED CT, LOINC, and RxNorm.

Provides offline deterministic terminology resolution with normalized string
matching and graceful fallback to free-text FHIR coding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TERMINOLOGY_DIR = Path(__file__).parent


class TerminologyLookup:
    def __init__(self, data_dir: Path | None = None):
        dir_path = data_dir or TERMINOLOGY_DIR
        self.snomed = self._load_json(dir_path / "snomed_codes.json")
        self.loinc = self._load_json(dir_path / "loinc_codes.json")
        self.rxnorm = self._load_json(dir_path / "rxnorm_codes.json")

    @staticmethod
    def _load_json(path: Path) -> dict[str, dict[str, str]]:
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _normalize(term: str) -> str:
        t = (term or "").lower().strip()
        t = re.sub(r"[^\w\s-]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def lookup_snomed(self, term: str) -> dict[str, str]:
        """Look up SNOMED CT code for symptom or diagnosis. Falls back to free text."""
        norm = self._normalize(term)
        if norm in self.snomed:
            return self.snomed[norm]
        # Partial match
        for k, v in self.snomed.items():
            if k in norm or norm in k:
                return v
        return {"code": None, "display": term, "system": None}

    def lookup_loinc(self, vital_type: str) -> dict[str, str]:
        """Look up LOINC code for vital sign or lab investigation. Falls back to free text."""
        norm = self._normalize(vital_type)
        if norm in self.loinc:
            return self.loinc[norm]
        for k, v in self.loinc.items():
            if k in norm or norm in k:
                return v
        return {"code": None, "display": vital_type, "system": None}

    def lookup_rxnorm(self, med_name: str) -> dict[str, str]:
        """Look up RxNorm code for medication. Falls back to free text."""
        norm = self._normalize(med_name)
        # Strip strength numbers e.g. "dolo 650" -> "dolo" if "dolo 650" not direct key
        if norm in self.rxnorm:
            return self.rxnorm[norm]
        clean_name = re.sub(r"\b\d+\s*(?:mg|ml|gm|mcg)?\b", "", norm).strip()
        if clean_name in self.rxnorm:
            return self.rxnorm[clean_name]
        for k, v in self.rxnorm.items():
            if k in norm or norm in k:
                return v
        return {"code": None, "display": med_name, "system": None}


# Global singleton
_DEFAULT_LOOKUP: TerminologyLookup | None = None

def get_terminology_lookup() -> TerminologyLookup:
    global _DEFAULT_LOOKUP
    if _DEFAULT_LOOKUP is None:
        _DEFAULT_LOOKUP = TerminologyLookup()
    return _DEFAULT_LOOKUP
