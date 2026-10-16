"""Clinical Entity Extraction Engine.

Implements:
1. RuleBasedSLMExtractor: Fast, deterministic, offline extraction with precise
   segment attribution, negation handling, and temporal medication logic.
2. LlamaCppSLMExtractor: GGUF SLM inference with grammar/JSON-schema constraints.
3. ClinicalExtractor: Unified facade with automatic fallback.
"""

from __future__ import annotations

import abc
import importlib
import json
import re

from clinical_service.extraction.prompt_templates import build_extraction_prompt
from clinical_service.extraction.temporal_med_logic import (
    classify_medication_status,
    extract_dose,
    extract_duration,
    extract_frequency,
    extract_instruction,
    is_negated,
)
from clinical_service.schemas.interface1_transcript import Transcript
from clinical_service.schemas.interface2_extraction import (
    ClinicalEntities,
    ClinicalExtraction,
    DiagnosisEntity,
    InvestigationEntity,
    MedicationEntity,
    SymptomEntity,
    VitalEntity,
)
from clinical_service.terminology.lookup import get_terminology_lookup


class BaseExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, transcript: Transcript) -> ClinicalExtraction:
        """Extracts structured clinical data from a transcript."""


class RuleBasedSLMExtractor(BaseExtractor):
    """Deterministic, high-precision extractor used for offline edge deployments,
    testing, and as a fallback when SLM weights are not downloaded."""

    def __init__(self):
        self.lookup = get_terminology_lookup()

        # Regular expressions for vitals
        self.temp_re = re.compile(
            r"\b(?:temp(?:erature)?|bukhar|fever)\s*(?:hai|is)?\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:f|c|deg|fahrenheit|celsius)?\b|\b(\d{2,3}(?:\.\d+)?)\s*(?:f|fahrenheit|degree fahrenheit|c|celsius)\b",
            re.IGNORECASE,
        )
        self.bp_re = re.compile(
            r"\b(?:bp|blood\s+pressure)\s*(?:hai|is)?\s*[:=]?\s*(\d{2,3})\s*(?:/|\s+by\s+)\s*(\d{2,3})\b|\b(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*hg)?\b",
            re.IGNORECASE,
        )
        self.pulse_re = re.compile(
            r"\b(?:pulse|heart\s+rate|hr)\s*(?:hai|is)?\s*[:=]?\s*(\d{2,3})\b|\b(\d{2,3})\s*(?:bpm|beats\s+per\s+minute)\b",
            re.IGNORECASE,
        )
        self.spo2_re = re.compile(
            r"\b(?:spo2|oxygen|saturation|o2)\s*(?:hai|is)?\s*[:=]?\s*(\d{2,3})\s*%?\b",
            re.IGNORECASE,
        )
        self.weight_re = re.compile(
            r"\b(?:weight|wajan)\s*(?:hai|is)?\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kilos?)\b",
            re.IGNORECASE,
        )

        # Common symptom dictionary keys
        self.symptom_keywords = [
            ("fever", ["fever", "bukhar", "pyrexia", "tap", "bijkhar"]),
            ("cough", ["cough", "khansi"]),
            ("productive cough", ["productive cough", "balgam wali khansi", "wet cough", "phlegm"]),
            ("cold", ["cold", "sardi", "jukham"]),
            ("runny nose", ["runny nose", "naak behna", "naak beh rahi", "naak beh", "nakhmehri", "nakberi", "rhinorrhea", "watery nasal", "running nose"]),
            ("headache", ["headache", "sar dard", "sar me dard", "sir dard", "sir me dard", "sermidhar", "sermidar", "head ache"]),
            ("abdominal pain", ["abdominal pain", "pet dard", "pet me dard", "stomach pain", "stomach ache", "pet kharab", "veitmehdar"]),
            ("sneezing", ["sneezing", "chheek", "chheenk", "chheekein", "chheekar", "chheekh", "tkr", "chhi kary"]),
            ("vomiting", ["vomiting", "ulti"]),
            ("nausea", ["nausea", "ji ghabrana", "mitli"]),
            ("loose motion", ["loose motion", "diarrhea", "diarrhoea", "dast"]),
            ("chest pain", ["chest pain", "chhati me dard", "seene me dard"]),
            ("breathlessness", ["breathlessness", "saans phoolna", "dyspnea", "saans lene me takleef"]),
            ("sore throat", ["sore throat", "gale me dard", "gale me kharash"]),
            ("body ache", ["body ache", "badan dard", "badan me dard"]),
            ("weakness", ["weakness", "kamzori", "kamjori", "thakan", "fatigue"]),
            ("loss of appetite", ["loss of appetite", "bhookh kam", "bhookh nahi", "anorexia", "appetite"]),
            ("chills", ["chills", "thandi", "kapkapi", "shivering"]),
            ("burning urination", ["burning urination", "peshab me jalan", "dysuria"]),
            ("acidity", ["acidity", "jalan", "pet me jalan", "gas"]),
            ("joint pain", ["joint pain", "jodon me dard", "arthralgia"]),
        ]

        # Common diagnosis keywords
        self.diagnosis_keywords = [
            ("viral fever", ["viral fever", "viral infection", "viral"]),
            ("acute bronchitis", ["bronchitis"]),
            ("upper respiratory tract infection", ["urti", "upper respiratory infection", "throat infection"]),
            ("essential hypertension", ["hypertension", "high bp", "blood pressure ki bimari"]),
            ("type 2 diabetes mellitus", ["type 2 diabetes", "diabetes", "sugar ki bimari", "diabetes mellitus"]),
            ("asthma", ["asthma", "dama"]),
            ("gastroenteritis", ["gastroenteritis", "food poisoning", "pet ka infection"]),
        ]

        # Common investigation keywords
        self.investigation_keywords = [
            ("Complete Blood Count (CBC)", ["cbc", "complete blood count", "blood test"]),
            ("Chest X-ray", ["x-ray", "xray", "chest x-ray", "chest xray"]),
            ("HbA1c", ["hba1c", "glycated hemoglobin"]),
            ("Fasting Blood Sugar", ["fasting blood sugar", "fbs", "sugar test"]),
            ("ECG", ["ecg", "ekg", "electrocardiogram"]),
            ("Urine Routine", ["urine routine", "urine test"]),
        ]

    def extract(self, transcript: Transcript) -> ClinicalExtraction:
        symptoms_map: dict[str, SymptomEntity] = {}
        diagnosis_map: dict[str, DiagnosisEntity] = {}
        medications_map: dict[str, MedicationEntity] = {}
        vitals_list: list[VitalEntity] = []
        investigations_list: list[InvestigationEntity] = []

        segments = transcript.segments

        for i, seg in enumerate(segments):
            seg_idx = seg.segment_id if seg.segment_id is not None else i
            text = seg.text
            lower_text = text.lower()
            speaker = seg.speaker.lower() if seg.speaker else ""

            # 1. Symptoms Extraction
            for sym_std, patterns in self.symptom_keywords:
                for pat in patterns:
                    m = re.search(r"\b" + re.escape(pat) + r"\b", lower_text)
                    if m:
                        span = m.span()
                        neg = is_negated(lower_text, span)
                        if not neg and speaker == "doctor" and i + 1 < len(segments):
                            next_text = segments[i + 1].text.lower()
                            if is_negated(next_text) or re.search(r"\b(?:nahi|no|na|woh sab nahi|kuch nahi)\b", next_text):
                                neg = True
                        dur = extract_duration(text)

                        sev = None
                        if re.search(r"\b(?:tez|severe|high|bohot|bahut)\b", lower_text):
                            sev = "severe"
                        elif re.search(r"\b(?:halka|mild|thoda)\b", lower_text):
                            sev = "mild"

                        if sym_std not in symptoms_map:
                            symptoms_map[sym_std] = SymptomEntity(
                                text=sym_std,
                                duration=dur,
                                severity=sev,
                                negated=neg,
                                source_segments=[seg_idx],
                            )
                        else:
                            curr = symptoms_map[sym_std]
                            if seg_idx not in curr.source_segments:
                                curr.source_segments.append(seg_idx)
                            if not curr.duration and dur:
                                curr.duration = dur
                            if not curr.severity and sev:
                                curr.severity = sev
                            if neg and not curr.negated:
                                curr.negated = True
                        break

            # 2. Diagnosis Extraction
            for diag_std, patterns in self.diagnosis_keywords:
                for pat in patterns:
                    if re.search(r"\b" + re.escape(pat) + r"\b", lower_text):
                        conf = "medium"
                        if re.search(r"\b(?:confirm|pakka|definitely)\b", lower_text):
                            conf = "high"
                        elif re.search(r"\b(?:lagta hai|suspect|maybe|ho sakta)\b", lower_text):
                            conf = "provisional"

                        if diag_std not in diagnosis_map:
                            diagnosis_map[diag_std] = DiagnosisEntity(
                                text=diag_std,
                                confidence=conf,
                                status="provisional",
                                source_segments=[seg_idx],
                            )
                        else:
                            curr = diagnosis_map[diag_std]
                            if seg_idx not in curr.source_segments:
                                curr.source_segments.append(seg_idx)
                        break

            # 3. Medications Extraction
            for med_key in self.lookup.rxnorm:
                if re.search(r"\b" + re.escape(med_key) + r"\b", lower_text):
                    name = med_key.title()
                    context = text
                    if i + 1 < len(segments):
                        context += " " + segments[i + 1].text

                    status = classify_medication_status(context)
                    dose = extract_dose(context)
                    freq = extract_frequency(context)
                    dur = extract_duration(context)
                    inst = extract_instruction(context)

                    if name not in medications_map:
                        medications_map[name] = MedicationEntity(
                            name=name,
                            dose=dose,
                            frequency=freq,
                            duration=dur,
                            status=status,
                            instructions=inst,
                            source_segments=[seg_idx],
                        )
                    else:
                        curr = medications_map[name]
                        if seg_idx not in curr.source_segments:
                            curr.source_segments.append(seg_idx)
                        if not curr.dose and dose:
                            curr.dose = dose
                        if not curr.frequency and freq:
                            curr.frequency = freq
                        if not curr.duration and dur:
                            curr.duration = dur
                        if not curr.instructions and inst:
                            curr.instructions = inst

            # 4. Vitals Extraction
            m_temp = self.temp_re.search(text)
            if m_temp:
                val = float(m_temp.group(1) or m_temp.group(2))
                unit = "degF" if val > 45 else "Cel"
                vitals_list.append(VitalEntity(type="temperature", value=val, unit=unit, source_segments=[seg_idx]))

            m_bp = self.bp_re.search(text)
            if m_bp:
                sys_val = int(m_bp.group(1) or m_bp.group(3))
                dia_val = int(m_bp.group(2) or m_bp.group(4))
                vitals_list.append(VitalEntity(type="blood_pressure_sys", value=sys_val, unit="mm[Hg]", source_segments=[seg_idx]))
                vitals_list.append(VitalEntity(type="blood_pressure_dia", value=dia_val, unit="mm[Hg]", source_segments=[seg_idx]))

            m_pulse = self.pulse_re.search(text)
            if m_pulse:
                val = int(m_pulse.group(1) or m_pulse.group(2))
                vitals_list.append(VitalEntity(type="pulse", value=val, unit="/min", source_segments=[seg_idx]))

            m_spo2 = self.spo2_re.search(text)
            if m_spo2:
                val = int(m_spo2.group(1))
                vitals_list.append(VitalEntity(type="spo2", value=val, unit="%", source_segments=[seg_idx]))

            m_wt = self.weight_re.search(text)
            if m_wt:
                val = float(m_wt.group(1))
                vitals_list.append(VitalEntity(type="weight", value=val, unit="kg", source_segments=[seg_idx]))

            # 5. Investigations Extraction
            for inv_std, patterns in self.investigation_keywords:
                for pat in patterns:
                    if re.search(r"\b" + re.escape(pat) + r"\b", lower_text):
                        st = "ordered"
                        if re.search(r"\b(?:agar|if|badhe|na ho to)\b", lower_text):
                            st = "recommended"
                        elif re.search(r"\b(?:report|result|dekha)\b", lower_text):
                            st = "reviewed"

                        if inv_std not in [x.name for x in investigations_list]:
                            investigations_list.append(
                                InvestigationEntity(name=inv_std, status=st, source_segments=[seg_idx])
                            )
                        break

        entities = ClinicalEntities(
            symptoms=list(symptoms_map.values()),
            diagnosis=list(diagnosis_map.values()),
            medications=list(medications_map.values()),
            vitals=vitals_list,
            investigations=investigations_list,
        )

        return ClinicalExtraction(session_id=transcript.session_id, entities=entities)


class LlamaCppSLMExtractor(BaseExtractor):
    """SLM inference via llama-cpp-python for quantised GGUF models."""

    def __init__(self, model_path: str, n_ctx: int = 4096, temperature: float = 0.1):
        try:
            llama_cpp = importlib.import_module("llama_cpp")
            llama_cls = llama_cpp.Llama
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "llama-cpp-python is an optional dependency required for LlamaCppSLMExtractor. "
                "Install it with: pip install llama-cpp-python"
            ) from exc
        self.llm = llama_cls(model_path=model_path, n_ctx=n_ctx, verbose=False)
        self.temperature = temperature

    def extract(self, transcript: Transcript) -> ClinicalExtraction:
        prompt = build_extraction_prompt(transcript.full_text(), transcript.session_id)
        schema = ClinicalExtraction.model_json_schema()
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a clinical scribe. Output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object", "schema": schema},
            temperature=self.temperature,
        )
        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)
        return ClinicalExtraction.model_validate(data)


class ClinicalExtractor:
    """Unified extractor with graceful fallback to rule-based parser."""

    def __init__(self, backend: str = "rule_based", model_path: str | None = None):
        self.backend = backend
        self.fallback = RuleBasedSLMExtractor()
        self.active_extractor: BaseExtractor = self.fallback

        if backend == "llama_cpp" and model_path:
            try:
                self.active_extractor = LlamaCppSLMExtractor(model_path)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to load llama.cpp model ({e}). Using rule-based extractor.")
                self.active_extractor = self.fallback

    def extract(self, transcript: Transcript) -> ClinicalExtraction:
        try:
            return self.active_extractor.extract(transcript)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Primary extraction failed ({e}), falling back to deterministic extractor.")
            return self.fallback.extract(transcript)
