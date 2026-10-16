"""Comprehensive Unit and Integration Tests for Person B: Clinical NLP & FHIR Service."""

from clinical_service.extraction.temporal_med_logic import (
    classify_medication_status,
    extract_dose,
    extract_duration,
    extract_frequency,
    is_negated,
)
from clinical_service.fhir_mapper.mapper import FHIRMapper
from clinical_service.schemas.interface1_transcript import Transcript, TranscriptSegment
from clinical_service.schemas.interface2_extraction import (
    ClinicalEntities,
    ClinicalExtraction,
    MedicationEntity,
    SymptomEntity,
    VitalEntity,
)
from clinical_service.schemas.interface3_bundle import ValidatedBundle
from clinical_service.service import ClinicalService
from clinical_service.terminology.lookup import get_terminology_lookup
from clinical_service.validation.grounding_checker import GroundingChecker


def test_interface1_transcript_schema():
    seg = TranscriptSegment(
        segment_id=0,
        start=0.0,
        end=4.2,
        speaker="doctor",
        text="Aapko kab se fever hai?",
        lang="hi-en",
        confidence=0.91,
    )
    t = Transcript(session_id="S001", language_mix="hi-en", segments=[seg])
    assert t.session_id == "S001"
    assert len(t.segments) == 1
    assert t.segments[0].speaker == "doctor"


def test_temporal_med_logic():
    # Negation
    assert is_negated("khansi bilkul nahi hai") is True
    assert is_negated("chhati me dard nahi ho raha") is True
    assert is_negated("no cough or cold reported") is True
    assert is_negated("tez bukhar hai") is False

    # Status
    assert classify_medication_status("pichle do saal se le raha hoon") == "existing"
    assert classify_medication_status("purani dawai regular chal rahi hai") == "existing"
    assert classify_medication_status("aaj se yeh shuru karo 3 din tak") == "new"
    assert classify_medication_status("purani wali band kar do") == "discontinued"

    # Dose & Frequency
    assert extract_dose("Dolo 650 mg lijiye") == "650 mg"
    assert extract_dose("1 tablet subah sham") == "1 tablet"
    assert extract_frequency("din me 3 baar khana khane ke baad") == "TDS"
    assert extract_frequency("subah sham ek ek goli") == "BD"
    assert extract_frequency("daily once in morning") == "OD"
    assert extract_duration("3 din tak lena") == "3 days"


def test_terminology_lookup():
    lookup = get_terminology_lookup()

    # SNOMED
    snomed_fever = lookup.lookup_snomed("fever")
    assert snomed_fever["code"] == "386661006"
    assert snomed_fever["system"] == "http://snomed.info/sct"

    snomed_bukhar = lookup.lookup_snomed("bukhar")
    assert snomed_bukhar["code"] == "386661006"

    # LOINC
    loinc_temp = lookup.lookup_loinc("temperature")
    assert loinc_temp["code"] == "8310-5"

    loinc_bp = lookup.lookup_loinc("blood_pressure_sys")
    assert loinc_bp["code"] == "8480-6"

    # RxNorm
    rx_para = lookup.lookup_rxnorm("paracetamol")
    assert rx_para["code"] == "7052"


def test_fhir_mapper_deterministic():
    extraction = ClinicalExtraction(
        session_id="S001",
        entities=ClinicalEntities(
            symptoms=[
                SymptomEntity(text="fever", duration="3 days", source_segments=[1]),
                SymptomEntity(text="cough", negated=True, source_segments=[2]),
            ],
            diagnosis=[],
            medications=[
                MedicationEntity(
                    name="Paracetamol",
                    dose="650 mg",
                    frequency="TDS",
                    duration="3 days",
                    status="new",
                    instructions="after food",
                    source_segments=[3],
                )
            ],
            vitals=[
                VitalEntity(type="temperature", value=101.4, unit="degF", source_segments=[1])
            ],
            investigations=[],
        ),
    )

    mapper = FHIRMapper()
    bundle = mapper.map_to_fhir_bundle(extraction)

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "document"
    assert len(bundle["entry"]) >= 4

    first_resource = bundle["entry"][0]["resource"]
    assert first_resource["resourceType"] == "Composition"
    assert first_resource["type"]["coding"][0]["code"] == "371530004"


def test_grounding_checker_catches_hallucinations():
    transcript = Transcript(
        session_id="S001",
        segments=[
            TranscriptSegment(start=0.0, end=3.0, speaker="patient", text="Mujhe bukhar hai.", lang="hi-en"),
            TranscriptSegment(start=3.1, end=6.0, speaker="doctor", text="Dolo 650 mg lijiye.", lang="hi-en"),
        ],
    )

    hallucinated_extraction = ClinicalExtraction(
        session_id="S001",
        entities=ClinicalEntities(
            medications=[
                MedicationEntity(
                    name="Dolo",
                    dose="1000 mg",
                    source_segments=[1],
                )
            ]
        ),
    )

    checker = GroundingChecker()
    flags = checker.verify(transcript, hallucinated_extraction)
    assert len(flags) > 0
    assert any("dose '1000 mg' not found" in f.issue for f in flags)


def test_end_to_end_clinical_service():
    service = ClinicalService()

    transcript_dict = {
        "session_id": "TEST_SESSION_101",
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "doctor", "text": "Aapko kya takleef hai?", "lang": "hi-en"},
            {"start": 2.1, "end": 6.5, "speaker": "patient", "text": "Doctor 3 din se bukhar hai aur badan me dard hai. Khansi nahi hai.", "lang": "hi-en"},
            {"start": 6.6, "end": 10.0, "speaker": "doctor", "text": "Temperature 101.2 F hai.", "lang": "hi-en"},
            {"start": 10.1, "end": 14.5, "speaker": "doctor", "text": "Yeh viral fever lag raha hai. Paracetamol 650 mg TDS lijiye 3 days tak.", "lang": "hi-en"},
        ],
    }

    validated_bundle: ValidatedBundle = service.process_transcript(transcript_dict)

    assert validated_bundle.session_id == "TEST_SESSION_101"
    assert validated_bundle.validation.schema_valid is True
    assert len(validated_bundle.validation.errors) == 0
    assert validated_bundle.fhir_bundle["type"] == "document"

    entities = service.extract_entities_only(transcript_dict).entities
    symptom_names = [s.text for s in entities.symptoms]
    assert "fever" in symptom_names
    assert "body ache" in symptom_names

    cough_item = next(s for s in entities.symptoms if s.text == "cough")
    assert cough_item.negated is True

    med = entities.medications[0]
    assert "Paracetamol" in med.name
    assert med.dose == "650 mg"
    assert med.frequency == "TDS"
    assert med.status == "new"


def test_speech_confidence_and_speaker_plausibility_grounding():
    transcript = Transcript(
        session_id="S_SPEECH_01",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                speaker="patient",
                text="Main pehle se Rozavel leta hoon.",
                confidence=0.45,
                lang="hi-en",
            ),
            TranscriptSegment(
                start=2.1,
                end=5.0,
                speaker="doctor",
                text="Theek hai, check karte hain.",
                confidence=0.95,
                lang="hi-en",
            ),
        ],
    )

    extraction = ClinicalExtraction(
        session_id="S_SPEECH_01",
        entities=ClinicalEntities(
            medications=[
                MedicationEntity(
                    name="Rozavel",
                    status="new",
                    source_segments=[0],
                )
            ]
        ),
    )

    checker = GroundingChecker()
    flags = checker.verify(transcript, extraction)

    assert len(flags) >= 2
    assert any("low-confidence speech turn" in f.issue for f in flags)
    assert any("patient speech turn without doctor confirmation" in f.issue for f in flags)


def test_medical_transcript_normalizer():
    from clinical_service.extraction.normalizer import get_medical_normalizer

    normalizer = get_medical_normalizer()

    # Test 1: cleans foreign script hallucinations
    raw1 = "G-celldior, khansi, huihe موجی"
    cleaned1 = normalizer.normalize_sentence(raw1)
    assert "khansi hui hai" in cleaned1
    assert "موجی" not in cleaned1

    # Test 2: cleans fuzzy drug names
    raw2 = "azithromycin car 500 mg lena hai"
    cleaned2 = normalizer.normalize_sentence(raw2)
    assert "azithromycin" in cleaned2

    # Test 3: cleans phonetic slang
    raw3 = "dolohyv any symptoms of fever"
    cleaned3 = normalizer.normalize_sentence(raw3)
    assert "do you have" in cleaned3
    assert "fever" in cleaned3
