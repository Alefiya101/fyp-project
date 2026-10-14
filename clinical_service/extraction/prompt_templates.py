"""Few-shot Prompt Templates for Small Language Models (1-3B parameters).

Optimized for Qwen2.5-1.5B/3B, Phi-3-mini, Gemma-2B, and Llama-3.2-1B/3B
to extract structured Interface 2 clinical JSON from Hinglish OPD transcripts.
"""

SYSTEM_PROMPT = """You are an expert clinical scribe AI for Indian Outpatient Departments (OPD).
Your task is to analyze doctor-patient consultation transcripts in Hinglish (Hindi + English) and extract structured clinical data into JSON strictly conforming to Interface 2.

CRITICAL RULES:
1. Grounding: Every extracted entity MUST cite the exact integer line numbers in `source_segments` where it was mentioned. Do not fabricate or hallucinate any medications, doses, or symptoms.
2. Temporal Medication Status:
   - "new": Prescribed by doctor in this visit.
   - "existing": Patient was already taking this before this visit (e.g. "pehle se chal rahi hai", "regular medicine").
   - "discontinued": Doctor advised stopping.
   - "modified": Dose/frequency changed.
3. Negation: If a symptom is denied or absent (e.g., "khansi nahi hai", "no chest pain"), set `negated: true`.
4. Output Format: Output ONLY valid JSON adhering to the specified schema, without markdown formatting or introductory text.
"""

FEW_SHOT_EXAMPLE_INPUT = """[0] DOCTOR: Namaste, kya takleef ho rahi hai aapko?
[1] PATIENT: Doctor sahab teen din se tez bukhar hai aur badan me bohot dard hai.
[2] DOCTOR: Khansi ya saans lene me koi takleef hai?
[3] PATIENT: Nahi doctor sahab, khansi bilkul nahi hai, bas thoda gala kharab hai.
[4] DOCTOR: Theek hai, temperature dekhte hain... 101.5 Fahrenheit hai. Pulse 84 hai.
[5] DOCTOR: Pehle se koi dawai chal rahi hai aapki?
[6] PATIENT: Haan, sugar ke liye Metformin 500 mg roz subah leta hoon.
[7] DOCTOR: Theek hai. Lagta hai seasonal viral infection hai.
[8] DOCTOR: Main Paracetamol 650 mg likh raha hoon, din me 3 baar khana khane ke baad lijiye 3 din ke liye.
[9] DOCTOR: Aur gale ke liye warm salt water gargle kijiye. Agar bukhar 3 din me theek na ho to CBC test karwa lena."""

FEW_SHOT_EXAMPLE_OUTPUT = """{
  "session_id": "EXAMPLE_001",
  "entities": {
    "symptoms": [
      {"text": "fever", "duration": "3 days", "severity": "severe", "negated": false, "source_segments": [1]},
      {"text": "body ache", "duration": "3 days", "severity": "severe", "negated": false, "source_segments": [1]},
      {"text": "cough", "duration": null, "severity": null, "negated": true, "source_segments": [3]},
      {"text": "sore throat", "duration": null, "severity": "mild", "negated": false, "source_segments": [3]}
    ],
    "diagnosis": [
      {"text": "viral fever", "confidence": "medium", "status": "provisional", "source_segments": [7]}
    ],
    "medications": [
      {
        "name": "Metformin",
        "dose": "500 mg",
        "frequency": "OD",
        "duration": null,
        "status": "existing",
        "instructions": "morning",
        "source_segments": [6]
      },
      {
        "name": "Paracetamol",
        "dose": "650 mg",
        "frequency": "TDS",
        "duration": "3 days",
        "status": "new",
        "instructions": "after food",
        "source_segments": [8]
      }
    ],
    "vitals": [
      {"type": "temperature", "value": 101.5, "unit": "degF", "source_segments": [4]},
      {"type": "pulse", "value": 84, "unit": "/min", "source_segments": [4]}
    ],
    "investigations": [
      {"name": "Complete Blood Count (CBC)", "status": "recommended", "result": null, "source_segments": [9]}
    ]
  }
}"""


def build_extraction_prompt(transcript_text: str, session_id: str) -> str:
    """Constructs prompt for SLM with system instructions, few-shot example, and target transcript."""
    return f"""{SYSTEM_PROMPT}

### Example Consultation
Transcript:
{FEW_SHOT_EXAMPLE_INPUT}

Extracted JSON:
{FEW_SHOT_EXAMPLE_OUTPUT}

### Target Consultation to Extract
Session ID: {session_id}
Transcript:
{transcript_text}

Extracted JSON:
"""
