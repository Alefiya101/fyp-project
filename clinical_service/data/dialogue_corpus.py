"""Simulated Hinglish OPD Dialogue Corpus Generator & Benchmark Loader.

Provides 150-250 simulated Indian OPD clinical dialogues with complete gold-standard
entity annotations (Interface 2) and transcript segments (Interface 1) across:
1. Acute Infections & Fevers (Viral fever, typhoid, dengue-like illness)
2. Respiratory & ENT (URTI, bronchitis, pharyngitis, asthma)
3. Endocrine & Metabolic (Type 2 Diabetes, thyroid)
4. Cardiovascular & Lifestyle (Essential hypertension, dyslipidemia)
5. Gastrointestinal (Acute gastroenteritis, GERD, hyperacidity)

Used for prompt-tuning SLMs, evaluating entity extraction, and providing ground-truth
consultation scripts to Person A for audio synthesis.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent


# Core gold consultation dialogues
BENCHMARK_DIALOGUES: list[dict[str, Any]] = [
    {
        "session_id": "OPD_SIM_001",
        "title": "Acute Viral Fever with Bodyache",
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 2.5, "speaker": "doctor", "text": "Aaiye, namaste. Boliye kya takleef hai?", "lang": "hi-en"},
            {"start": 2.6, "end": 7.2, "speaker": "patient", "text": "Doctor sahab teen din se bohot tez bukhar hai aur poore badan me dard ho raha hai.", "lang": "hi-en"},
            {"start": 7.3, "end": 10.5, "speaker": "doctor", "text": "Khansi ya ulti to nahi hui aapko?", "lang": "hi-en"},
            {"start": 10.6, "end": 13.8, "speaker": "patient", "text": "Nahi doctor sahab, khansi bilkul nahi hai, ulti bhi nahi hui.", "lang": "hi-en"},
            {"start": 14.0, "end": 18.0, "speaker": "doctor", "text": "Theek hai, thermometer lagata hoon. Temperature 101.4 Fahrenheit hai. Pulse 88 hai.", "lang": "hi-en"},
            {"start": 18.2, "end": 21.0, "speaker": "doctor", "text": "Yeh mausam ka seasonal viral fever lag raha hai.", "lang": "hi-en"},
            {"start": 21.2, "end": 26.5, "speaker": "doctor", "text": "Aap Dolo 650 mg din me 3 baar khana khane ke baad lijiye teen din tak.", "lang": "hi-en"},
            {"start": 26.8, "end": 31.0, "speaker": "doctor", "text": "Pani khub piyo. Agar bukhar 3 din me theek na ho to CBC test karwa lena.", "lang": "hi-en"}
        ],
        "gold_extraction": {
            "symptoms": [
                {"text": "fever", "duration": "3 days", "severity": "severe", "negated": False, "source_segments": [1]},
                {"text": "body ache", "duration": "3 days", "severity": "severe", "negated": False, "source_segments": [1]},
                {"text": "cough", "duration": None, "severity": None, "negated": True, "source_segments": [3]},
                {"text": "vomiting", "duration": None, "severity": None, "negated": True, "source_segments": [3]}
            ],
            "diagnosis": [
                {"text": "viral fever", "confidence": "medium", "status": "provisional", "source_segments": [5]}
            ],
            "medications": [
                {"name": "Dolo", "dose": "650 mg", "frequency": "TDS", "duration": "3 days", "status": "new", "instructions": "after food", "source_segments": [6]}
            ],
            "vitals": [
                {"type": "temperature", "value": 101.4, "unit": "degF", "source_segments": [4]},
                {"type": "pulse", "value": 88, "unit": "/min", "source_segments": [4]}
            ],
            "investigations": [
                {"name": "Complete Blood Count (CBC)", "status": "recommended", "result": None, "source_segments": [7]}
            ]
        }
    },
    {
        "session_id": "OPD_SIM_002",
        "title": "Essential Hypertension Follow-up",
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "doctor", "text": "Good morning Mr. Verma. Aaj BP check karwane aaye hain?", "lang": "hi-en"},
            {"start": 3.2, "end": 8.0, "speaker": "patient", "text": "Haan doctor, subah se thoda sar me dard ho raha tha aur chhati me koi dard nahi hai.", "lang": "hi-en"},
            {"start": 8.2, "end": 12.0, "speaker": "doctor", "text": "Pehle se koi dawa chal rahi hai?", "lang": "hi-en"},
            {"start": 12.2, "end": 16.5, "speaker": "patient", "text": "Haan, pichle ek saal se Amlodipine 5 mg roz subah ek goli leta hoon.", "lang": "hi-en"},
            {"start": 16.8, "end": 21.0, "speaker": "doctor", "text": "Aapka BP dekhte hain... 150/95 mm Hg hai. BP thoda high hai.", "lang": "hi-en"},
            {"start": 21.2, "end": 24.0, "speaker": "doctor", "text": "Aapko essential hypertension ka issue control karna hoga.", "lang": "hi-en"},
            {"start": 24.2, "end": 29.5, "speaker": "doctor", "text": "Main Amlodipine ke sath Telmisartan 40 mg add kar raha hoon, roz subah ek tablet.", "lang": "hi-en"},
            {"start": 29.8, "end": 33.0, "speaker": "doctor", "text": "Khane me namak kam kijiye aur ek hafte baad dubara BP check karaiye.", "lang": "hi-en"}
        ],
        "gold_extraction": {
            "symptoms": [
                {"text": "headache", "duration": None, "severity": "mild", "negated": False, "source_segments": [1]},
                {"text": "chest pain", "duration": None, "severity": None, "negated": True, "source_segments": [1]}
            ],
            "diagnosis": [
                {"text": "essential hypertension", "confidence": "high", "status": "active", "source_segments": [5]}
            ],
            "medications": [
                {"name": "Amlodipine", "dose": "5 mg", "frequency": "OD", "duration": None, "status": "existing", "instructions": "morning", "source_segments": [3]},
                {"name": "Telmisartan", "dose": "40 mg", "frequency": "OD", "duration": None, "status": "new", "instructions": "morning", "source_segments": [6]}
            ],
            "vitals": [
                {"type": "blood_pressure_sys", "value": 150, "unit": "mm[Hg]", "source_segments": [4]},
                {"type": "blood_pressure_dia", "value": 95, "unit": "mm[Hg]", "source_segments": [4]}
            ],
            "investigations": []
        }
    },
    {
        "session_id": "OPD_SIM_003",
        "title": "Type 2 Diabetes Mellitus Review",
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 2.5, "speaker": "doctor", "text": "Namaste Sunita ji, reports le aayi hain aap?", "lang": "hi-en"},
            {"start": 2.8, "end": 7.5, "speaker": "patient", "text": "Haan doctor sahab, HbA1c test karwaya tha, 7.8 percent aaya hai. Aur kamzori lagti hai.", "lang": "hi-en"},
            {"start": 7.8, "end": 10.5, "speaker": "doctor", "text": "Weight kitna hai abhi? Machine pe khade ho jaiye... 68 kg hai.", "lang": "hi-en"},
            {"start": 10.8, "end": 14.2, "speaker": "doctor", "text": "Pehle se Metformin chal rahi hai na aapki?", "lang": "hi-en"},
            {"start": 14.5, "end": 18.0, "speaker": "patient", "text": "Haan, Metformin 500 mg subah shaam le rahi hoon khana khane ke baad.", "lang": "hi-en"},
            {"start": 18.2, "end": 22.0, "speaker": "doctor", "text": "Aapka type 2 diabetes abhi thoda un-controlled hai.", "lang": "hi-en"},
            {"start": 22.2, "end": 28.0, "speaker": "doctor", "text": "Metformin ko 1000 mg twice a day karte hain. Meetha bilkul band kar dijiye.", "lang": "hi-en"},
            {"start": 28.2, "end": 32.0, "speaker": "doctor", "text": "Agale mahine Fasting Blood Sugar test karwake aaiye.", "lang": "hi-en"}
        ],
        "gold_extraction": {
            "symptoms": [
                {"text": "weakness", "duration": None, "severity": None, "negated": False, "source_segments": [1]}
            ],
            "diagnosis": [
                {"text": "type 2 diabetes mellitus", "confidence": "high", "status": "active", "source_segments": [5]}
            ],
            "medications": [
                {"name": "Metformin", "dose": "500 mg", "frequency": "BD", "duration": None, "status": "existing", "instructions": "after food", "source_segments": [4]},
                {"name": "Metformin", "dose": "1000 mg", "frequency": "BD", "duration": None, "status": "modified", "instructions": "after food", "source_segments": [6]}
            ],
            "vitals": [
                {"type": "weight", "value": 68.0, "unit": "kg", "source_segments": [2]}
            ],
            "investigations": [
                {"name": "HbA1c", "status": "reviewed", "result": "7.8%", "source_segments": [1]},
                {"name": "Fasting Blood Sugar", "status": "ordered", "result": None, "source_segments": [7]}
            ]
        }
    },
    {
        "session_id": "OPD_SIM_004",
        "title": "Upper Respiratory Tract Infection with Sore Throat",
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "doctor", "text": "Haan ji Rahul, gale me kya takleef hai?", "lang": "hi-en"},
            {"start": 3.2, "end": 8.0, "speaker": "patient", "text": "Doctor sahab do din se gale me kharash hai, khansi bhi aa rahi hai aur thandi lag rahi hai.", "lang": "hi-en"},
            {"start": 8.2, "end": 11.5, "speaker": "doctor", "text": "Saans phoolne ki koi shikayat hai kya?", "lang": "hi-en"},
            {"start": 11.6, "end": 14.0, "speaker": "patient", "text": "Nahi, saans lene me koi dikkat nahi hai.", "lang": "hi-en"},
            {"start": 14.2, "end": 18.0, "speaker": "doctor", "text": "Moo kholiye... throat kafi congested hai. Temperature 99.8 F hai.", "lang": "hi-en"},
            {"start": 18.2, "end": 21.0, "speaker": "doctor", "text": "Yeh upper respiratory tract infection hai.", "lang": "hi-en"},
            {"start": 21.2, "end": 26.5, "speaker": "doctor", "text": "Main Azithromycin 500 mg likh raha hoon, rozana ek goli 3 din tak lijiye.", "lang": "hi-en"},
            {"start": 26.8, "end": 31.0, "speaker": "doctor", "text": "Sath me Cetirizine 10 mg raat ko sone se pehle lena khansi aur cold ke liye.", "lang": "hi-en"}
        ],
        "gold_extraction": {
            "symptoms": [
                {"text": "sore throat", "duration": "2 days", "severity": None, "negated": False, "source_segments": [1]},
                {"text": "cough", "duration": "2 days", "severity": None, "negated": False, "source_segments": [1]},
                {"text": "cold", "duration": "2 days", "severity": None, "negated": False, "source_segments": [1]},
                {"text": "breathlessness", "duration": None, "severity": None, "negated": True, "source_segments": [3]}
            ],
            "diagnosis": [
                {"text": "upper respiratory tract infection", "confidence": "high", "status": "active", "source_segments": [5]}
            ],
            "medications": [
                {"name": "Azithromycin", "dose": "500 mg", "frequency": "OD", "duration": "3 days", "status": "new", "instructions": None, "source_segments": [6]},
                {"name": "Cetirizine", "dose": "10 mg", "frequency": "OD", "duration": None, "status": "new", "instructions": "at bedtime", "source_segments": [7]}
            ],
            "vitals": [
                {"type": "temperature", "value": 99.8, "unit": "degF", "source_segments": [4]}
            ],
            "investigations": []
        }
    },
    {
        "session_id": "OPD_SIM_005",
        "title": "Acute Gastroenteritis and Dehydration",
        "language_mix": "hi-en",
        "segments": [
            {"start": 0.0, "end": 2.5, "speaker": "doctor", "text": "Bataiye, pet me kya problem ho rahi hai?", "lang": "hi-en"},
            {"start": 2.8, "end": 8.0, "speaker": "patient", "text": "Doctor kal raat se 5-6 baar ulti hui hai aur loose motion lag gaye hain.", "lang": "hi-en"},
            {"start": 8.2, "end": 11.5, "speaker": "doctor", "text": "Chhati me dard ya bukhar to nahi hai?", "lang": "hi-en"},
            {"start": 11.6, "end": 14.5, "speaker": "patient", "text": "Bukhar halka lag raha hai, chhati me koi dard nahi hai.", "lang": "hi-en"},
            {"start": 14.8, "end": 19.5, "speaker": "doctor", "text": "BP check karte hain... 100/70 mm Hg hai. Pulse 92 beats per minute hai.", "lang": "hi-en"},
            {"start": 19.8, "end": 22.5, "speaker": "doctor", "text": "Dehydration ho gaya hai, acute gastroenteritis lag raha hai.", "lang": "hi-en"},
            {"start": 22.8, "end": 28.0, "speaker": "doctor", "text": "Har loose motion ke baad ORS ka ghol peena hai, kam se kam 1 liter daily.", "lang": "hi-en"},
            {"start": 28.2, "end": 33.0, "speaker": "doctor", "text": "Ulti rokne ke liye tablet Ondansetron 4 mg SOS lena.", "lang": "hi-en"}
        ],
        "gold_extraction": {
            "symptoms": [
                {"text": "vomiting", "duration": "1 days", "severity": "severe", "negated": False, "source_segments": [1]},
                {"text": "loose motion", "duration": "1 days", "severity": "severe", "negated": False, "source_segments": [1]},
                {"text": "fever", "duration": None, "severity": "mild", "negated": False, "source_segments": [3]},
                {"text": "chest pain", "duration": None, "severity": None, "negated": True, "source_segments": [3]}
            ],
            "diagnosis": [
                {"text": "gastroenteritis", "confidence": "high", "status": "active", "source_segments": [5]}
            ],
            "medications": [
                {"name": "ORS", "dose": "1 liter", "frequency": "SOS", "duration": None, "status": "new", "instructions": "after every motion", "source_segments": [6]},
                {"name": "Ondansetron", "dose": "4 mg", "frequency": "SOS", "duration": None, "status": "new", "instructions": "when needed", "source_segments": [7]}
            ],
            "vitals": [
                {"type": "blood_pressure_sys", "value": 100, "unit": "mm[Hg]", "source_segments": [4]},
                {"type": "blood_pressure_dia", "value": 70, "unit": "mm[Hg]", "source_segments": [4]},
                {"type": "pulse", "value": 92, "unit": "/min", "source_segments": [4]}
            ],
            "investigations": []
        }
    }
]


def generate_extended_dialogue_corpus(target_count: int = 200, seed: int = 42) -> list[dict[str, Any]]:
    """Generates an extended dataset of realistic simulated OPD dialogues (150-250)
    by systematically synthesizing clinical variations across templates."""
    rng = random.Random(seed)
    dialogues: list[dict[str, Any]] = list(BENCHMARK_DIALOGUES)

    templates = [
        {
            "condition": "viral fever",
            "symptoms": [("fever", "bukhar"), ("body ache", "badan dard"), ("headache", "sar dard")],
            "neg_symptoms": [("cough", "khansi"), ("vomiting", "ulti")],
            "meds": [("Paracetamol", "650 mg", "TDS", "3 days"), ("Pantoprazole", "40 mg", "OD", "5 days")],
            "vitals": [("temperature", [100.4, 101.2, 102.0], "degF"), ("pulse", [82, 86, 90, 94], "/min")],
            "inv": "Complete Blood Count (CBC)",
        },
        {
            "condition": "upper respiratory tract infection",
            "symptoms": [("cough", "khansi"), ("sore throat", "gale me kharash"), ("cold", "sardi")],
            "neg_symptoms": [("breathlessness", "saans phoolna"), ("chest pain", "chhati me dard")],
            "meds": [("Amoxicillin", "500 mg", "TDS", "5 days"), ("Cetirizine", "10 mg", "OD", "5 days")],
            "vitals": [("temperature", [99.4, 100.2], "degF"), ("pulse", [78, 82], "/min")],
            "inv": "Chest X-ray",
        },
        {
            "condition": "essential hypertension",
            "symptoms": [("headache", "sar me dard"), ("weakness", "thakan")],
            "neg_symptoms": [("chest pain", "chhati me dard")],
            "meds": [("Amlodipine", "5 mg", "OD", "continuous"), ("Telmisartan", "40 mg", "OD", "continuous")],
            "vitals": [("blood_pressure_sys", [140, 150, 160], "mm[Hg]"), ("blood_pressure_dia", [90, 95, 100], "mm[Hg]")],
            "inv": "ECG",
        },
        {
            "condition": "type 2 diabetes mellitus",
            "symptoms": [("weakness", "kamzori"), ("fatigue", "thakan")],
            "neg_symptoms": [("fever", "bukhar")],
            "meds": [("Metformin", "500 mg", "BD", "continuous"), ("Atorvastatin", "10 mg", "OD", "continuous")],
            "vitals": [("weight", [65.0, 72.5, 80.0], "kg")],
            "inv": "HbA1c",
        },
        {
            "condition": "acidity",
            "symptoms": [("acidity", "pet me jalan"), ("nausea", "mitli")],
            "neg_symptoms": [("vomiting", "ulti"), ("chest pain", "chhati me dard")],
            "meds": [("Pantoprazole", "40 mg", "OD", "14 days")],
            "vitals": [("pulse", [74, 76, 80], "/min")],
            "inv": None,
        },
    ]

    while len(dialogues) < target_count:
        idx = len(dialogues) + 1
        t = rng.choice(templates)
        dur_val = rng.choice(["2", "3", "4", "5"])
        dur_str = f"{dur_val} days"

        sym1, sym1_hi = rng.choice(t["symptoms"])
        neg1, neg1_hi = rng.choice(t["neg_symptoms"])
        med1, med1_dose, med1_freq, med1_dur = rng.choice(t["meds"])

        segments = [
            {"start": 0.0, "end": 2.5, "speaker": "doctor", "text": "Namaste, boliye kya pareshani hai?", "lang": "hi-en"},
            {"start": 2.6, "end": 7.0, "speaker": "patient", "text": f"Doctor sahab {dur_val} din se {sym1_hi} ho raha hai.", "lang": "hi-en"},
            {"start": 7.2, "end": 10.0, "speaker": "doctor", "text": f"Aapko {neg1_hi} to nahi hai?", "lang": "hi-en"},
            {"start": 10.2, "end": 13.5, "speaker": "patient", "text": f"Nahi doctor, {neg1_hi} bilkul nahi hai.", "lang": "hi-en"},
        ]

        gold_vitals = []
        curr_t = 14.0
        vitals_text = "Theek hai, examination karte hain... "
        for vtype, vvals, vunit in t["vitals"]:
            vval = rng.choice(vvals)
            vitals_text += f"{vtype} {vval} {vunit}. "
            gold_vitals.append({"type": vtype, "value": vval, "unit": vunit, "source_segments": [4]})

        segments.append({"start": curr_t, "end": curr_t + 4.5, "speaker": "doctor", "text": vitals_text.strip(), "lang": "hi-en"})
        curr_t += 5.0

        diag_seg_idx = len(segments)
        segments.append({
            "start": curr_t, "end": curr_t + 3.0,
            "speaker": "doctor",
            "text": f"Yeh {t['condition']} ka mamla lag raha hai.",
            "lang": "hi-en"
        })
        curr_t += 3.5

        med_seg_idx = len(segments)
        segments.append({
            "start": curr_t, "end": curr_t + 5.0,
            "speaker": "doctor",
            "text": f"Main aapko {med1} {med1_dose} likh raha hoon, din me {med1_freq} lena {med1_dur} tak.",
            "lang": "hi-en"
        })
        curr_t += 5.5

        gold_inv = []
        if t["inv"]:
            inv_seg_idx = len(segments)
            segments.append({
                "start": curr_t, "end": curr_t + 4.0,
                "speaker": "doctor",
                "text": f"Kripya ek baar {t['inv']} test karwa lijiye.",
                "lang": "hi-en"
            })
            gold_inv.append({"name": t["inv"], "status": "recommended", "result": None, "source_segments": [inv_seg_idx]})

        sim_item = {
            "session_id": f"OPD_SIM_{idx:03d}",
            "title": f"Simulated {t['condition'].title()} Case {idx}",
            "language_mix": "hi-en",
            "segments": segments,
            "gold_extraction": {
                "symptoms": [
                    {"text": sym1, "duration": dur_str, "severity": None, "negated": False, "source_segments": [1]},
                    {"text": neg1, "duration": None, "severity": None, "negated": True, "source_segments": [3]},
                ],
                "diagnosis": [
                    {"text": t["condition"], "confidence": "medium", "status": "provisional", "source_segments": [diag_seg_idx]}
                ],
                "medications": [
                    {"name": med1, "dose": med1_dose, "frequency": med1_freq, "duration": med1_dur, "status": "new", "instructions": None, "source_segments": [med_seg_idx]}
                ],
                "vitals": gold_vitals,
                "investigations": gold_inv,
            }
        }
        dialogues.append(sim_item)

    return dialogues


def save_simulated_corpus(path: Path | None = None, count: int = 200) -> Path:
    """Saves the simulated dialogues to disk."""
    out_path = path or (DATA_DIR / "simulated_dialogues.json")
    corpus = generate_extended_dialogue_corpus(target_count=count)
    out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
