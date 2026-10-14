#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_dataset_b.py

Synthetic Hinglish doctor-patient OPD dialogues with gold labels, for the
Edge-AI Ambient Clinical Scribe project (Dataset B replacement).

Each dialogue record carries:
  transcript  -> Interface 1 (A -> B): session_id, language_mix, segments[]
  extraction  -> Interface 2 (inside B): session_id, entities{...}
  metadata    -> scenario, split, style, hard_cases, distractors, voice hints

Every extracted entity carries source_segments (indices into transcript.segments),
and the script verifies that key values (doses, durations, vitals, results)
literally appear in the segments they cite (the same rule the grounding check uses).

Usage:  python generate_dataset_b.py --per-scenario 18 --seed 42 --out-dir out
Clinical content is template-generated and NOT clinician-reviewed.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Numbers -> words (for text_tts, so a TTS engine reads digits correctly)
# ----------------------------------------------------------------------------
EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
           "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
HI_NUM = ["zero", "ek", "do", "teen", "chaar", "paanch", "chhe", "saat", "aath", "nau",
          "das", "gyarah", "barah", "terah", "chaudah", "pandrah", "solah", "satrah",
          "athaarah", "unnees", "bees"]
HI_CTX = {"din", "hafta", "hafte", "mahina", "mahine", "baar", "saal"}


def en_words(n):
    if n < 20:
        return EN_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return EN_TENS[t] + (" " + EN_ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return EN_ONES[h] + " hundred" + (" " + en_words(r) if r else "")
    th, r = divmod(n, 1000)
    return en_words(th) + " thousand" + (" " + en_words(r) if r else "")


def num_words(s, nxt):
    if "." in s:
        a, b = s.split(".")
        return en_words(int(a)) + " point " + " ".join(EN_ONES[int(ch)] for ch in b)
    n = int(s)
    if nxt in HI_CTX:
        if n <= 20:
            return HI_NUM[n]
        if n == 30:
            return "tees"
    return en_words(n)


ABBR = [("mg/dL", "milligram per decilitre"), ("mIU/L", "milli international units per litre"),
        ("mcg", "microgram"), ("mg", "milligram"), ("ml", "millilitre"),
        ("HbA1c", "H b A one c"), ("SpO2", "S P O two"), ("NS1", "N S one"),
        ("TDS", "T D S"), ("BD", "B D"), ("OD", "O D"), ("SOS", "S O S"), ("ORS", "O R S"),
        ("ECG", "E C G"), ("CBC", "C B C"), ("TSH", "T S H"), ("KFT", "K F T"),
        ("UTI", "U T I"), ("GERD", "G E R D"), ("PFT", "P F T"), ("BP", "B P"),
        ("URTI", "U R T I")]


def spoken(text):
    s = re.sub(r"\b([A-Za-z]{1,4})-\s+(?=\1)", "", text, flags=re.I)  # drop stutter fragments
    s = s.replace("—", "...").replace("%", " percent")
    for k, v in ABBR:
        s = re.sub(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", v, s)

    def rep(m):
        nx = re.match(r"\s*([A-Za-z]+)", m.string[m.end():])
        return num_words(m.group(), nx.group(1).lower() if nx else "")
    s = re.sub(r"\d+(?:\.\d+)?", rep, s)
    return re.sub(r"\s+", " ", s).strip()


def fnum(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def fmt_dur(n, unit, lang="hi"):
    if lang == "en":
        return f"{n} {unit}" + ("" if n == 1 else "s")
    if unit == "day":
        return f"{n} din"
    if unit == "week":
        return f"{n} hafta" if n == 1 else f"{n} hafte"
    return f"{n} mahina" if n == 1 else f"{n} mahine"


def dur_norm(d):
    n, u = d
    return f"{n} {u}" + ("" if n == 1 else "s")


def dur_phrase(rng, d):
    lang = "hi" if rng.random() < 0.75 else "en"
    x = fmt_dur(d[0], d[1], lang)
    if lang == "en":
        return rng.choice([f"for {x}", f"next {x}", f"{x} tak"])
    return rng.choice([f"{x} tak", f"{x} ke liye"])


# ----------------------------------------------------------------------------
# Language pools
# ----------------------------------------------------------------------------
GREET_DOC = ["Namaste, baithiye.", "Haan ji, aaiye baithiye.", "Namaste. Aaiye, baithiye.",
             "Aaiye aaiye, baithiye.", "Haan ji, namaste."]
PAT_GREET = ["Namaste doctor sahab.", "Namaste sir.", "Good morning doctor.", "Namaste doctor.",
             "Namaste, doctor sahab."]
DOC_AGE = ["Pehle bataiye, umar kitni hai aapki?", "Age kitni hai aapki?", "Aapki age kya hogi?",
           "Umar kitni hai?"]
PAT_AGE = ["{age} saal.", "{age} years.", "Ji, {age} saal {ka}.", "{age} saal, doctor sahab."]
OPEN_DOC = ["Bataiye, kya takleef hai?", "Haan ji, kya problem hai?", "Boliye, aaj kya pareshani hai?",
            "Kya dikkat hai, bataiye?", "Haan boliye, kaise aana hua?"]
OPEN_FU = ["Haan ji, kaisa chal raha hai ab? Dawai le rahe hain na?", "Aaiye, follow-up ke liye aaye hain? Kaisa hai ab?",
           "Bataiye, pichli baar ke baad kaisa raha?", "Haan ji, check-up ke liye aaye hain? Kaisa chal raha hai?"]
PAT_ROUTINE = ["Doctor sahab, mujhe {dxpat} hai, follow-up ke liye aaya hoon.",
               "Routine check-up ke liye aaya hoon, {dxpat} ke liye.",
               "Meri {dxpat} ki dawai chal rahi hai, check karwane aaya hoon.",
               "{dxpat} ka follow-up hai doctor sahab."]
DOC_BACK = ["Hmm.", "Achha.", "Okay.", "Haan.", "Achha achha.", "Theek hai.", "Hmm hmm.", "Okay okay."]
PAT_BACK = ["Ji.", "Haan ji.", "Theek hai.", "Okay.", "Ji doctor.", "Hmm.", "Achha."]
DOC_LEAD = ["hmm", "achha", "theek hai", "okay", "haan", "dekhiye", "toh", "right", "chaliye", "matlab"]
DOC_MID = ["hmm", "matlab", "toh", "dekhiye", "achha"]
# patient fillers are neutral hesitations only (no haan/ji/nahi), so they never flip polarity
PAT_LEAD = ["um", "uh", "matlab", "woh", "actually", "arre", "hmm", "mmm", "vaise", "you know", "wo kya hai ki",
            "kya kehte hain", "dekhiye", "toh"]
PAT_MID = ["um", "uh", "matlab", "woh", "actually", "you know", "hmm", "mmm", "vaise", "wo kya hai ki", "kya kehte hain", "arre"]
CHATTER = ["Kal ek shaadi mein gaya tha, wahan bahar ka khana kha liya tha.",
           "Office mein aajkal kaam bahut hai, time hi nahi milta.",
           "Mere padosi ko bhi pichle mahine aisa hi hua tha.",
           "Mausam badal raha hai na, isliye sab bimaar pad rahe hain.",
           "Aaj traffic bahut tha, isliye late ho gaya aane mein.",
           "Kal baarish mein bheeg gaya tha rickshaw mein."]
ANXIOUS_Q = ["Kuch serious toh nahi hai na doctor?", "Doctor sahab, koi badi problem toh nahi hai na?",
             "Mujhe bahut tension ho rahi hai, kuch gambhir toh nahi?"]
ANXIOUS_A = ["Nahi nahi, ghabraiye mat, ye common hai.", "Aap tension mat lijiye, sab theek ho jayega.",
             "Nahi, aisi koi badi baat nahi hai."]
ACKS = ["Ji.", "Theek hai.", "Okay doctor.", "Haan ji, samajh gaya.", "Achha."]
CLOSE_PAT = ["Thank you doctor.", "Dhanyavaad doctor.", "Thanks doctor sahab.", "Theek hai doctor, thank you."]
CLOSE_DOC = ["Theek hai, apna khayal rakhiye.", "Get well soon.", "Theek hai, dhyan rakhiye.", "Chaliye, jaldi theek ho jayenge."]

GEN_Q = ["{t} bhi hai?", "{t} toh nahi hai?", "Kya {t} bhi hai?", "{t} hai kya?"]
GEN_POS = ["Haan, {t} bhi hai.", "{t} bhi hai thoda.", "Haan ji, {t} hai."]
GEN_NEG = ["Nahi, {t} nahi hai.", "{t} toh nahi hai.", "Nahi nahi, {t} nahi hai."]
GEN_DPOS = ["{d} se {t} hai.", "Mujhe {d} se {t} hai.", "{t} hai, {d} se."]
CC_GEN = ["{t} hai, {d} se.", "{d} se {t} hai doctor sahab.", "Mujhe {d} se {t} hai.",
          "{t} hai, {d} ho gaye.", "Doctor sahab, {d} se {t} hai, theek nahi ho raha."]
GROUP_Q = ["{lst} — inmein se kuch hai?", "{lst} toh nahi hai na?", "Aur koi problem? {lst}?"]
GROUP_NEG = ["Nahi, inmein se kuch nahi hai.", "Nahi nahi, kuch nahi hai.", "Nahi doctor, woh sab nahi hai.", "Ji nahi."]


def S(name, t, **kw):
    d = dict(name=name, t=t)
    d.update(kw)
    return d


SYM = {
    "fever": S("fever", ["bukhar", "fever"]),
    "body ache": S("body ache", ["body ache", "badan dard", "body pain"]),
    "headache": S("headache", ["sar dard", "headache", "sir mein dard"]),
    "weakness": S("weakness", ["kamzori", "weakness"]),
    "chills": S("chills", ["thand lagna"], q=["Thand lagti hai?", "Kaanpi aati hai kya?"],
                pos=["Haan, thand lag rahi hai.", "Haan, chills aa rahe hain."],
                neg=["Nahi, thand nahi lag rahi.", "Chills nahi aa rahe."],
                dpos=["{d} se thand lag rahi hai.", "Mujhe {d} se chills aa rahe hain."]),
    "loss of appetite": S("loss of appetite", ["bhookh kam lagna"], q=["Bhookh theek se lagti hai?", "Khana khane ka mann hai?"],
                          pos=["Haan, bhookh kam lag rahi hai.", "Bhookh bilkul nahi lag rahi."],
                          neg=["Bhookh theek hai.", "Haan khana kha raha hoon theek se."],
                          dpos=["{d} se bhookh kam lag rahi hai."]),
    "dry cough": S("dry cough", ["sookhi khansi", "dry cough"]),
    "productive cough": S("productive cough", ["balgam wali khansi", "khansi mein balgam"]),
    "cough": S("cough", ["khansi", "cough"]),
    "sore throat": S("sore throat", ["gale mein dard", "sore throat", "gala kharab"]),
    "runny nose": S("runny nose", ["naak behna"], q=["Naak beh rahi hai?", "Naak behna toh nahi hai?"],
                    pos=["Haan, naak beh rahi hai.", "Naak bahut beh rahi hai."], neg=["Naak nahi beh rahi."],
                    dpos=["{d} se naak beh rahi hai."]),
    "sneezing": S("sneezing", ["chheenk"], q=["Chheenkein aati hain?", "Bahut chheenk aati hai kya?"],
                  pos=["Haan, chheenkein aa rahi hain.", "Subah bahut chheenkein aati hain."], neg=["Chheenkein nahi aa rahi."],
                  dpos=["{d} se chheenkein aa rahi hain.", "Mujhe {d} se chheenkein aa rahi hain."]),
    "itchy eyes": S("itchy eyes", ["aankhon mein khujli", "eyes itching"]),
    "breathlessness": S("breathlessness", ["saans ki takleef", "breathlessness"]),
    "wheezing": S("wheezing", ["saans mein seeti"], q=["Saans lete waqt seeti ki awaaz aati hai?", "Wheezing jaisa lagta hai?"],
                  pos=["Haan, saans mein seeti ki awaaz aati hai.", "Raat ko seeti bajti hai saans mein."],
                  neg=["Seeti ki awaaz nahi aati."], dpos=["{d} se saans mein seeti ki awaaz aa rahi hai."]),
    "chest tightness": S("chest tightness", ["seene mein jakdan", "chest tightness"]),
    "chest pain": S("chest pain", ["chest pain", "seene mein dard"]),
    "nausea": S("nausea", ["ji michlana"], q=["Ji michlata hai?", "Nausea hai kya?"],
                pos=["Haan, ji michlata hai.", "Nausea jaisa lagta hai."], neg=["Ji nahi michlata.", "Nausea nahi hai."],
                dpos=["{d} se ji michlata hai."]),
    "vomiting": S("vomiting", ["ulti", "vomiting"], q=["Ulti hui kya?", "Vomiting toh nahi hui?"],
                  pos=["Haan, ulti ho rahi hai.", "Ek do baar ulti hui."], neg=["Nahi, ulti nahi hui.", "Ulti nahi hai."],
                  dpos=["{d} se ulti ho rahi hai."]),
    "loose motion": S("loose motion", ["loose motion", "dast"]),
    "abdominal pain": S("abdominal pain", ["pet dard", "pet mein dard", "stomach pain"]),
    "acidity": S("acidity", ["acidity", "seene mein jalan", "pet mein jalan"]),
    "sour belching": S("sour belching", ["khatti dakaar"], q=["Khatti dakaar aati hai?", "Khatte dakaar aate hain?"],
                       pos=["Haan, khatti dakaar aati hai."], neg=["Khatti dakaar nahi aati."],
                       dpos=["{d} se khatti dakaar aa rahi hai."]),
    "bloating": S("bloating", ["pet phoolna", "gas", "pet mein gas"]),
    "black stools": S("black stools", ["kaala potty", "black stool"]),
    "blood in stool": S("blood in stool", ["potty mein khoon", "blood in stool"]),
    "excessive thirst": S("excessive thirst", ["pyaas zyada lagna"], q=["Pyaas zyada lagti hai?", "Bahut pyaas lagti hai kya?"],
                          pos=["Haan, pyaas bahut lagti hai."], neg=["Pyaas zyada nahi lagti."],
                          dpos=["{d} se pyaas bahut lagti hai."]),
    "frequent urination": S("frequent urination", ["baar baar peshab"], q=["Peshab baar baar aata hai?", "Raat ko peshab ke liye uthna padta hai?"],
                            pos=["Haan, baar baar peshab aata hai."], neg=["Peshab ki dikkat nahi hai."],
                            dpos=["{d} se baar baar peshab aa raha hai."]),
    "burning micturition": S("burning micturition", ["peshab mein jalan", "burning urination"]),
    "tingling in feet": S("tingling in feet", ["pairon mein jhunjhuni", "tingling"]),
    "blurred vision": S("blurred vision", ["dhundhla dikhna", "blurred vision"]),
    "dizziness": S("dizziness", ["chakkar"], q=["Chakkar aate hain?", "Dizziness toh nahi hai?"],
                   pos=["Haan, chakkar aate hain.", "Kabhi kabhi chakkar aata hai."], neg=["Chakkar nahi aate."],
                   dpos=["{d} se chakkar aa rahe hain."]),
    "palpitations": S("palpitations", ["dhadkan tez hona", "palpitations"]),
    "hair fall": S("hair fall", ["baal jhadna", "hair fall"]),
    "weight gain": S("weight gain", ["weight badhna", "weight gain"]),
    "constipation": S("constipation", ["kabz", "constipation"]),
    "cold intolerance": S("cold intolerance", ["thand zyada lagna"], q=["Thand zyada lagti hai?", "Thand bardasht nahi hoti kya?"],
                          pos=["Haan, thand bahut lagti hai."], neg=["Thand zyada nahi lagti."],
                          dpos=["{d} se thand bahut lag rahi hai."]),
    "back pain": S("back pain", ["kamar dard", "back pain"]),
    "knee pain": S("knee pain", ["ghutne mein dard", "knee pain"]),
    "photophobia": S("photophobia", ["roshni se dikkat", "light se problem"]),
}

FREQ = {
    "OD": ["din mein ek baar", "roz ek baar", "once a day", "OD"],
    "BD": ["subah shaam", "din mein 2 baar", "twice a day", "BD", "subah aur raat ko"],
    "TDS": ["din mein 3 baar", "thrice a day", "TDS", "subah dopahar shaam", "3 baar"],
    "HS": ["raat ko sone se pehle", "raat ko ek baar", "at bedtime"],
    "SOS": ["zaroorat padne par", "jab zaroorat ho tab", "SOS", "agar takleef ho toh"],
    "AFTER_STOOL": ["har loose motion ke baad", "jitni baar loose motion ho, utni baar"],
}
FREQ_NORM = {"OD": "OD", "BD": "BD", "TDS": "TDS", "HS": "HS", "SOS": "SOS", "AFTER_STOOL": "after each loose stool", None: None}
TIMING = {"after_food": ["khana khane ke baad", "khane ke baad", "after food"],
          "before_food": ["khane se pehle", "before food"],
          "empty_stomach": ["khali pet", "empty stomach", "subah khali pet"]}
TIMING_NORM = {"after_food": "after food", "before_food": "before food", "empty_stomach": "empty stomach", None: None}

DX_HIGH = ["Aapko {dx} hai.", "Yeh {dx} hai.", "Symptoms aur check-up se {dx} hi lag raha hai, confirm hai.", "Dekhiye, {dx} hai aapko."]
DX_MED = ["Mujhe lagta hai {dx} hai.", "Most likely {dx} hai.", "{dx} ho sakta hai, abhi aise hi lag raha hai.",
          "Symptoms se toh {dx} jaisa lag raha hai."]
DX_STATE = ["Aapka {dx} {state}.", "{dx} ki baat karein toh, {state}.", "Dekhiye, {dx} {state}."]

DOC_VITAL_INTRO = ["Chaliye, ek baar check kar lete hain.", "Zara vitals check kar leta hoon.", "Ek minute, dekh leta hoon."]
FAMILY_REL = ["mere papa", "meri mummy", "mere bhai", "meri didi", "mere dada ji"]
FAMILY_COND = {"sugar": "diabetes", "BP": "hypertension", "thyroid": "hypothyroidism",
               "asthma": "asthma", "heart ki problem": "heart disease"}


# ----------------------------------------------------------------------------
# Case builders
# ----------------------------------------------------------------------------
def med(name, generic, dose, freq, dur=None, timing=None, status="new", form="tablet", **kw):
    m = dict(name=name, generic=generic, dose=dose, freq=freq, duration=dur, timing=timing,
             status=status, form=form)
    m.update(kw)
    return m


def paracetamol(rng, freq, dur, status="new", timing=None):
    name, dose = rng.choice([("Dolo", "650 mg"), ("Crocin", "500 mg"), ("Paracetamol", "500 mg"), ("Paracetamol", "650 mg")])
    alt = "500 mg" if dose == "650 mg" else "650 mg"
    return med(name, "Paracetamol", dose, freq, dur, timing, status, alt_dose=alt)


def self_para(rng):
    name, dose = rng.choice([("Dolo", "650 mg"), ("Crocin", "500 mg"), ("Paracetamol", "500 mg")])
    return med(name, "Paracetamol", dose, None, None, None, "self_taken")


def vit(type_, value, unit, src="measured", **kw):
    v = dict(type=type_, value=value, unit=unit, source=src)
    v.update(kw)
    return v


def v_temp(val, unit, src="measured"):
    return vit("temperature", val, unit, src)


def v_bp(s, d):
    return vit("blood_pressure", f"{s}/{d}", "mm[Hg]", _s=s, _d=d)


def v_pulse(v):
    return vit("pulse", v, "/min")


def v_spo2(v):
    return vit("oxygen_saturation", v, "%")


def v_rr(v):
    return vit("respiratory_rate", v, "/min")


def v_glu(kind, v):
    return vit("blood_glucose_" + kind, v, "mg/dL", "report")


def inv(text, status="advised", result=None, result_say=None):
    return dict(text=text, status=status, result=result, result_say=result_say)


def dx(text, names, conf="high", status="new"):
    return dict(text=text, names=names, conf=conf, status=status)


def common(rng, sex_w=0.5, amin=18, amax=65):
    return dict(age=rng.randint(amin, amax), sex="female" if rng.random() < sex_w else "male",
                routine=False, secondary=[], negated=[], vitals=[], investigations=[], advice=[],
                new_meds=[], existing_meds=[], self_med=None, family=False, history_none=False,
                reported_temp=None, follow_up=None, dxpat=None, dx_state=None)


def dsec(rng, keys, maxn):
    return [(k, (rng.randint(1, maxn), "day") if rng.random() < 0.5 else None) for k in keys]


def pick_adv(rng, pool, k):
    return rng.sample(pool, min(k, len(pool)))


def s_viral_fever(rng):
    c = common(rng, 0.5, 18, 60)
    n = rng.choice([2, 3, 3, 4, 5])
    c["primary"] = ("fever", (n, "day"))
    c["secondary"] = dsec(rng, rng.sample(["body ache", "headache", "weakness", "chills", "loss of appetite"], rng.choice([1, 2, 3])), n)
    c["negated"] = rng.sample(["dry cough", "vomiting", "loose motion", "sore throat", "burning micturition"], rng.choice([2, 3]))
    useF = rng.random() < 0.8
    if useF:
        t = rng.choice([99.5, 100, 100.4, 100.8, 101, 101.4, 102, 102.5, 103])
        vt = v_temp(t, "[degF]")
    else:
        vt = v_temp(rng.choice([37.8, 38.2, 38.5, 39, 39.4]), "Cel")
    c["vitals"] = [vt, v_pulse(rng.randint(82, 108))]
    if rng.random() < 0.5:
        c["vitals"].append(v_bp(rng.choice([108, 112, 118, 122, 126]), rng.choice([70, 74, 78, 80, 84])))
    if useF and rng.random() < 0.3:
        c["reported_temp"] = v_temp(rng.choice([99, 99.5, 100, 100.5]), "[degF]", "patient_reported")
    c["dx"] = dx("viral fever", ["viral fever", "viral infection"], rng.choice(["high", "medium"]))
    c["new_meds"] = [paracetamol(rng, "TDS", (3, "day"), timing="after_food")]
    if rng.random() < 0.3:
        c["new_meds"].append(med("ORS", "ORS", "1 sachet", "TDS", (3, "day"), None, "new", "sachet"))
    if rng.random() < 0.4:
        c["self_med"] = self_para(rng)
    c["investigations"] = [inv("CBC")] if rng.random() < 0.5 else ([inv("dengue NS1"), inv("CBC")] if n >= 4 and rng.random() < 0.5 else [])
    c["advice"] = pick_adv(rng, [
        ("adequate rest", ["Aaram kijiye, kaam se chhutti le lijiye.", "Zyada rest kijiye aur bahar mat ghumiye."]),
        ("increase fluid intake", ["Paani aur nimbu paani zyada peeyen.", "Khoob paani peeyen, dehydration na ho."]),
        ("light diet", ["Halka khana khaiye, khichdi ya dalia.", "Tala hua khana mat khaiye, halka khana lijiye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=3, unit="day", cond=True)
    c["history_none"] = rng.random() < 0.5
    return c


def s_uri(rng):
    c = common(rng, 0.5, 18, 65)
    pool = ["dry cough", "productive cough", "sore throat", "runny nose"]
    prim = rng.choice(pool)
    n = rng.choice([2, 3, 4, 5, 7])
    c["primary"] = (prim, (n, "day"))
    others = [k for k in ["dry cough", "productive cough", "sore throat", "runny nose", "fever", "headache", "body ache", "sneezing"]
              if k != prim and not (prim in ("dry cough", "productive cough") and k in ("dry cough", "productive cough"))]
    sec = rng.sample(others, rng.choice([1, 2, 3]))
    c["secondary"] = dsec(rng, sec, n)
    c["negated"] = rng.sample(["breathlessness", "chest pain", "vomiting", "loose motion"], rng.choice([2, 3]))
    c["vitals"] = [v_temp(rng.choice([98.6, 99, 99.4, 100, 100.4]), "[degF]"), v_pulse(rng.randint(72, 96)), v_spo2(rng.randint(96, 99))]
    pharyngitis = prim == "sore throat" and rng.random() < 0.5
    has_cough = prim in ("dry cough", "productive cough") or any(k in ("dry cough", "productive cough") for k, _ in c["secondary"])
    has_fever = prim == "fever" or any(k == "fever" for k, _ in c["secondary"])
    if pharyngitis:
        c["dx"] = dx("acute pharyngitis", ["pharyngitis", "throat infection"], "medium")
        c["new_meds"].append(med("Azithromycin", "Azithromycin", "500 mg", "OD", (3, "day"), "after_food"))
    else:
        c["dx"] = dx("upper respiratory tract infection", ["viral URTI", "upper respiratory infection", "viral infection"], rng.choice(["high", "medium"]))
    c["new_meds"].append(med("Cetirizine", "Cetirizine", "10 mg", "HS", (5, "day")))
    if has_cough:
        c["new_meds"].append(med("Ambroxol syrup", "Ambroxol", "10 ml", "TDS", (5, "day"), None, "new", "syrup"))
    if has_fever:
        c["new_meds"].append(paracetamol(rng, "SOS", None))
    if rng.random() < 0.35:
        c["self_med"] = self_para(rng)
    c["advice"] = pick_adv(rng, [
        ("steam inhalation", ["Din mein 2 baar bhaap lijiye.", "Garam paani ki bhaap leni hai, subah shaam."]),
        ("warm water gargles", ["Namak ke garam paani se gargle kijiye.", "Gunguna paani peeyen aur gargle karein."]),
        ("avoid cold drinks", ["Thanda paani, ice cream aur cold drink se bachiye.", "Thandi cheezein mat lijiye abhi."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=5, unit="day", cond=True)
    c["history_none"] = rng.random() < 0.4
    return c


def s_diabetes(rng):
    c = common(rng, 0.45, 40, 72)
    c.update(routine=True, dxpat="sugar", family=rng.random() < 0.4)
    outcome = rng.choices(["continue", "increase", "add"], [35, 35, 30])[0]
    c["dx"] = dx("type 2 diabetes mellitus", ["diabetes"], "high", "existing")
    c["dx_state"] = "control mein hai" if outcome == "continue" else "thoda high chal raha hai"
    mf = med("Metformin", "Metformin", "500 mg", "BD", None, "after_food", "existing", pat_gen="sugar ki goli")
    if outcome == "increase":
        mf.update(status="changed", dose="1000 mg", previous_dose="500 mg")
    c["existing_meds"] = [mf]
    if outcome == "add":
        c["new_meds"] = [med("Glimepiride", "Glimepiride", "1 mg", "OD", (1, "month"), "before_food")]
    prim_pool = ["excessive thirst", "frequent urination", "tingling in feet", "weakness", "blurred vision"]
    sec = rng.sample(prim_pool, rng.choice([0, 1, 2]))
    c["secondary"] = [(k, None) for k in sec]
    c["negated"] = rng.sample([k for k in ["vomiting", "chest pain", "dizziness", "blurred vision"] if k not in sec], 2)
    f = rng.choice([96, 108, 118, 126, 138, 152, 168]) if outcome == "continue" else rng.choice([138, 152, 168, 176, 188])
    c["vitals"] = [v_bp(rng.choice([118, 122, 128, 132, 138]), rng.choice([74, 78, 82, 86])),
                   v_glu("fasting", f), v_glu("postprandial", f + rng.choice([40, 52, 64, 78]))]
    hb = {"continue": rng.choice([6.6, 6.8, 7.0]), "increase": rng.choice([8.1, 8.4, 8.8, 9.2]), "add": rng.choice([7.6, 7.9, 8.2])}[outcome]
    c["investigations"] = [inv("HbA1c", "reviewed", f"{fnum(hb)} %", f"{fnum(hb)}%")]
    if rng.random() < 0.6:
        c["investigations"].append(inv(rng.choice(["lipid profile", "KFT", "urine routine"])))
    c["advice"] = pick_adv(rng, [
        ("limit sugar and sweets", ["Meetha aur chawal kam kijiye.", "Mithai aur chini bilkul kam kar dijiye."]),
        ("daily walking 30 minutes", ["Roz 30 minute walk kijiye.", "Subah shaam thoda tahliye, kam se kam 30 minute."]),
        ("foot care", ["Pairon ka dhyan rakhiye, chot na lage.", "Roz pair dekhte rahiye, koi zakhm toh nahi."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=1, unit="month", cond=False)
    return c


def s_hypertension(rng):
    c = common(rng, 0.45, 40, 75)
    c.update(routine=True, dxpat="BP", family=rng.random() < 0.4)
    outcome = rng.choices(["continue", "add", "increase"], [35, 35, 30])[0]
    c["dx"] = dx("hypertension", ["hypertension", "high BP"], "high", "existing")
    c["dx_state"] = "control mein hai" if outcome == "continue" else "abhi high aa raha hai"
    am = med("Amlodipine", "Amlodipine", "5 mg", "OD", None, None, "existing", pat_gen="BP ki goli")
    if outcome == "increase":
        am.update(status="changed", dose="10 mg", previous_dose="5 mg")
    c["existing_meds"] = [am]
    if outcome == "add":
        c["new_meds"] = [med("Telmisartan", "Telmisartan", "40 mg", "OD", (1, "month"))]
    if outcome == "continue":
        c["vitals"] = [v_bp(rng.choice([118, 122, 126, 130]), rng.choice([76, 80, 84])), v_pulse(rng.randint(66, 84))]
    else:
        c["vitals"] = [v_bp(rng.choice([146, 150, 156, 162]), rng.choice([92, 96, 100])), v_pulse(rng.randint(72, 92))]
    sec = rng.sample(["headache", "dizziness"], rng.choice([0, 1, 2]))
    c["secondary"] = [(k, None) for k in sec]
    c["negated"] = rng.sample([k for k in ["chest pain", "breathlessness", "dizziness", "headache"] if k not in sec], 2)
    c["investigations"] = rng.choice([[], [inv("ECG")], [inv("lipid profile")], [inv("ECG"), inv("KFT")]])
    c["advice"] = pick_adv(rng, [
        ("reduce salt intake", ["Namak kam khaiye.", "Khane mein namak aur achar kam kijiye."]),
        ("regular walking", ["Roz walk kijiye.", "Subah thodi der tahliye, regular."]),
        ("monitor BP at home", ["Ghar pe BP check karte rahiye.", "Hafte mein 2-3 baar ghar pe BP naapiye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = rng.choice([dict(n=2, unit="week", cond=False), dict(n=1, unit="month", cond=False)])
    return c


def s_gastritis(rng):
    c = common(rng, 0.5, 20, 65)
    prim = rng.choice(["acidity", "abdominal pain", "sour belching"])
    c["primary"] = (prim, (rng.choice([3, 5, 7, 10]), "day") if rng.random() < 0.7 else (rng.choice([2, 3]), "week"))
    sec = [k for k in ["bloating", "nausea", "sour belching", "acidity", "loss of appetite", "abdominal pain"] if k != prim]
    c["secondary"] = dsec(rng, rng.sample(sec, rng.choice([1, 2, 3])), 3)
    c["negated"] = rng.sample(["vomiting", "loose motion", "chest pain", "black stools"], rng.choice([2, 3]))
    c["vitals"] = [v_pulse(rng.randint(72, 92))] + ([v_bp(rng.choice([112, 118, 124, 128]), rng.choice([72, 78, 82]))] if rng.random() < 0.5 else [])
    if rng.random() < 0.7:
        c["dx"] = dx("gastritis", ["gastritis", "gastric problem"], rng.choice(["high", "medium"]))
    else:
        c["dx"] = dx("gastroesophageal reflux disease", ["GERD", "acid reflux"], "medium")
    c["new_meds"] = [med("Pantoprazole", "Pantoprazole", "40 mg", "OD", (2, "week"), "empty_stomach"),
                     med("Antacid syrup", "Antacid", "10 ml", "SOS", None, "after_food", "new", "syrup")]
    if rng.random() < 0.4:
        c["self_med"] = med("Eno", "Antacid", None, None, None, None, "self_taken",
                            self_say=["Haan, Eno pi li thi.", "Eno liya tha ek do baar."])
    c["investigations"] = [inv("H pylori test")] if rng.random() < 0.2 else []
    c["advice"] = pick_adv(rng, [
        ("avoid spicy and oily food", ["Teekha aur tala hua khana band kijiye.", "Masaledar aur oily khana avoid kijiye."]),
        ("limit tea and coffee", ["Chai coffee kam kijiye.", "Chai aur coffee zyada mat peeyen."]),
        ("small frequent meals", ["Thoda thoda karke baar baar khaiye.", "Ek saath zyada mat khaiye, chhote meals lijiye."]),
        ("avoid lying down after meals", ["Khane ke baad turant let mat jaiye, 2-3 ghante ruk ke.", "Khana khaake seedha mat soiye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=2, unit="week", cond=False)
    c["family"] = rng.random() < 0.2
    c["history_none"] = rng.random() < 0.4
    return c


def s_gastroenteritis(rng):
    c = common(rng, 0.5, 18, 60)
    n = rng.choice([1, 2, 3])
    c["primary"] = ("loose motion", (n, "day"))
    sec = rng.sample(["vomiting", "abdominal pain", "nausea", "fever", "weakness"], rng.choice([2, 3]))
    c["secondary"] = dsec(rng, sec, n)
    c["negated"] = ["blood in stool"] + (["black stools"] if "fever" in sec else ["fever"])
    c["vitals"] = [v_temp(rng.choice([98.6, 99.4, 100, 100.8]), "[degF]"), v_pulse(rng.randint(88, 112)),
                   v_bp(rng.choice([100, 104, 108, 114]), rng.choice([64, 68, 72, 76]))]
    c["dx"] = dx("acute gastroenteritis", ["acute gastroenteritis", "stomach infection", "food poisoning jaisa infection"], "high")
    c["new_meds"] = [med("ORS", "ORS", "1 sachet", "AFTER_STOOL", (3, "day"), None, "new", "sachet")]
    if "vomiting" in sec or "nausea" in sec:
        c["new_meds"].append(med("Ondansetron", "Ondansetron", "4 mg", "SOS", None))
    if "fever" in sec:
        c["new_meds"].append(paracetamol(rng, "SOS", None))
    c["investigations"] = [inv("stool routine")] if rng.random() < 0.3 else []
    c["advice"] = pick_adv(rng, [
        ("boiled or clean water", ["Ubla hua ya filter ka paani peeyen.", "Paani ubaal ke peeyen."]),
        ("light diet such as khichdi and curd", ["Khichdi aur dahi lijiye.", "Halka khana, khichdi dahi ya dalia."]),
        ("avoid outside and oily food", ["Bahar ka aur oily khana bilkul nahi.", "Street food abhi band kijiye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=2, unit="day", cond=True)
    c["history_none"] = rng.random() < 0.4
    return c


def s_headache(rng):
    c = common(rng, 0.55, 18, 55)
    c["primary"] = ("headache", rng.choice([(2, "day"), (3, "day"), (1, "week"), (2, "week")]))
    c["secondary"] = dsec(rng, rng.sample(["photophobia", "nausea", "dizziness", "weakness"], rng.choice([1, 2])), 2)
    c["negated"] = rng.sample(["vomiting", "fever", "blurred vision"], rng.choice([2, 3]))
    c["vitals"] = [v_bp(rng.choice([110, 118, 124, 130, 134]), rng.choice([70, 76, 80, 84])), v_pulse(rng.randint(68, 90))]
    if rng.random() < 0.5:
        c["dx"] = dx("migraine", ["migraine"], "medium")
    else:
        c["dx"] = dx("tension-type headache", ["tension headache"], "high")
    if rng.random() < 0.5:
        c["new_meds"] = [paracetamol(rng, "SOS", None)]
    else:
        c["new_meds"] = [med("Ibuprofen", "Ibuprofen", "400 mg", "SOS", None, "after_food")]
    if rng.random() < 0.3:
        c["self_med"] = self_para(rng)
    c["investigations"] = [inv("eye examination")] if rng.random() < 0.3 else []
    c["advice"] = pick_adv(rng, [
        ("adequate sleep", ["Poori neend lijiye, 7-8 ghante.", "Raat ko time pe soiye."]),
        ("adequate hydration", ["Paani khoob peeyen.", "Din bhar paani peete rahiye."]),
        ("reduce screen time", ["Mobile aur laptop ka time kam kijiye.", "Screen se thoda break lete rahiye."]),
        ("do not skip meals", ["Khana skip mat kijiye.", "Time pe khana khaiye, bhooke mat rahiye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=2, unit="week", cond=True)
    c["history_none"] = rng.random() < 0.5
    return c


def s_asthma(rng):
    c = common(rng, 0.5, 20, 65)
    known = rng.random() < 0.4
    prim = rng.choice(["wheezing", "breathlessness"])
    c["primary"] = (prim, rng.choice([(3, "day"), (5, "day"), (1, "week"), (2, "week")]))
    sec = [k for k in ["dry cough", "chest tightness", "breathlessness", "wheezing"] if k != prim]
    c["secondary"] = dsec(rng, rng.sample(sec, rng.choice([1, 2])), 3)
    c["negated"] = rng.sample(["fever", "chest pain"], 2)
    c["vitals"] = [v_spo2(rng.randint(94, 98)), v_pulse(rng.randint(84, 104)), v_rr(rng.randint(18, 24))]
    c["dx"] = dx("bronchial asthma", ["asthma", "bronchial asthma"], rng.choice(["high", "medium"]), "existing" if known else "new")
    if known:
        c["dxpat"] = "asthma"
    c["new_meds"] = [med("Salbutamol inhaler", "Salbutamol", "2 puffs", "SOS", None, None, "new", "inhaler"),
                     med("Montelukast", "Montelukast", "10 mg", "HS", (1, "month"))]
    c["investigations"] = [inv("chest X-ray")] + ([inv("PFT")] if rng.random() < 0.3 else [])
    c["advice"] = pick_adv(rng, [
        ("avoid dust and smoke", ["Dhool, dhuan aur smoke se bachiye.", "Dhool mitti ke paas mat jaiye, mask lagaiye."]),
        ("avoid cold drinks and ice", ["Thanda paani aur ice cream mat lijiye.", "Fridge ka thanda paani avoid kijiye."]),
        ("avoid smoking", ["Dhoomrapaan bilkul band kijiye.", "Beedi cigarette se door rahiye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=2, unit="week", cond=False)
    c["family"] = rng.random() < 0.3
    return c


def s_allergic_rhinitis(rng):
    c = common(rng, 0.5, 15, 55)
    prim = rng.choice(["sneezing", "runny nose"])
    c["primary"] = (prim, rng.choice([(1, "week"), (2, "week"), (5, "day"), (1, "month")]))
    sec = [k for k in ["itchy eyes", "runny nose", "sneezing", "headache"] if k != prim]
    c["secondary"] = dsec(rng, rng.sample(sec, rng.choice([1, 2])), 5)
    c["negated"] = rng.sample(["fever", "breathlessness", "sore throat"], 2)
    c["vitals"] = [v_temp(rng.choice([98.4, 98.6, 99]), "[degF]")] if rng.random() < 0.6 else [v_spo2(rng.randint(97, 99))]
    c["dx"] = dx("allergic rhinitis", ["allergic rhinitis", "dust allergy", "allergy"], "high")
    c["new_meds"] = [med("Cetirizine", "Cetirizine", "10 mg", "HS", (7, "day"))]
    if rng.random() < 0.5:
        c["new_meds"].append(med("Montelukast", "Montelukast", "10 mg", "HS", (7, "day")))
    if rng.random() < 0.4:
        c["new_meds"].append(med("Saline nasal spray", "Sodium chloride", "2 sprays", "BD", (7, "day"), None, "new", "spray"))
    if rng.random() < 0.3:
        c["self_med"] = med("Cetirizine", "Cetirizine", "10 mg", None, None, None, "self_taken")
    c["advice"] = pick_adv(rng, [
        ("avoid dust exposure", ["Dhool se bachiye, mask pehniye.", "Ghar mein jhaadu ki jagah geela poncha lagaiye."]),
        ("wash bedsheets regularly", ["Bedsheet aur takiye ke cover hafte mein dhoyein.", "Chadar roz jhaad kar bichhaiye."]),
        ("steam inhalation", ["Bhaap lijiye subah shaam.", "Garam paani ki bhaap lena faydemand hoga."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=1, unit="week", cond=True)
    c["history_none"] = rng.random() < 0.4
    return c


def s_hypothyroid(rng):
    c = common(rng, 0.8, 25, 60)
    c.update(routine=True, dxpat="thyroid", family=rng.random() < 0.4)
    outcome = rng.choices(["continue", "increase"], [45, 55])[0]
    dose0 = rng.choice([50, 75, 100])
    step = {50: 75, 75: 100, 100: 125}
    lv = med("Levothyroxine", "Levothyroxine", f"{dose0} mcg", "OD", None, "empty_stomach", "existing", pat_gen="thyroid ki goli")
    if outcome == "increase":
        lv.update(status="changed", dose=f"{step[dose0]} mcg", previous_dose=f"{dose0} mcg")
    c["existing_meds"] = [lv]
    c["dx"] = dx("hypothyroidism", ["hypothyroidism", "thyroid"], "high", "existing")
    c["dx_state"] = "control mein hai" if outcome == "continue" else "thoda badha hua hai"
    sec = rng.sample(["weakness", "weight gain", "hair fall", "constipation", "cold intolerance"], rng.choice([1, 2, 3]) if outcome == "increase" else rng.choice([0, 1]))
    c["secondary"] = [(k, None) for k in sec]
    c["negated"] = ["palpitations"] + ([rng.choice(["vomiting", "chest pain"])] if rng.random() < 0.6 else [])
    tsh = rng.choice([1.8, 2.4, 3.1, 3.8]) if outcome == "continue" else rng.choice([7.2, 8.4, 9.6, 11.2])
    c["investigations"] = [inv("TSH", "reviewed", f"{fnum(tsh)} mIU/L", fnum(tsh))]
    if outcome == "increase":
        c["investigations"].append(inv("TSH"))
    c["vitals"] = [v_pulse(rng.randint(62, 84))] + ([v_bp(rng.choice([110, 118, 124]), rng.choice([70, 76, 80]))] if rng.random() < 0.5 else [])
    c["advice"] = pick_adv(rng, [
        ("take levothyroxine on empty stomach in the morning", ["Subah khali pet lijiye, aadhe ghante tak kuch mat khaiye.", "Dawai subah khali pet hi leni hai."]),
        ("regular exercise", ["Thodi exercise kijiye, roz.", "Walk ya yoga regular kijiye."]),
    ], 2)
    c["follow_up"] = dict(n=6, unit="week", cond=False) if outcome == "increase" else dict(n=3, unit="month", cond=False)
    return c


def s_msk(rng):
    c = common(rng, 0.5, 30, 70)
    prim = rng.choice(["back pain", "knee pain"])
    c["primary"] = (prim, rng.choice([(3, "day"), (1, "week"), (2, "week"), (1, "month")]))
    c["secondary"] = dsec(rng, rng.sample(["body ache", "weakness"], rng.choice([0, 1, 2])), 3)
    c["negated"] = ["fever", "tingling in feet"] if prim == "back pain" else ["fever", "back pain"]
    c["vitals"] = [v_bp(rng.choice([118, 124, 130, 136]), rng.choice([76, 80, 84]))] if rng.random() < 0.6 else []
    if prim == "back pain":
        c["dx"] = dx("mechanical low back pain", ["muscle spasm", "mechanical back pain", "back strain"], "medium")
    else:
        c["dx"] = dx("osteoarthritis of knee", ["knee osteoarthritis", "ghutne ka arthritis"], "medium")
    c["new_meds"] = [med("Ibuprofen", "Ibuprofen", "400 mg", "TDS", (5, "day"), "after_food"),
                     med("Pantoprazole", "Pantoprazole", "40 mg", "OD", (5, "day"), "before_food")]
    if prim == "back pain" and rng.random() < 0.5:
        c["new_meds"].append(med("Thiocolchicoside", "Thiocolchicoside", "4 mg", "BD", (5, "day")))
    if rng.random() < 0.4:
        c["self_med"] = med("Volini spray", "Diclofenac topical", None, None, None, None, "self_taken",
                            self_say=["Haan, Volini spray lagaya tha.", "Volini spray lagaya tha par kuch farak nahi pada."])
    c["investigations"] = [inv("X-ray lumbar spine" if prim == "back pain" else "X-ray knee")] if rng.random() < 0.4 else []
    c["advice"] = pick_adv(rng, [
        ("hot compress", ["Garam sek kijiye din mein 2 baar.", "Garam paani ki bottle se sek lijiye."]),
        ("avoid lifting heavy weights", ["Bhaari saamaan mat uthaiye.", "Jhukkar bhaari cheez uthane se bachiye."]),
        ("physiotherapy exercises", ["Physio wali exercise kijiye, roz.", "Halki stretching exercise karte rahiye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=1, unit="week", cond=True)
    c["history_none"] = rng.random() < 0.4
    return c


def s_uti(rng):
    c = common(rng, 0.75, 18, 60)
    n = rng.choice([1, 2, 3])
    c["primary"] = ("burning micturition", (n, "day"))
    c["secondary"] = dsec(rng, rng.sample(["frequent urination", "abdominal pain"], rng.choice([1, 2])), n)
    c["negated"] = ["fever", "vomiting", "back pain"][:rng.choice([2, 3])]
    c["vitals"] = [v_temp(rng.choice([98.4, 98.6, 99]), "[degF]")] + ([v_bp(rng.choice([110, 118, 124]), rng.choice([70, 76, 80]))] if rng.random() < 0.5 else [])
    c["dx"] = dx("urinary tract infection", ["UTI", "urine infection", "urinary tract infection"], "medium")
    c["new_meds"] = [med("Nitrofurantoin", "Nitrofurantoin", "100 mg", "BD", (5, "day"), "after_food")]
    if rng.random() < 0.3:
        c["new_meds"].append(paracetamol(rng, "SOS", None))
    c["investigations"] = [inv("urine routine"), inv("urine culture")]
    c["advice"] = pick_adv(rng, [
        ("drink plenty of water", ["Paani khoob peeyen, din mein 3 litre.", "Zyada se zyada paani peeyen."]),
        ("do not hold urine", ["Peshab rokiye mat.", "Peshab aane par turant jaaiye, rokna nahi."]),
        ("maintain genital hygiene", ["Safai ka dhyan rakhiye.", "Hygiene ka khayal rakhiye, saaf kapde pehniye."]),
    ], rng.choice([2, 3]))
    c["follow_up"] = dict(n=5, unit="day", cond=False)
    c["history_none"] = rng.random() < 0.4
    return c


SCENARIOS = {
    "viral_fever": s_viral_fever, "upper_respiratory_infection": s_uri, "diabetes_followup": s_diabetes,
    "hypertension_followup": s_hypertension, "gastritis_acidity": s_gastritis,
    "acute_gastroenteritis": s_gastroenteritis, "headache_migraine": s_headache,
    "asthma": s_asthma, "allergic_rhinitis": s_allergic_rhinitis,
    "hypothyroid_followup": s_hypothyroid, "musculoskeletal_pain": s_msk, "urinary_tract_infection": s_uti,
}

PROPER = {"dolo", "crocin", "paracetamol", "metformin", "amlodipine", "telmisartan", "glimepiride", "pantoprazole",
          "cetirizine", "ambroxol", "azithromycin", "ondansetron", "salbutamol", "montelukast", "levothyroxine",
          "ibuprofen", "nitrofurantoin", "antacid", "thiocolchicoside", "saline", "eno", "volini", "ors",
          "cbc", "ecg", "hba1c", "tsh", "kft", "pft", "bp", "spo2", "x-ray", "i"}


# ----------------------------------------------------------------------------
# Dialogue builder
# ----------------------------------------------------------------------------
class Builder:
    def __init__(self, rng, style, sex):
        self.rng, self.style, self.sex = rng, style, sex
        self.turns = []
        self.ent = {k: [] for k in ["demographics", "symptoms", "diagnosis", "medications", "vitals",
                                    "investigations", "advice", "follow_up"]}
        self.hard = set()

    def add(self, kind, obj):
        obj["source_segments"] = []
        self.ent[kind].append(obj)
        return (kind, len(self.ent[kind]) - 1)

    def say(self, spk, text, tags=(), raw=False, overlap=False, distractor=None):
        self.turns.append(dict(spk=spk, text=text, tags=list(tags), raw=raw, overlap=overlap,
                               distractor=distractor, cont=False))
        return len(self.turns) - 1


def gen(text, sex):
    f = sex == "female"
    return (text.replace("{lta}", "leti" if f else "leta").replace("{jata}", "jaati" if f else "jaata")
            .replace("{ka}", "ki" if f else "ka"))


def cap(s):
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
        if ch.isdigit():
            return s
    return s


def sym_term(rng, key):
    return rng.choice(SYM[key]["t"])


def sym_q(rng, key):
    s = SYM[key]
    return rng.choice(s["q"]) if "q" in s else rng.choice(GEN_Q).format(t=sym_term(rng, key))


def sym_pos(rng, key):
    s = SYM[key]
    return rng.choice(s["pos"]) if "pos" in s else rng.choice(GEN_POS).format(t=sym_term(rng, key))


def sym_neg(rng, key):
    s = SYM[key]
    return rng.choice(s["neg"]) if "neg" in s else rng.choice(GEN_NEG).format(t=sym_term(rng, key))


def sym_dpos(rng, key, d, primary=False):
    s = SYM[key]
    if "dpos" in s:
        return rng.choice(s["dpos"]).format(d=d)
    pool = CC_GEN if primary else GEN_DPOS
    return rng.choice(pool).format(t=sym_term(rng, key), d=d)


def vital_text(rng, v):
    t, val = v["type"], v["value"]
    if t == "temperature":
        x = fnum(val)
        if v["unit"] == "[degF]":
            return rng.choice(["Temperature {x} hai.", "Temperature {x} Fahrenheit hai.", "Bukhar {x} degree hai.",
                               "{x} degree Fahrenheit temperature hai."]).format(x=x)
        return rng.choice(["Temperature {x} Celsius hai.", "{x} degree Celsius temperature hai.", "Temperature {x} hai."]).format(x=x)
    if t == "blood_pressure":
        return rng.choice(["BP {s} by {d} hai.", "Blood pressure {s} by {d} aa raha hai.", "BP {s} by {d} hai, check kar liya."]).format(s=v["_s"], d=v["_d"])
    if t == "pulse":
        return rng.choice(["Pulse {x} hai.", "Pulse rate {x} per minute hai.", "Dhadkan {x} hai, regular hai."]).format(x=val)
    if t == "oxygen_saturation":
        return rng.choice(["SpO2 {x}% hai.", "Oxygen saturation {x} percent hai.", "SpO2 {x} hai, theek hai."]).format(x=val)
    if t == "respiratory_rate":
        return rng.choice(["Respiratory rate {x} per minute hai.", "Saans ki speed {x} per minute hai."]).format(x=val)
    if t == "blood_glucose_fasting":
        return rng.choice(["Report mein fasting sugar {x} hai.", "Aapka fasting sugar {x} aaya hai."]).format(x=val)
    if t == "blood_glucose_postprandial":
        return rng.choice(["Aur khane ke baad wala sugar {x} hai.", "Post meal sugar {x} hai."]).format(x=val)
    raise ValueError(t)


def vital_digits(v):
    if v["type"] == "blood_pressure":
        return [str(v["_s"]), str(v["_d"])]
    return [fnum(v["value"])]


def dose_say(rng, m):
    d = m.get("dose")
    if not d:
        return ""
    if d.endswith(" mg") and rng.random() < 0.35:
        return d[:-3]
    return d


def join(*parts):
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).replace(" ,", ",").strip()


def build_dialogue(scn, idx, seed, per_scn, scn_pos):
    rng = random.Random(seed * 100003 + idx)
    style = rng.choices(["calm", "talkative", "terse", "anxious"], [4, 3, 2, 2])[0]
    c = SCENARIOS[scn](rng)
    b = Builder(rng, style, c["sex"])
    D, P = "doctor", "patient"

    # --- greeting + age
    b.say(D, rng.choice(GREET_DOC), raw=True)
    b.say(P, rng.choice(PAT_GREET), raw=True)
    if rng.random() < 0.8:
        b.say(D, rng.choice(DOC_AGE))
        tg = b.add("demographics", dict(age=c["age"]))
        b.say(P, rng.choice(PAT_AGE).format(age=c["age"], ka="{ka}"), [tg])
    b.say(D, rng.choice(OPEN_FU if c["routine"] else OPEN_DOC))

    # --- chief complaint
    sym_tags = {}
    dx_tag = None
    pending = [(k, d) for k, d in c["secondary"]]
    if c["routine"]:
        dx_tag = b.add("diagnosis", dict(text=c["dx"]["text"], confidence="high", status="existing"))
        b.say(P, rng.choice(PAT_ROUTINE).format(dxpat=c["dxpat"]), [dx_tag])
        if rng.random() < 0.6:
            b.say(D, rng.choice(["Kitne saal se hai ye?", "Kab se chal raha hai ye?"]))
            b.say(P, rng.choice(["{y} saal se.", "Karib {y} saal ho gaye."]).format(y=rng.randint(2, 12)),
                  distractor=dict(kind="numeric_distractor", note="years since diagnosis; not a medication duration"))
            b.hard.add("numeric_distractor")
    else:
        k, d = c["primary"]
        n, unit = d
        lang = "hi" if rng.random() < 0.75 else "en"
        dtxt = fmt_dur(n, unit, lang)
        if rng.random() < 0.12 and n >= 1:
            wrong = n + 1 if n == 1 or rng.random() < 0.5 else n - 1
            dtxt = f"{fmt_dur(wrong, unit, lang)}... nahi nahi, {fmt_dur(n, unit, lang)}"
            b.hard.add("self_correction")
        tg = b.add("symptoms", dict(text=SYM[k]["name"], present=True, duration=dur_norm(d)))
        text = sym_dpos(rng, k, dtxt, primary=True)
        tags = [tg]
        if pending and rng.random() < 0.35 and style != "terse":
            k2, d2 = pending.pop(0)
            t2 = sym_term(rng, k2)
            tg2 = b.add("symptoms", dict(text=SYM[k2]["name"], present=True, duration=None))
            text += " " + rng.choice(["Saath mein {t} bhi hai.", "Aur {t} bhi hai."]).format(t=t2)
            tags.append(tg2)
        if style == "talkative" and rng.random() < 0.6:
            text += " " + rng.choice(CHATTER)
            b.hard.add("chatter")
        b.say(P, text, tags)

    # --- symptom Q&A
    items = [("pos", k, d) for k, d in pending]
    negs = list(c["negated"])
    if len(negs) >= 2 and rng.random() < 0.5:
        items.append(("gneg", negs, None))
        b.hard.add("grouped_negation")
    else:
        items += [("neg", k, None) for k in negs]
    rng.shuffle(items)
    b.hard.add("negation")
    for kind, k, d in items:
        if kind == "pos":
            tg = b.add("symptoms", dict(text=SYM[k]["name"], present=True, duration=dur_norm(d) if d else None))
            b.say(D, cap(sym_q(rng, k)))
            if d:
                b.say(P, sym_dpos(rng, k, fmt_dur(d[0], d[1], "hi" if rng.random() < 0.8 else "en")), [tg])
            else:
                b.say(P, sym_pos(rng, k), [tg])
        elif kind == "neg":
            tg = b.add("symptoms", dict(text=SYM[k]["name"], present=False, duration=None))
            b.say(D, cap(sym_q(rng, k)))
            b.say(P, sym_neg(rng, k), [tg])
        else:
            tgs = [b.add("symptoms", dict(text=SYM[x]["name"], present=False, duration=None)) for x in k]
            terms = [sym_term(rng, x) for x in k]
            lst = terms[0] if len(terms) == 1 else ", ".join(terms[:-1]) + " ya " + terms[-1]
            b.say(D, cap(rng.choice(GROUP_Q).format(lst=lst)))
            b.say(P, rng.choice(GROUP_NEG), tgs)

    # --- patient-reported temperature
    if c["reported_temp"]:
        v = c["reported_temp"]
        tg = b.add("vitals", dict(type=v["type"], value=v["value"], unit=v["unit"], source=v["source"]))
        b.say(D, rng.choice(["Ghar pe temperature check kiya tha?", "Thermometer se dekha tha kuch?"]))
        b.say(P, rng.choice(["Haan, ghar pe thermometer se {x} tha.", "Ghar pe {x} aaya tha Fahrenheit mein."]).format(x=fnum(v["value"])), [tg])
        b.hard.add("patient_reported_vital")

    # --- self medication
    sm = c["self_med"]
    if sm:
        tg = b.add("medications", dict(name=sm["name"], generic=sm["generic"], dose=sm["dose"], frequency=None,
                                       duration=None, instructions=None, status="self_taken"))
        b.say(D, rng.choice(["Koi dawai li hai ab tak?", "Kuch dawai khayi aapne?", "Ghar pe kuch liya hai kya?"]))
        if sm.get("self_say"):
            b.say(P, rng.choice(sm["self_say"]), [tg])
        else:
            b.say(P, rng.choice(["Haan, {n} {d} li thi.", "{n} {d} khayi thi kal raat, par aaram nahi aaya.",
                                 "Maine ghar pe {n} {d} le li thi.", "Ek {n} {d} li thi subah."]).format(n=sm["name"], d=sm["dose"]), [tg])
        b.hard.add("self_medication")
    elif not c["routine"] and rng.random() < 0.4:
        b.say(D, rng.choice(["Koi dawai li hai ab tak?", "Kuch dawai khayi aapne?"]))
        b.say(P, rng.choice(["Nahi, abhi tak kuch nahi li.", "Nahi doctor, seedha aapke paas aaya."]))

    # --- history: existing medications (routine) / family / allergy / past illness
    exist_tags = {}
    if c["existing_meds"]:
        b.say(D, rng.choice(["Abhi kaun si dawai chal rahi hai?", "Dawai kaun si le rahe hain abhi?"]))
        for i, m in enumerate(c["existing_meds"]):
            tg = b.add("medications", dict(name=m["name"], generic=m["generic"], dose=m["dose"], frequency=FREQ_NORM[m["freq"]],
                                           duration=None, instructions=TIMING_NORM[m["timing"]], status=m["status"],
                                           **({"previous_dose": m["previous_dose"]} if m.get("previous_dose") else {})))
            exist_tags[id(m)] = tg
            fz = rng.choice(FREQ[m["freq"]])
            tm = ", " + rng.choice(TIMING[m["timing"]]) if m["timing"] else ""
            first_dose = m.get("previous_dose") or m["dose"]
            if rng.random() < 0.4:
                b.say(P, f"{m['pat_gen'].capitalize()} {{lta}} hoon, {fz}.", [tg])
                b.say(D, join(f"{m['name']} {first_dose}, {fz}{tm}, theek?"), [tg])
                b.say(P, rng.choice(["Haan ji, wahi hai.", "Ji haan, wahi.", "Haan, wahi dawai hai."]), [tg])
                b.hard.add("brand_confirmation")
            else:
                b.say(P, rng.choice(["{n} {d} {{lta}} hoon, {f}{t}.", "Main {n} {d} {{lta}} hoon, {f}{t}.", "{n} {d} chal rahi hai, {f}{t}."])
                      .format(n=m["name"], d=first_dose, f=fz, t=tm), [tg])
            if i < len(c["existing_meds"]) - 1:
                b.say(D, "Aur koi?")
        b.say(D, "Dawai regular le rahe hain na?")
        b.say(P, rng.choice(["Haan, regular {lta} hoon.", "Kabhi kabhi bhool {jata} hoon, par zyadatar regular.", "Haan ji, roz {lta} hoon."]))
    if c["history_none"]:
        b.say(D, rng.choice(["Pehle se koi bimari? Sugar, BP kuch?", "Koi purani bimari toh nahi hai?"]))
        b.say(P, rng.choice(["Nahi, kuch nahi hai.", "Nahi doctor, bilkul theek rehta hoon."]))
    if c["family"]:
        pool = list(FAMILY_COND)
        own = {"sugar": "diabetes_followup", "BP": "hypertension_followup", "thyroid": "hypothyroid_followup", "asthma": "asthma"}
        pool = [x for x in pool if own.get(x) != scn]
        cond = rng.choice(pool)
        rel = rng.choice(FAMILY_REL)
        b.say(D, rng.choice(["Ghar mein kisi ko sugar, BP ya koi aur bimari hai?", "Family mein kisi ko koi bimari hai?"]))
        b.say(P, rng.choice(["{r} ko {c} hai.", "Haan, {r} ko {c} hai, dawai chalti hai unki."]).format(r=rel, c=cond),
              distractor=dict(kind="family_history", must_not_extract=FAMILY_COND[cond]))
        b.hard.add("family_history_distractor")
    if rng.random() < 0.35:
        b.say(D, rng.choice(["Kisi dawai se allergy toh nahi hai?", "Koi allergy hai dawai ya khane se?"]))
        b.say(P, rng.choice(["Nahi, koi allergy nahi hai.", "Nahi doctor, kisi cheez se allergy nahi."]))

    # --- vitals
    b.say(D, rng.choice(DOC_VITAL_INTRO))
    b.say(P, rng.choice(["Ji.", "Theek hai.", "Haan ji."]), raw=True)
    vlist = [vv for vv in c["vitals"]]
    rng.shuffle(vlist)
    i = 0
    while i < len(vlist):
        k = rng.choice([1, 1, 2, 3])
        chunk = vlist[i:i + k]
        i += k
        tags = []
        sents = []
        for v in chunk:
            vv = {kk: v[kk] for kk in ("type", "value", "unit", "source")}
            tags.append(b.add("vitals", vv))
            sents.append(vital_text(rng, v))
        b.say(D, " ".join(sents), tags)
        if rng.random() < 0.3:
            b.say(P, rng.choice(["Kitna hai?", "Normal hai na?", "Theek hai doctor?"]), raw=True)
            b.say(D, rng.choice(["Haan, dekhte hain.", "Ek minute, aage bhi dekh leta hoon.", "Chaliye, aage dekhte hain."]), raw=True)

    # --- diagnosis
    dxd = c["dx"]
    if c["routine"]:
        b.say(D, rng.choice(DX_STATE).format(dx=rng.choice(dxd["names"]), state=c["dx_state"]), [dx_tag])
    else:
        dtag = b.add("diagnosis", dict(text=dxd["text"], confidence=dxd["conf"], status=dxd["status"]))
        tpl = DX_HIGH if dxd["conf"] == "high" else DX_MED
        b.say(D, rng.choice(tpl).format(dx=rng.choice(dxd["names"])), [dtag])
        if dxd["status"] == "existing" and c.get("dxpat"):
            pass
        if style == "anxious":
            b.say(P, rng.choice(ANXIOUS_Q))
            b.say(D, rng.choice(ANXIOUS_A))
            b.hard.add("anxious_patient")

    # --- investigations (before or after medicines)
    def do_inv():
        for iv in c["investigations"]:
            tg = b.add("investigations", dict(text=iv["text"], status=iv["status"], result=iv["result"]))
            if iv["status"] == "reviewed":
                b.say(D, rng.choice(["Aapka {x} dekha, {r} aaya hai.", "Report mein {x} {r} hai.", "{x} {r} aaya hai aapka."])
                      .format(x=iv["text"], r=iv["result_say"]), [tg])
            else:
                b.say(D, rng.choice(["{x} kara lijiye.", "Ek {x} likh deta hoon, kal subah karwa lijiye.",
                                     "{x} karwa lena, report ke saath aaiye.", "Saath mein {x} bhi kara lijiye."]).format(x=iv["text"]), [tg])
            if rng.random() < 0.4:
                b.say(P, rng.choice(ACKS), raw=True)

    inv_first = rng.random() < 0.3
    if inv_first:
        do_inv()

    # --- medications
    def prescribe(m, tg):
        dsy = dose_say(rng, m)
        name = m["name"]
        t = ", " + rng.choice(TIMING[m["timing"]]) if m["timing"] else ""
        fz = rng.choice(FREQ[m["freq"]]) if m["freq"] else ""
        dur = dur_phrase(rng, m["duration"]) if m["duration"] else ""
        if m["status"] == "existing":
            tpl = rng.choice(["{n} {d} wahi chalne dijiye, {f}{t}.", "{n} {d} continue kijiye, {f}{t}.",
                              "Jo {n} {d} le rahe hain, woh jaari rakhiye, {f}{t}."])
            b.say(D, join(tpl.format(n=name, d=dsy, f=fz, t=t)), [tg])
        elif m["status"] == "changed":
            old = m["previous_dose"][:-3] if m["previous_dose"].endswith(" mg") else m["previous_dose"]
            tpl = rng.choice(["{n} ab {d} kar dete hain, {f}{t}.", "{n} {o} se badha ke {d} kar raha hoon, {f}{t}.",
                              "Dose badhani padegi, {n} {d} kar dijiye, {f}{t}."])
            b.say(D, join(tpl.format(n=name, d=m["dose"], o=old, f=fz, t=t)), [tg])
            b.hard.add("dose_change")
        else:
            if m.get("alt_dose") and rng.random() < 0.12:
                alt = m["alt_dose"][:-3]
                dsy = f"{alt}... nahi nahi, {m['dose'][:-3]}"
                b.hard.add("dose_self_correction")
            if m["freq"] in ("SOS",):
                tpl = rng.choice(["{n} {d}, {f}{t}.", "Aur {n} {d} rakh lijiye, {f}{t}.", "{n} {d} — {f}{t}."])
                b.say(D, join(tpl.format(n=name, d=dsy, f=fz, t=t)), [tg])
            elif rng.random() < 0.35 and dur:
                b.say(D, join(rng.choice(["{n} {d} likh deta hoon.", "Pehli dawai {n} {d} hai."]).format(n=name, d=dsy)), [tg])
                b.say(P, rng.choice(["Kitni baar leni hai?", "Kab kab leni hai?", "Kaise leni hai doctor?"]), raw=True)
                b.say(D, join(f"{fz}, {dur}{t}."), [tg])
                b.hard.add("split_prescription")
            else:
                tpl = rng.choice(["{n} {d} likh raha hoon, {f}, {u}{t}.", "Main aapko {n} {d} de raha hoon. {f}, {u}{t}.",
                                  "{n} {d}, {f}{t}, {u}.", "Ek {n} {d} likh deta hoon — {f}, {u}{t}."])
                b.say(D, join(tpl.format(n=name, d=dsy, f=fz, u=dur, t=t)), [tg])
            if dur and style != "terse" and rng.random() < 0.25:
                f2 = rng.choice(FREQ[m["freq"]])
                b.say(P, join(f"Toh {name}, {f2}, {dur}, theek?"), [tg])
                b.say(D, rng.choice(["Haan, bilkul sahi.", "Ji, sahi hai.", "Haan haan, wahi."]), [tg])
                b.hard.add("patient_recap")
        if rng.random() < 0.5:
            b.say(P, rng.choice(ACKS), raw=True)

    for m in c["existing_meds"]:
        if m["status"] == "changed":
            prescribe(m, exist_tags[id(m)])
        else:
            prescribe(m, exist_tags[id(m)])
    for m in c["new_meds"]:
        tg = b.add("medications", dict(name=m["name"], generic=m["generic"], dose=m["dose"], frequency=FREQ_NORM[m["freq"]],
                                       duration=dur_norm(m["duration"]) if m["duration"] else None,
                                       instructions=TIMING_NORM[m["timing"]], status="new"))
        prescribe(m, tg)
    if any(m["name"] in ("Dolo", "Crocin") for m in c["new_meds"] + c["existing_meds"] + ([sm] if sm else [])):
        b.hard.add("brand_name")

    if not inv_first:
        do_inv()

    # --- advice
    advs = list(c["advice"])
    ask = rng.random() < 0.5
    if ask:
        b.say(P, rng.choice(["Khane peene mein kuch parhez?", "Kuch dhyan rakhna hai kya doctor?", "Aur kuch karna hai ghar pe?"]))
    i = 0
    while i < len(advs):
        k = 2 if (i + 1 < len(advs) and rng.random() < 0.3) else 1
        grp = advs[i:i + k]
        i += k
        tags = [b.add("advice", dict(text=a)) for a, _ in grp]
        b.say(D, " ".join(rng.choice(v) for _, v in grp), tags)

    # --- follow up
    fu = c["follow_up"]
    if fu:
        dtxt = fmt_dur(fu["n"], fu["unit"], "hi" if rng.random() < 0.8 else "en")
        tg = b.add("follow_up", dict(text=f"after {dur_norm((fu['n'], fu['unit']))}",
                                      condition="if not better" if fu["cond"] else None))
        if fu["cond"]:
            b.say(D, rng.choice(["Agar {d} mein aaram na ho toh dobara aaiye.", "{d} mein theek na lage toh wapas dikhaiye."]).format(d=dtxt), [tg])
        else:
            b.say(D, rng.choice(["{d} baad dobara dikha dijiye.", "{d} baad follow-up ke liye aaiye."]).format(d=dtxt), [tg])
        b.say(P, rng.choice(["Ji, theek hai.", "Okay doctor.", "Ji zaroor."]), raw=True)

    b.say(P, rng.choice(CLOSE_PAT))
    b.say(D, rng.choice(CLOSE_DOC))
    return b, c, style


# ----------------------------------------------------------------------------
# Post-processing: disfluencies, backchannels, interruptions, timing
# ----------------------------------------------------------------------------
PROTECT = {"mg", "mcg", "ml", "din", "hafta", "hafte", "mahina", "mahine", "percent", "degree", "baar", "per", "puffs",
           "puff", "saal", "sachet", "sprays", "by", "days", "day", "weeks", "week", "months", "month", "fahrenheit",
           "celsius", "tak", "se", "ke", "ya", "—", "...", "nahi", "nahin", "hai", "hain", "hui", "tha", "thi", "na"}
FILL_P = dict(calm=0.25, talkative=0.4, terse=0.1, anxious=0.4)
FILL_N = dict(calm=[0, 0, 1], talkative=[0, 1, 2], terse=[0, 0, 0, 1], anxious=[1, 1, 2])
REPEATABLE = {"mujhe", "main", "wo", "woh", "aap", "aapko", "toh", "matlab", "actually", "haan", "ji", "ye", "yeh", "kal",
              "mere", "meri", "maine", "kya", "doctor"}


def has_digit(s):
    return any(ch.isdigit() for ch in s)


def disfluent(text, spk, style, rng, allow_mid=True):
    scale = 1.0 if spk == "patient" else 0.5
    lead = PAT_LEAD if spk == "patient" else DOC_LEAD
    mid = PAT_MID if spk == "patient" else DOC_MID
    toks = text.split(" ")
    # mid-sentence fillers
    nf = rng.choice(FILL_N[style]) if (allow_mid and rng.random() < scale) else 0
    valid = [i for i in range(1, len(toks) - 1)
             if not has_digit(toks[i]) and not has_digit(toks[i + 1])
             and toks[i + 1].strip(".,?").lower() not in PROTECT and toks[i].strip(".,?").lower() not in PROTECT
             and "..." not in toks[i] and "—" not in toks[i] and not toks[i].endswith(("?", "."))]
    pos = sorted(rng.sample(valid, min(nf, len(valid))), reverse=True)
    for i in pos:
        prev = toks[i]
        if not prev.endswith(","):
            toks[i] = prev + ","
        toks.insert(i + 1, rng.choice(mid) + ",")
    s = " ".join(toks)
    # repetition of a function word at the start
    first = s.split(" ")[0].strip(",.").lower()
    if first in REPEATABLE and rng.random() < 0.25 * scale * (2 if style in ("anxious", "talkative") else 1):
        w = s.split(" ")[0].rstrip(",")
        s = w + ", " + s[0].lower() + s[1:]
    # false start (patient only)
    elif spk == "patient" and rng.random() < 0.08:
        w = s.split(" ")[0]
        if w.isalpha() and len(w) >= 5 and w.lower() not in PROPER:
            s = f"{w[:2].lower()}- {w} " + " ".join(s.split(" ")[1:])
    # leading filler
    if rng.random() < FILL_P[style] * scale:
        f = rng.choice(lead)
        w0 = s.split(" ")[0]
        if w0.strip(",.?").lower() not in PROPER and not has_digit(w0) and not (w0.isupper() and len(w0) > 1):
            s = s[0].lower() + s[1:]
        s = f"{f}, {s}"
    if spk == "patient" and rng.random() < 0.08 * (2 if style == "anxious" else 1) and s.endswith("."):
        s = s[:-1] + "..."
    return cap(s)


def postprocess(b, rng):
    style = b.style
    turns = b.turns
    # 1) interruptions (doctor turn split at a comma, patient backchannel overlaps)
    made = 0
    i = 0
    out = []
    while i < len(turns):
        t = turns[i]
        if (t["spk"] == "doctor" and made < 2 and not t["raw"] and len(t["text"].split()) >= 9
                and "," in t["text"] and rng_ok(rng, 0.10)):
            parts = t["text"].split(", ")
            cut = None
            for j in range(1, len(parts)):
                a, rest = ", ".join(parts[:j]), ", ".join(parts[j:])
                if len(a.split()) >= 3 and len(rest.split()) >= 3:
                    cut = j
                    break
            if cut:
                a, rest = ", ".join(parts[:cut]), ", ".join(parts[cut:])
                t1 = dict(t, text=a + " —")
                bc = dict(spk="patient", text=rng.choice(["haan haan", "ji ji", "haan samajh gaya", "hmm haan"]), tags=[],
                          raw=True, overlap=True, distractor=None, cont=False)
                t2 = dict(t, text=rest, raw=True, cont=True)
                t2["text"] = rest[0].upper() + rest[1:] if rest[0].isalpha() else rest
                out += [t1, bc, t2]
                made += 1
                b.hard.add("interruption")
                i += 1
                continue
        out.append(t)
        i += 1
    turns[:] = out
    # 2) backchannels after long statements followed by same speaker
    out = []
    for i, t in enumerate(turns):
        out.append(t)
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        if (nxt and nxt["spk"] == t["spk"] and t["text"].endswith(".") and len(t["text"].split()) >= 9
                and not t["cont"] and rng.random() < 0.3):
            other = "patient" if t["spk"] == "doctor" else "doctor"
            out.append(dict(spk=other, text=rng.choice(PAT_BACK if other == "patient" else DOC_BACK), tags=[], raw=True,
                            overlap=False, distractor=None, cont=False))
    turns[:] = out
    # 3) disfluencies + gender inflection
    for t in turns:
        text = t["text"]
        if t["spk"] == "patient":
            text = gen(text, b.sex)
        if not t["raw"] and not t["cont"]:
            structured = any(k in ("medications", "vitals", "investigations") for k, _ in t["tags"])
            text = disfluent(text, t["spk"], style, rng, allow_mid=not structured)
        t["text"] = cap(text) if not t["cont"] else text
    if any("... nahi nahi," in t["text"] or "... nahi," in t["text"] for t in turns):
        b.hard.add("self_correction_present")


def rng_ok(rng, p):
    return rng.random() < p


HINDI_HINTS = {"hai", "hain", "nahi", "kya", "aap", "aapko", "mujhe", "se", "ke", "ki", "ka", "ko", "mein", "bhi", "toh", "haan",
               "ji", "hoon", "din", "baar", "subah", "shaam", "raat", "khana", "khane", "baad", "pehle", "dawai", "theek",
               "kijiye", "likh", "deta", "bataiye", "aaiye", "baithiye", "tak", "liye", "aur", "thoda", "zyada", "kam", "ab",
               "raha", "rahi", "hui", "hua", "li", "thi", "tha", "aaya", "aata", "lagta", "lag", "kar", "karna", "lijiye",
               "peeyen", "mat", "namaste", "dhanyavaad", "hafte", "hafta", "mahine", "mahina", "saal", "wahi", "bilkul"}


def seg_lang(text):
    toks = re.findall(r"[A-Za-z']+", text.lower())
    hits = sum(1 for t in toks if t in HINDI_HINTS)
    return "en" if hits == 0 else "hi-en"


def finalize(b, c, scn, idx, split, seed):
    rng = b.rng
    turns = b.turns
    # timing (estimated; replace with real timings after TTS/recording)
    segs = []
    t = round(rng.uniform(0.3, 1.0), 2)
    prev_end = 0.0
    for i, tr in enumerate(turns):
        tts = spoken(tr["text"])
        dur = max(0.5, len(tts.split()) / 2.4 + 0.15)
        start = max(0.0, prev_end - 0.25) if tr["overlap"] else t
        end = start + dur
        segs.append(dict(idx=i, start=round(start, 2), end=round(end, 2), speaker=tr["spk"], text=tr["text"],
                         text_tts=tts, lang=seg_lang(tr["text"]), overlap=tr["overlap"], confidence=1.0))
        prev_end = end
        t = end + rng.uniform(0.15, 0.7)
    # source segments + distractors
    distractors = []
    for i, tr in enumerate(turns):
        for kind, j in tr["tags"]:
            ss = b.ent[kind][j]["source_segments"]
            if i not in ss:
                ss.append(i)
        if tr["distractor"]:
            distractors.append(dict(segment=i, **tr["distractor"]))
    ent = {}
    for kind in ["demographics", "follow_up"]:
        ent[kind] = b.ent[kind][0] if b.ent[kind] else None
    for kind in ["symptoms", "diagnosis", "medications", "vitals", "investigations", "advice"]:
        ent[kind] = b.ent[kind]
    sid = f"S{idx + 1:04d}"
    n_pat_f = ["patient_f1", "patient_f2", "patient_f3"]
    n_pat_m = ["patient_m1", "patient_m2", "patient_m3"]
    meta = dict(scenario=scn, split=split, style=b.style, patient=dict(age=c["age"], sex=c["sex"]),
                hard_cases=sorted(b.hard), distractors=distractors,
                voice_hint=dict(doctor=rng.choice(["doctor_m1", "doctor_m2", "doctor_f1"]),
                                patient=rng.choice(n_pat_f if c["sex"] == "female" else n_pat_m)),
                timing_source="estimated", generator_seed=seed)
    return dict(session_id=sid,
                metadata=meta,
                transcript=dict(session_id=sid, language_mix="hi-en", segments=segs),
                extraction=dict(session_id=sid, entities=ent))


# ----------------------------------------------------------------------------
# Validation (grounding rule: key values must literally appear in cited segments)
# ----------------------------------------------------------------------------
def validate(d):
    issues = []
    segs = d["transcript"]["segments"]
    E = d["extraction"]["entities"]

    def txt(ss):
        return " ".join(segs[i]["text"] for i in ss).lower()

    def need(ss, digits, what):
        t = txt(ss)
        for x in digits:
            if x not in t:
                issues.append(f"{d['session_id']}: {what} value '{x}' not in segments {ss}")

    for it in E["symptoms"]:
        if not it["source_segments"]:
            issues.append(f"{d['session_id']}: symptom without source")
        if it.get("duration"):
            need(it["source_segments"], re.findall(r"\d+", it["duration"]), "symptom duration")
    for it in E["medications"]:
        ss = it["source_segments"]
        if not ss:
            issues.append(f"{d['session_id']}: med without source")
            continue
        if it["name"].split()[0].lower() not in txt(ss):
            issues.append(f"{d['session_id']}: med name {it['name']} not in segments")
        for key in ("dose", "duration", "previous_dose"):
            if it.get(key):
                need(ss, re.findall(r"\d+", it[key]), "med " + key)
    for it in E["vitals"]:
        digs = [str(it["value"].split("/")[0]), it["value"].split("/")[1]] if it["type"] == "blood_pressure" else [fnum(it["value"])]
        need(it["source_segments"], digs, "vital " + it["type"])
    for it in E["investigations"]:
        if it["text"].split()[0].lower() not in txt(it["source_segments"]):
            issues.append(f"{d['session_id']}: investigation {it['text']} not in segments")
        if it.get("result"):
            need(it["source_segments"], [re.match(r"[\d.]+", it["result"]).group()], "investigation result")
    for it in E["advice"] + E["diagnosis"]:
        if not it["source_segments"]:
            issues.append(f"{d['session_id']}: entity without source: {it}")
    if E["follow_up"]:
        need(E["follow_up"]["source_segments"], re.findall(r"\d+", E["follow_up"]["text"]), "follow_up")
    for s in segs:
        if "{" in s["text"] or "}" in s["text"] or "  " in s["text"]:
            issues.append(f"{d['session_id']}: bad text '{s['text']}'")
    return issues


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="out")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dialogues, issues = [], []
    idx = 0
    n = a.per_scenario
    n_val = max(1, round(n * 0.17))
    n_test = max(1, round(n * 0.17))
    for scn in SCENARIOS:
        for k in range(n):
            split = "test" if k >= n - n_test else ("val" if k >= n - n_test - n_val else "train")
            b, c, style = build_dialogue(scn, idx, a.seed, n, k)
            postprocess(b, b.rng)
            d = finalize(b, c, scn, idx, split, a.seed)
            issues += validate(d)
            dialogues.append(d)
            idx += 1
    meta = dict(name="Synthetic Hinglish OPD dialogues (Dataset B replacement)", version="1.0", n_dialogues=len(dialogues),
                scenarios=list(SCENARIOS), seed=a.seed,
                notes=["Template-generated; clinical content is NOT clinician-reviewed.",
                       "Segment timings are estimated; overwrite after TTS or recording.",
                       "text = gold transcript (digits kept); text_tts = TTS-ready (numbers and abbreviations spelled out).",
                       "entities follow Interface 2, with extensions: demographics, advice, follow_up, status, source, generic."])
    (out / "dataset_b_synthetic_opd.json").write_text(json.dumps(dict(meta=meta, dialogues=dialogues), ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "dataset_b_segments.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "split", "scenario", "seg_idx", "speaker", "voice_hint", "start", "end", "overlap", "lang", "text", "text_tts"])
        for d in dialogues:
            m = d["metadata"]
            for s in d["transcript"]["segments"]:
                w.writerow([d["session_id"], m["split"], m["scenario"], s["idx"], s["speaker"], m["voice_hint"][s["speaker"]],
                            s["start"], s["end"], s["overlap"], s["lang"], s["text"], s["text_tts"]])
    segs = sum(len(d["transcript"]["segments"]) for d in dialogues)
    hc = Counter(h for d in dialogues for h in d["metadata"]["hard_cases"])
    print(f"dialogues={len(dialogues)} segments={segs} validation_issues={len(issues)}")
    print("splits:", Counter(d["metadata"]["split"] for d in dialogues))
    print("hard cases:", dict(hc))
    for x in issues[:30]:
        print("ISSUE", x)


if __name__ == "__main__":
    main()
