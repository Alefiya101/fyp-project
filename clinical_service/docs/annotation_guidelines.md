# Clinical NLP & Annotation Guidelines: Edge-AI Ambient Clinical Scribe

## 1. Overview & Purpose
This document specifies the annotation schema, linguistic conventions, and clinical entity rules for converting **Hinglish (Hindi-English code-switched) Indian OPD clinical dialogues** into structured clinical extractions (**Interface 2 JSON**).

These rules ensure consistent evaluation, precise prompt-tuning for 1–3B parameter Small Language Models (SLMs), and auditable grounding against transcript lines.

---

## 2. Entity Types & Field Definitions

### 2.1 Symptoms (`symptoms`)
Complaints, symptoms, or physical discomfort reported by the patient or inquired by the physician.
* **`text`** *(string, required)*: Standardized or primary clinical symptom name (e.g., `fever`, `cough`, `headache`, `chest pain`, `nausea`, `loose motion`, `saans phoolna`).
* **`duration`** *(string, optional)*: Explicit temporal duration reported (e.g., `3 days`, `kal raat se`, `2 weeks`).
* **`severity`** *(string, optional)*: Qualitative modifier (e.g., `mild`, `moderate`, `severe`, `tez`).
* **`negated`** *(boolean, required)*: `true` if the symptom was denied or ruled out; `false` otherwise.
* **`source_segments`** *(list of integers, required)*: Zero- or one-based indices of the transcript segments containing the statement.

### 2.2 Diagnoses (`diagnosis`)
Provisional impression, differential diagnosis, or confirmed clinical condition established by the doctor.
* **`text`** *(string, required)*: Diagnosis name (e.g., `Acute Upper Respiratory Infection`, `Viral Fever`, `Essential Hypertension`, `Type 2 Diabetes Mellitus`).
* **`confidence`** *(enum: `high` | `medium` | `low` | `provisional` | `differential`)*:
  * `high` / `confirmed`: Direct definitive assertion ("Aapko viral fever hai").
  * `medium` / `provisional`: Probable working diagnosis ("Lagta hai typhoid ho sakta hai").
  * `differential`: Alternative possibility under consideration.
* **`status`** *(enum: `active` | `provisional` | `resolved` | `recurrence`)*.
* **`source_segments`** *(list of integers, required)*.

### 2.3 Medications (`medications`)
Prescribed pharmaceuticals, injections, or patient home medications.
* **`name`** *(string, required)*: Generic or brand name (e.g., `Paracetamol`, `Dolo 650`, `Amoxicillin`, `Metformin`, `Amlodipine`).
* **`dose`** *(string, optional)*: Strength or dosage form (e.g., `500 mg`, `650 mg`, `1 tablet`, `5 ml syrup`, `2 puffs`).
* **`frequency`** *(string, optional)*: Administration schedule normalized or as stated (e.g., `OD` / once daily, `BD` / twice daily, `TDS` / thrice daily, `QID`, `SOS` / jab zaroorat ho).
* **`duration`** *(string, optional)*: Course length (e.g., `3 days`, `5 days`, `continuous`).
* **`status`** *(enum: `new` | `existing` | `discontinued` | `modified`)*:
  * `new`: Prescribed during this consultation.
  * `existing`: Ongoing medication already taken by the patient prior to this visit.
  * `discontinued`: Doctor advises stopping an existing drug.
  * `modified`: Dosage or frequency changed during consultation.
* **`instructions`** *(string, optional)*: Food/timing instructions (e.g., `after food`, `khana khane ke baad`, `empty stomach`, `subah khali pet`).
* **`source_segments`** *(list of integers, required)*.

### 2.4 Vitals (`vitals`)
Physical examination measurements recorded during the encounter.
* **`type`** *(string, required)*:
  * `temperature` (Body Temperature)
  * `blood_pressure_sys` / `blood_pressure_dia` (or combined `blood_pressure`)
  * `pulse` (Heart rate / pulse)
  * `spo2` (Oxygen saturation)
  * `respiratory_rate`
  * `weight`
* **`value`** *(number or string, required)*: Value recorded (e.g., `101.2`, `120/80`, `78`, `98`).
* **`unit`** *(string, required)*: Standard unit (`degF`, `Cel`, `mm[Hg]`, `/min`, `%`, `kg`).
* **`source_segments`** *(list of integers, required)*.

### 2.5 Investigations (`investigations`)
Laboratory or imaging tests ordered or discussed.
* **`name`** *(string, required)*: e.g., `Complete Blood Count (CBC)`, `Chest X-ray PA view`, `HbA1c`, `Serum Creatinine`, `Lipid Profile`.
* **`status`** *(enum: `ordered` | `reviewed` | `recommended` | `completed`)*.
* **`result`** *(string, optional)*: If test values were brought in and reviewed.
* **`source_segments`** *(list of integers, required)*.

---

## 3. Hinglish Specific Annotation Rules

### 3.1 Negation Detection Rules
Clinical dialogue in Indian OPDs frequently employs Hindi negation markers. Negated symptoms MUST have `negated: true`.
* **Negation Keywords**:
  * Hindi: *nahi*, *nhi*, *na*, *mat*, *koi nahi* (e.g., "khansi nahi hai", "dard bilkul nahi ho raha", "vomiting to nahi hui").
  * English: *no*, *denies*, *without*, *absent*, *not* (e.g., "no chest pain", "not feeling dizzy").
* **Rule**: When the patient or doctor confirms absence of a symptom, create the symptom entity with `negated: true`. Do NOT omit negated symptoms, as negative findings (pertinent negatives) are clinically critical in differential diagnosis!

### 3.2 Medication Temporal Status (New vs Existing)
Distinguishing active chronic medications from acute prescriptions is vital:
* **Existing Medication Clues**:
  * "Main pehle se le raha hoon" / "Pehle se chal rahi hai"
  * "Pichle do saal se BP ki goli le raha hoon"
  * "Regular medicine" / "daily le rahe ho?"
  * -> Annotate as `status: "existing"`.
* **New Prescription Clues**:
  * "Yeh medicine likh raha hoon" / "Aap yeh 3 din tak lijiye"
  * "Aaj se shuru karo" / "Subah sham lena"
  * -> Annotate as `status: "new"`.
* **Modified / Discontinued**:
  * "Ab dose badha dete hain" -> `status: "modified"`.
  * "Purani wali band kar do" -> `status: "discontinued"`.

### 3.3 Strict Grounding Rule (`source_segments`)
* Every entity MUST cite the exact line numbers (segment indices) where it is explicitly stated or directly inferred.
* If a dosage number (e.g., `650 mg`, `3 days`) does NOT appear in the cited segments or dialogue, the model MUST NOT fabricate it. Any ungrounded number triggers a Grounding Flag in Interface 3.
