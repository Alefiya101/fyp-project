# Ayushman Bharat Digital Mission (ABDM) FHIR R4 Profile Study: OPConsultRecord

## 1. Overview
The **National Health Authority (NHA)** specifies FHIR Release 4 (R4) implementation guides for the Ayushman Bharat Digital Mission (ABDM).
The **OPConsultRecord** (Outpatient Consultation Record) is defined as a FHIR **`Bundle`** of type **`document`**.

### Document Bundle Structure
```
Bundle (resourceType: "Bundle", type: "document")
├── entry[0]: Composition (resourceType: "Composition", type: "OPConsultRecord")
│    ├── subject -> Reference to Patient
│    ├── author -> Reference to Practitioner
│    ├── encounter -> Reference to Encounter
│    └── section[] (Chief Complaints, Physical Examination, Medical History, Diagnosis, Medications, Investigations)
├── entry[1]: Patient
├── entry[2]: Practitioner
├── entry[3]: Encounter
├── entry[4...N]: Condition (Symptoms & Diagnoses)
├── entry[...]: Observation (Vitals with LOINC)
├── entry[...]: MedicationRequest (Prescribed Drugs)
└── entry[...]: ServiceRequest / DiagnosticReport (Investigations)
```

---

## 2. Resource Mapping Table

| Interface 2 Field | Target FHIR R4 Resource | ABDM Profile / System | Mandatory / Core Fields |
|:---|:---|:---|:---|
| **Root Header** | `Composition` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/OPConsultRecord` | `status="final"`, `type` (SNOMED: `371530004` Clinical consultation report), `subject`, `date`, `author`, `title="Outpatient Consultation Note"`, `section[]` |
| **Session Metadata** | `Patient` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient` | `id`, `name`, `identifier` (ABHA address / ID) |
| **Doctor Role** | `Practitioner` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/Practitioner` | `id`, `name`, `identifier` (HPR ID or NMC registration) |
| **Encounter** | `Encounter` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/Encounter` | `status="finished"`, `class` (AMB - ambulatory), `subject` |
| **`symptoms`** | `Condition` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/Condition` | `clinicalStatus="active"`, `verificationStatus` ("confirmed" or "refuted" if negated), `category` (symptom: SNOMED `418799008`), `code` (SNOMED CT), `subject` |
| **`diagnosis`** | `Condition` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/Condition` | `clinicalStatus="active"`, `verificationStatus` ("provisional" / "confirmed"), `category` (encounter-diagnosis: SNOMED `42493007`), `code` (SNOMED CT), `subject` |
| **`vitals`** | `Observation` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation` | `status="final"`, `category` (vital-signs), `code` (LOINC), `valueQuantity` (value, unit, UCUM system), `subject` |
| **`medications`** | `MedicationRequest` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/MedicationRequest` | `status="active"`, `intent="order"`, `medicationCodeableConcept` (RxNorm/SNOMED or text), `dosageInstruction` (timing, route, doseAndRate), `subject` |
| **`investigations`** | `ServiceRequest` | `https://nrces.in/ndhm/fhir/r4/StructureDefinition/ServiceRequest` | `status="active"`, `intent="order"`, `code` (LOINC / SNOMED test name), `subject` |

---

## 3. Required Standard Terminology Coding Systems
1. **SNOMED CT**: `http://snomed.info/sct` — used for symptoms, clinical findings, diagnoses, and procedure/test requests.
2. **LOINC**: `http://loinc.org` — used for vital sign observations and diagnostic lab observation codes.
3. **UCUM**: `http://unitsofmeasure.org` — standard clinical units for quantities (e.g., `Cel`, `degF`, `mm[Hg]`, `/min`, `%`).
4. **RxNorm**: `http://www.nlm.nih.gov/research/umls/rxnorm` — standard medication identifiers.
