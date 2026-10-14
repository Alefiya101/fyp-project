"""ABDM Profile Constants and Helper Definitions.

Defines URIs, profile identifiers, and section codes for ABDM OPConsultRecord.
Reference: https://nrces.in/ndhm/fhir/r4/StructureDefinition/OPConsultRecord
"""

# ABDM StructureDefinition URLs
ABDM_OP_CONSULT_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/OPConsultRecord"
ABDM_PATIENT_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient"
ABDM_PRACTITIONER_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Practitioner"
ABDM_ENCOUNTER_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Encounter"
ABDM_CONDITION_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Condition"
ABDM_OBSERVATION_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation"
ABDM_MEDICATION_REQUEST_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/MedicationRequest"
ABDM_SERVICE_REQUEST_PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/ServiceRequest"

# Standard Code Systems
SYSTEM_SNOMED = "http://snomed.info/sct"
SYSTEM_LOINC = "http://loinc.org"
SYSTEM_RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
SYSTEM_UCUM = "http://unitsofmeasure.org"
SYSTEM_CONDITION_CLINICAL = "http://terminology.hl7.org/CodeSystem/condition-clinical"
SYSTEM_CONDITION_VER_STATUS = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
SYSTEM_CONDITION_CATEGORY = "http://terminology.hl7.org/CodeSystem/condition-category"
SYSTEM_OBSERVATION_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
SYSTEM_ACT_CODE = "http://terminology.hl7.org/CodeSystem/v3-ActCode"

# OPConsultRecord Document Composition Code
OP_CONSULT_TYPE_CODE = "371530004"
OP_CONSULT_TYPE_DISPLAY = "Clinical consultation report"

# Section Codes (SNOMED CT)
SECTION_CHIEF_COMPLAINTS_CODE = "422843007"
SECTION_CHIEF_COMPLAINTS_TITLE = "Chief Complaints"

SECTION_DIAGNOSIS_CODE = "4241000179101"
SECTION_DIAGNOSIS_TITLE = "Provisional / Final Diagnosis"

SECTION_VITALS_CODE = "1184593002"
SECTION_VITALS_TITLE = "Physical Examination / Vitals"

SECTION_MEDICATIONS_CODE = "721912009"
SECTION_MEDICATIONS_TITLE = "Medications Prescribed"

SECTION_INVESTIGATIONS_CODE = "721963009"
SECTION_INVESTIGATIONS_TITLE = "Investigations Advised"
