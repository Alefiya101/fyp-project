"""Command-Line Interface for Person B: Clinical NLP & FHIR Service.

Supports:
1. Processing single transcript file -> producing Validated FHIR R4 Bundle.
2. Batch processing directory of transcripts.
3. System evaluation on simulated dialogue benchmark (Entity F1, Med Accuracy, Schema Validity %).
4. Exporting frozen JSON Schema contracts for Interfaces 1, 2, and 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure package directory is in sys.path
_PKG_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PKG_DIR.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from clinical_service.schemas import export_json_schemas
from clinical_service.schemas.interface1_transcript import Transcript
from clinical_service.service import ClinicalService


def evaluate_benchmark(benchmark_path: Path, max_samples: int = 50) -> None:
    """Evaluates extraction accuracy, medication metrics, and schema validity %."""
    if not benchmark_path.is_file():
        sys.exit(f"[ERROR] Benchmark file not found: {benchmark_path}")

    data: list[dict[str, Any]] = json.loads(benchmark_path.read_text(encoding="utf-8"))[:max_samples]
    print("\n=======================================================")
    print(f"  RUNNING BENCHMARK EVALUATION ({len(data)} Simulated Dialogues)")
    print("=======================================================\n")

    service = ClinicalService()

    total_gold_symptoms = 0
    total_pred_symptoms = 0
    correct_symptoms = 0

    total_gold_meds = 0
    total_pred_meds = 0
    correct_med_names = 0
    correct_med_doses = 0

    total_gold_vitals = 0
    total_pred_vitals = 0
    correct_vitals = 0

    valid_fhir_count = 0
    total_grounding_flags = 0

    for idx, item in enumerate(data):
        transcript = Transcript(
            session_id=item["session_id"],
            language_mix=item.get("language_mix", "hi-en"),
            segments=item["segments"],
        )
        gold = item.get("gold_extraction", {})

        result = service.process_transcript(transcript)
        pred_entities = service.extract_entities_only(transcript).entities

        # 1. Symptoms
        gold_syms = {s["text"].lower() for s in gold.get("symptoms", []) if not s.get("negated", False)}
        pred_syms = {s.text.lower() for s in pred_entities.symptoms if not s.negated}
        total_gold_symptoms += len(gold_syms)
        total_pred_symptoms += len(pred_syms)
        correct_symptoms += len(gold_syms.intersection(pred_syms))

        # 2. Medications
        gold_med_list = gold.get("medications", [])
        pred_med_list = pred_entities.medications
        total_gold_meds += len(gold_med_list)
        total_pred_meds += len(pred_med_list)

        for gm in gold_med_list:
            gname = gm["name"].lower()
            matching_pred = [pm for pm in pred_med_list if gname in pm.name.lower() or pm.name.lower() in gname]
            if matching_pred:
                correct_med_names += 1
                pm = matching_pred[0]
                if gm.get("dose") and pm.dose and gm["dose"].replace(" ", "") in pm.dose.replace(" ", ""):
                    correct_med_doses += 1

        # 3. Vitals
        gold_vit = {f"{v['type']}_{v['value']}" for v in gold.get("vitals", [])}
        pred_vit = {f"{v.type}_{v.value}" for v in pred_entities.vitals}
        total_gold_vitals += len(gold_vit)
        total_pred_vitals += len(pred_vit)
        correct_vitals += len(gold_vit.intersection(pred_vit))

        # 4. FHIR & Grounding
        if result.validation.schema_valid:
            valid_fhir_count += 1
        total_grounding_flags += len(result.grounding_flags)

    # Compute Metrics
    sym_prec = correct_symptoms / total_pred_symptoms if total_pred_symptoms else 1.0
    sym_rec = correct_symptoms / total_gold_symptoms if total_gold_symptoms else 1.0
    sym_f1 = (2 * sym_prec * sym_rec / (sym_prec + sym_rec)) if (sym_prec + sym_rec) else 0.0

    med_prec = correct_med_names / total_pred_meds if total_pred_meds else 1.0
    med_rec = correct_med_names / total_gold_meds if total_gold_meds else 1.0
    med_f1 = (2 * med_prec * med_rec / (med_prec + med_rec)) if (med_prec + med_rec) else 0.0
    dose_acc = correct_med_doses / total_gold_meds if total_gold_meds else 1.0

    schema_valid_pct = (valid_fhir_count / len(data)) * 100

    print("--- CLINICAL ENTITY EXTRACTION RESULTS ---")
    print(f"Symptom Extraction Precision: {sym_prec:.2%}")
    print(f"Symptom Extraction Recall   : {sym_rec:.2%}")
    print(f"Symptom Extraction F1-Score : {sym_f1:.2%}")
    print(f"Medication Name Recall      : {med_rec:.2%}")
    print(f"Medication Name Precision   : {med_prec:.2%}")
    print(f"Medication Name F1-Score    : {med_f1:.2%}")
    print(f"Medication Dose Accuracy    : {dose_acc:.2%}")
    print(f"Vital Signs Precision/Recall: {correct_vitals}/{total_gold_vitals}")
    print("\n--- FHIR R4 & ABDM CONFORMANCE ---")
    print(f"FHIR R4 Schema Validity     : {schema_valid_pct:.1f}% ({valid_fhir_count}/{len(data)})")
    print(f"Total Grounding Flags Raised: {total_grounding_flags}")
    print("=======================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical NLP & FHIR Service (Person B)")
    parser.add_argument("--input", "-i", type=str, help="Path to input Interface 1 Transcript JSON")
    parser.add_argument("--output", "-o", type=str, help="Path to write output Interface 3 Validated Bundle JSON")
    parser.add_argument("--export-schemas", type=str, help="Export JSON Schemas to given folder")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark evaluation on simulated dialogues")
    parser.add_argument("--benchmark-file", type=str, default="clinical_service/data/simulated_dialogues.json", help="Path to benchmark JSON")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to evaluate")
    parser.add_argument("--strict-grounding", action="store_true", help="Enable strict grounding checks")

    args = parser.parse_args()

    if args.export_schemas:
        export_json_schemas(args.export_schemas)
        print(f"[INFO] Exported JSON Schemas to {args.export_schemas}")
        return

    if args.evaluate:
        evaluate_benchmark(Path(args.benchmark_file), max_samples=args.max_samples)
        return

    if not args.input:
        parser.print_help()
        sys.exit(0)

    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"[ERROR] Input file not found: {in_path}")

    service = ClinicalService(strict_grounding=args.strict_grounding)
    result = service.process_transcript(in_path)

    out_json = result.model_dump_json(indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json, encoding="utf-8")
        print(f"[INFO] Successfully written Validated Bundle to {out_path}")
        print(f"       Schema Valid : {result.validation.schema_valid}")
        print(f"       Grounding Flags: {len(result.grounding_flags)}")
    else:
        print(out_json)


if __name__ == "__main__":
    main()
