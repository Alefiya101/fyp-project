# -*- coding: utf-8 -*-
"""baseline_eval.py

Zero-shot WER/CER evaluation of a faster-whisper model against the
cleaned dataset parquet produced by run_pipeline.py.

Usage
-----
    python -m speech_service.asr.baseline_eval \
        --data-dir cs_data \
        --model-size base \
        --output baseline_wer.csv

Output CSV columns: clip_id, lang_tag, reference, hypothesis, wer
Summary CSV columns: model, lang_tag, mean_wer, mean_cer, clip_count
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_for_wer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein / len(ref))."""
    if not ref:
        return float("nan")
    # Simple DP
    r, h = list(ref), list(hyp)
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[n] / m


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word error rate using jiwer, or NaN if either string is empty."""
    if not reference.strip() or not hypothesis.strip():
        return float("nan")
    try:
        import jiwer  # noqa: PLC0415
        r = _normalise_for_wer(reference)
        h = _normalise_for_wer(hypothesis)
        if not r:
            return float("nan")
        return float(jiwer.wer(r, h))
    except Exception as exc:
        logger.warning("WER computation failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
):
    """Load a faster-whisper WhisperModel on CPU."""
    from faster_whisper import WhisperModel  # noqa: PLC0415
    logger.info("Loading faster-whisper model '%s' on %s (%s)…", model_size, device, compute_type)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_clip(model, audio_path: str) -> str:
    """Transcribe a single WAV clip; return the concatenated text.

    Loads audio manually via soundfile to avoid the 'metadata_errors'
    incompatibility between some soundfile versions and faster-whisper's
    internal audio loader.
    """
    try:
        import soundfile as sf  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # Resample to 16kHz if needed
        if sr != 16_000:
            from math import gcd  # noqa: PLC0415
            from scipy.signal import resample_poly  # noqa: PLC0415
            g = gcd(sr, 16_000)
            wav = resample_poly(wav, 16_000 // g, sr // g).astype(np.float32)
        segments, _ = model.transcribe(wav, beam_size=1, language=None)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as exc:
        logger.warning("Transcription failed for '%s': %s", audio_path, exc)
        return ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    parquet_path: str,
    model_size: str = "base",
    output_csv: str = "baseline_wer.csv",
) -> float:
    """Evaluate a faster-whisper model on a parquet dataset.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file produced by run_pipeline.py.
        Expected columns: ``clip_id``, ``audio_path``, ``reference_text``,
        optionally ``lang_tag``.
    model_size : str
        faster-whisper model size (e.g. ``"base"``, ``"medium"``).
    output_csv : str
        Path for the per-clip results CSV.

    Returns
    -------
    float
        Median WER across all clips (NaN-excluded).  Returns NaN if no
        valid WER could be computed.
    """
    df = pd.read_parquet(parquet_path)
    required = {"clip_id", "audio_path", "reference_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet is missing columns: {missing}")

    if "lang_tag" not in df.columns:
        df["lang_tag"] = "unknown"

    model = load_model(model_size)

    rows: list[dict] = []
    for _, clip in df.iterrows():
        audio_path = str(clip["audio_path"])
        reference = str(clip.get("reference_text") or "")
        lang_tag = str(clip.get("lang_tag", "unknown"))

        hypothesis = transcribe_clip(model, audio_path)
        wer = compute_wer(reference, hypothesis)
        cer = _cer(
            _normalise_for_wer(reference),
            _normalise_for_wer(hypothesis),
        ) if reference.strip() and hypothesis.strip() else float("nan")

        rows.append({
            "clip_id": clip["clip_id"],
            "lang_tag": lang_tag,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer,
            "cer": cer,
        })

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["clip_id", "lang_tag", "reference", "hypothesis", "wer", "cer"])
        writer.writeheader()
        writer.writerows(rows)

    wers = [r["wer"] for r in rows if not math.isnan(r["wer"])]
    median_wer = float(np.median(wers)) if wers else float("nan")

    # Write summary table
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    _write_summary(rows, model_size, summary_path)

    logger.info(
        "Evaluation complete. Clips=%d  Median WER=%.3f  CSV=%s",
        len(rows), median_wer, out_path,
    )
    return median_wer


def _write_summary(rows: list[dict], model_size: str, path: Path) -> None:
    from collections import defaultdict  # noqa: PLC0415

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r["lang_tag"]].append(r)
    groups["all"] = rows

    summary_rows = []
    for lang_tag, group in groups.items():
        wers = [r["wer"] for r in group if not math.isnan(r["wer"])]
        cers = [r["cer"] for r in group if not math.isnan(r["cer"])]
        summary_rows.append({
            "model": model_size,
            "lang_tag": lang_tag,
            "mean_wer": float(np.mean(wers)) if wers else float("nan"),
            "mean_cer": float(np.mean(cers)) if cers else float("nan"),
            "clip_count": len(group),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "lang_tag", "mean_wer", "mean_cer", "clip_count"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Summary written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline ASR evaluation (faster-whisper, CPU)")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing clean_dataset.parquet")
    p.add_argument("--model-size", default="base",
                   help="faster-whisper model size (default: base)")
    p.add_argument("--output", default="baseline_wer.csv",
                   help="Output CSV path (default: baseline_wer.csv)")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    parquet = str(Path(args.data_dir) / "clean_dataset.parquet")
    median = evaluate_dataset(parquet, args.model_size, args.output)
    print(f"Median WER: {median:.4f}")
