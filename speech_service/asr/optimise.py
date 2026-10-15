# -*- coding: utf-8 -*-
"""optimise.py

Convert a fine-tuned Whisper HuggingFace checkpoint to CTranslate2 INT8
format and benchmark its Real-Time Factor (RTF) on CPU.

Usage
-----
    # Convert
    python -m speech_service.asr.optimise \
        --checkpoint-dir speech_service/models/stage2 \
        --output-dir speech_service/models/ct2_stage2

    # Convert + RTF benchmark
    python -m speech_service.asr.optimise \
        --checkpoint-dir speech_service/models/stage2 \
        --output-dir speech_service/models/ct2_stage2 \
        --data-dir cs_data \
        --benchmark-clips 20
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_ctranslate2(
    checkpoint_dir: str,
    output_dir: str,
    quantization: str = "int8",
    force: bool = False,
) -> None:
    """Convert a Whisper HF checkpoint to CTranslate2 format.

    Parameters
    ----------
    checkpoint_dir : str
        Path to a Whisper HuggingFace checkpoint directory
        (must contain ``config.json``).
    output_dir : str
        Destination directory for the CTranslate2 model.
    quantization : str
        Quantization type (default ``"int8"``).
    force : bool
        If ``False`` (default) and *output_dir* already exists, raise
        ``FileExistsError``.  If ``True``, overwrite silently.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_dir* does not exist.
    FileExistsError
        If *output_dir* already exists and *force* is ``False``.
    RuntimeError
        If the conversion command fails.
    """
    src = Path(checkpoint_dir)
    dst = Path(output_dir)

    if not src.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {src}")

    if dst.exists() and not force:
        raise FileExistsError(
            f"Output directory '{dst}' already exists. "
            "Pass force=True (or --force on the CLI) to overwrite."
        )

    # Try Python API first; fall back to CLI subprocess
    try:
        import ctranslate2  # noqa: PLC0415
        converter = ctranslate2.converters.OpusMTConverter(str(src))
        converter.convert(str(dst), quantization=quantization, force=force)
    except (ImportError, AttributeError):
        _convert_via_cli(str(src), str(dst), quantization, force)
    except Exception:
        # OpusMTConverter is for OpusMT; for Whisper use the CLI
        _convert_via_cli(str(src), str(dst), quantization, force)

    # Report
    if dst.exists():
        size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"CTranslate2 model written to: {dst}")
        print(f"Model size: {size_mb:.1f} MB")
    else:
        raise RuntimeError(f"Conversion completed but output directory '{dst}' was not created.")


def _convert_via_cli(src: str, dst: str, quantization: str, force: bool) -> None:
    """Run ``ct2-transformers-converter`` as a subprocess."""
    cmd = [
        sys.executable, "-m", "ctranslate2.bin.convert_transformers",
        "--model", src,
        "--output_dir", dst,
        "--quantization", quantization,
        "--model_spec", "WhisperConfig",
    ]
    if force:
        cmd.append("--force")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try the standard entry-point name
        cmd2 = ["ct2-transformers-converter",
                "--model", src,
                "--output_dir", dst,
                "--quantization", quantization]
        if force:
            cmd2.append("--force")
        logger.info("Retrying with entry-point: %s", " ".join(cmd2))
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            raise RuntimeError(
                f"Conversion failed (exit {result2.returncode}):\n{result2.stderr}"
            )


# ---------------------------------------------------------------------------
# RTF benchmark
# ---------------------------------------------------------------------------

def benchmark_rtf(
    model_dir: str,
    data_dir: str,
    n_clips: int = 20,
    output_json: str = "speech_service/asr/rtf_benchmark.json",
) -> float:
    """Benchmark Real-Time Factor of the CTranslate2 model on CPU.

    Parameters
    ----------
    model_dir : str
        Path to the CTranslate2 model directory.
    data_dir : str
        Directory containing ``clean_dataset.parquet`` and WAV files.
    n_clips : int
        Number of clips to benchmark (default 20).
    output_json : str
        Path to write the JSON results.

    Returns
    -------
    float
        RTF value (< 1.0 means faster-than-real-time).
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    model = WhisperModel(model_dir, device="cpu", compute_type="int8")

    parquet = Path(data_dir) / "clean_dataset.parquet"
    df = pd.read_parquet(parquet, columns=["audio_path", "duration"])
    df = df[df["audio_path"].apply(lambda p: Path(p).exists())]
    df = df.head(n_clips)

    if df.empty:
        logger.error("No valid audio clips found in '%s'.", data_dir)
        return float("nan")

    total_audio_s = 0.0
    total_proc_s = 0.0

    for _, row in df.iterrows():
        audio_duration = float(row.get("duration", 0.0))
        t0 = time.perf_counter()
        try:
            # Load audio manually to avoid soundfile metadata_errors incompatibility
            import soundfile as sf  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            from math import gcd  # noqa: PLC0415
            from scipy.signal import resample_poly  # noqa: PLC0415
            wav, sr = sf.read(str(row["audio_path"]), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16_000:
                g = gcd(sr, 16_000)
                wav = resample_poly(wav, 16_000 // g, sr // g).astype(np.float32)
            segs, _ = model.transcribe(wav, beam_size=1)
            list(segs)  # consume the generator
        except Exception as exc:
            logger.warning("Transcription failed for '%s': %s", row["audio_path"], exc)
            continue
        elapsed = time.perf_counter() - t0
        total_audio_s += audio_duration
        total_proc_s += elapsed

    if total_audio_s <= 0:
        logger.error("Total audio duration is zero; cannot compute RTF.")
        return float("nan")

    rtf = total_proc_s / total_audio_s
    result = {
        "rtf": rtf,
        "total_clips": len(df),
        "total_audio_seconds": total_audio_s,
        "total_processing_seconds": total_proc_s,
        "model_dir": model_dir,
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("RTF benchmark: %.3f  (written to %s)", rtf, out)

    if rtf > 1.0:
        print(
            f"[WARN] RTF={rtf:.3f} > 1.0 — model is slower than real time on this CPU. "
            "Consider a smaller model or further quantisation."
        )
    else:
        print(f"RTF={rtf:.3f} (faster than real time).")

    return rtf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Whisper checkpoint to CTranslate2 INT8 and benchmark RTF")
    p.add_argument("--checkpoint-dir", required=True,
                   help="HuggingFace Whisper checkpoint directory")
    p.add_argument("--output-dir", required=True,
                   help="CTranslate2 output directory")
    p.add_argument("--quantization", default="int8",
                   help="Quantization type (default: int8)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output directory if it exists")
    p.add_argument("--data-dir", default=None,
                   help="Dataset directory for RTF benchmark (optional)")
    p.add_argument("--benchmark-clips", type=int, default=20,
                   help="Number of clips for RTF benchmark (default: 20)")
    p.add_argument("--rtf-output", default="speech_service/asr/rtf_benchmark.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    convert_to_ctranslate2(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        quantization=args.quantization,
        force=args.force,
    )

    if args.data_dir:
        benchmark_rtf(
            model_dir=args.output_dir,
            data_dir=args.data_dir,
            n_clips=args.benchmark_clips,
            output_json=args.rtf_output,
        )
