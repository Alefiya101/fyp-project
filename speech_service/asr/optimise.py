# -*- coding: utf-8 -*-
"""Load, optionally export, and benchmark Hugging Face Whisper checkpoints.

Usage
-----
    # Verify/download the shared stage-2 model for local use
    python -m speech_service.asr.optimise \
        --checkpoint-dir alefiya101/whisper-hinglish-clinical-stage2 \
        --output-dir speech_service/models/stage2

    # Benchmark the checkpoint on CPU
    python -m speech_service.asr.optimise \
        --checkpoint-dir alefiya101/whisper-hinglish-clinical-stage2 \
        --data-dir cs_data \
        --benchmark-clips 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "alefiya101/whisper-hinglish-clinical-stage2"


def load_transformers_model(model_name_or_path: str, device: str = "cpu"):
    """Load a Whisper checkpoint and processor from disk or the Hugging Face Hub."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    model.to(torch.device(device))
    model.eval()
    return model, processor


def prepare_transformers_model(
    checkpoint_dir: str,
    output_dir: str | None = None,
    force: bool = False,
) -> None:
    """Validate/load a checkpoint and optionally save a local HF-format copy."""
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None and destination.exists() and not force:
        raise FileExistsError(
            f"Output directory '{destination}' already exists. "
            "Pass --force to save/overwrite the checkpoint there."
        )

    model, processor = load_transformers_model(checkpoint_dir)
    if destination is None:
        logger.info("Loaded Transformers checkpoint '%s'.", checkpoint_dir)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination, safe_serialization=True)
        processor.save_pretrained(destination)
        logger.info("Saved Transformers model and processor to '%s'.", destination)


def _load_audio(path: str) -> tuple[np.ndarray, float]:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    wav, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != 16_000:
        divisor = gcd(sample_rate, 16_000)
        wav = resample_poly(
            wav, 16_000 // divisor, sample_rate // divisor
        ).astype(np.float32)
    return wav, len(wav) / 16_000


def benchmark_rtf(
    model_dir: str,
    data_dir: str,
    n_clips: int = 20,
    output_json: str = "speech_service/asr/rtf_benchmark.json",
) -> float:
    """Benchmark Transformers Whisper real-time factor on local audio clips."""
    import pandas as pd
    import torch

    model, processor = load_transformers_model(model_dir)
    parquet = Path(data_dir) / "clean_dataset.parquet"
    df = pd.read_parquet(parquet, columns=["audio_path"])
    df = df[
        df["audio_path"].apply(
            lambda path: isinstance(path, (str, Path)) and Path(path).is_file()
        )
    ]
    df = df.head(n_clips)

    if df.empty:
        raise ValueError(f"No valid audio clips found in '{parquet}'.")

    total_audio_s = 0.0
    total_proc_s = 0.0
    processed_clips = 0

    for audio_path in df["audio_path"]:
        started = time.perf_counter()
        try:
            wav, duration_s = _load_audio(str(audio_path))
            inputs = processor(wav, sampling_rate=16_000, return_tensors="pt")
            input_features = inputs.input_features.to(model.device)
            with torch.inference_mode():
                model.generate(input_features, task="transcribe", language=None)
            total_proc_s += time.perf_counter() - started
            total_audio_s += duration_s
            processed_clips += 1
        except Exception as exc:
            logger.warning("Transcription failed for '%s': %s", audio_path, exc)

    if total_audio_s <= 0:
        raise RuntimeError("No clips could be transcribed; cannot compute RTF.")

    rtf = total_proc_s / total_audio_s
    result = {
        "rtf": rtf,
        "processed_clips": processed_clips,
        "requested_clips": len(df),
        "total_audio_seconds": total_audio_s,
        "total_processing_seconds": total_proc_s,
        "model": model_dir,
        "runtime": "transformers",
        "device": "cpu",
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("RTF=%.3f (written to %s)", rtf, out)
    if rtf > 1.0:
        print(f"[WARN] RTF={rtf:.3f} > 1.0 — slower than real time on this CPU.")
    else:
        print(f"RTF={rtf:.3f} (faster than real time).")
    return rtf


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and benchmark a Hugging Face Transformers Whisper checkpoint"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_MODEL,
        help="Local HF checkpoint directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional local directory to save a copy in Transformers format",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow saving into an existing output directory",
    )
    parser.add_argument("--data-dir", default=None, help="Dataset directory for RTF benchmark")
    parser.add_argument("--benchmark-clips", type=int, default=20)
    parser.add_argument("--rtf-output", default="speech_service/asr/rtf_benchmark.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    prepare_transformers_model(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    if args.data_dir:
        benchmark_rtf(
            model_dir=args.output_dir or args.checkpoint_dir,
            data_dir=args.data_dir,
            n_clips=args.benchmark_clips,
            output_json=args.rtf_output,
        )
