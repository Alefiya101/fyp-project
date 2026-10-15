# -*- coding: utf-8 -*-
"""finetune.py

CPU-only fine-tuning of openai/whisper-tiny using HuggingFace Seq2SeqTrainer.
Compatible with transformers 5.x.  No LoRA, no GPU.

Audio loading and feature extraction happen entirely inside the data collator
so that HuggingFace Datasets never has to serialise variable-length waveforms
(which causes Arrow schema truncation issues).

Usage
-----
    python -m speech_service.asr.finetune \
        --stage 1 --data-dir cs_data \
        --model-out speech_service/models/stage1 \
        --epochs 3 --max-clips 200

    python -m speech_service.asr.finetune \
        --stage 2 --data-dir out_200 \
        --model-out speech_service/models/stage2 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STAGE1_CHECKPOINT = Path("speech_service/models/stage1")
_BASE_MODEL = "openai/whisper-tiny"
_MAX_SAMPLES = 30 * 16_000   # 30 s × 16 kHz
_N_MEL_FRAMES = 3000          # Whisper encoder fixed input length
_SR = 16_000


# ---------------------------------------------------------------------------
# Dataset — stores only paths and text, nothing audio-shaped
# ---------------------------------------------------------------------------

def load_dataset_from_parquet(parquet_path: str, max_clips: Optional[int] = None):
    """Return a HuggingFace Dataset with columns: audio_path, text."""
    from datasets import Dataset  # noqa: PLC0415

    df = pd.read_parquet(parquet_path)
    for col in ("audio_path", "reference_text"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Drop missing / nonexistent paths
    mask = df["audio_path"].isna() | (df["audio_path"].astype(str).str.strip() == "")
    df = df[~mask].reset_index(drop=True)
    mask2 = ~df["audio_path"].apply(lambda p: Path(p).exists())
    if mask2.sum():
        logger.warning("Dropping %d clips not found on disk.", int(mask2.sum()))
    df = df[~mask2].reset_index(drop=True)

    if max_clips:
        df = df.head(max_clips)

    logger.info("Dataset: %d clips.", len(df))
    return Dataset.from_pandas(
        df[["audio_path", "reference_text"]].rename(columns={"reference_text": "text"})
    )


# ---------------------------------------------------------------------------
# Audio loading helper
# ---------------------------------------------------------------------------

def _load_wav(path: str) -> np.ndarray:
    """Load a WAV file as mono float32 at 16 kHz, max 30 s."""
    import soundfile as sf  # noqa: PLC0415
    from scipy.signal import resample_poly  # noqa: PLC0415

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != _SR:
        g = gcd(sr, _SR)
        wav = resample_poly(wav, _SR // g, sr // g).astype(np.float32)
    return wav[:_MAX_SAMPLES]


# ---------------------------------------------------------------------------
# Data collator — does all audio work here, never touches the dataset columns
# ---------------------------------------------------------------------------

def _make_collator(processor):
    """Collator that loads audio and runs feature extraction at batch time."""

    def collate(samples: List[Dict]) -> Dict:
        waveforms, texts = [], []
        for s in samples:
            try:
                wav = _load_wav(s["audio_path"])
                waveforms.append(wav)
                texts.append(s["text"] or "")
            except Exception as exc:
                logger.warning("Collator: skipping '%s': %s", s.get("audio_path"), exc)

        if not waveforms:
            raise RuntimeError("Empty batch — all clips failed to load.")

        # Feature extraction → always (B, 80, 3000)
        # max_length is in SAMPLES (480000 = 30 s × 16 kHz).
        # The feature extractor converts samples → mel frames internally,
        # always producing exactly 3000 frames regardless of clip length.
        fe = processor.feature_extractor(
            waveforms,
            sampling_rate=_SR,
            return_tensors="pt",
            padding="max_length",
            max_length=_MAX_SAMPLES,   # 480000 samples, NOT 3000 frames
            truncation=True,
        )

        # Tokenise labels
        tok = processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448,
        )
        labels = tok["input_ids"].masked_fill(tok["attention_mask"].ne(1), -100)
        # Remove leading BOS if present
        if labels.shape[1] > 0 and (labels[:, 0] == processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        fe["labels"] = labels
        return fe

    return collate


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def build_training_args(output_dir: str, **overrides) -> Any:
    from transformers import Seq2SeqTrainingArguments  # noqa: PLC0415

    cfg: Dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=False,
        predict_with_generate=True,
        generation_max_length=448,
        fp16=False,
        bf16=False,
        use_cpu=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,   # collator handles column selection
    )
    cfg.update(overrides)
    return Seq2SeqTrainingArguments(**cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_finetuning(
    model_name_or_path: str,
    parquet_path: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    max_clips: Optional[int] = None,
) -> None:
    from transformers import (  # noqa: PLC0415
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainer,
    )

    logger.info("Loading model: %s", model_name_or_path)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    ds = load_dataset_from_parquet(parquet_path, max_clips=max_clips)

    training_args = build_training_args(
        output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=_make_collator(processor),
    )

    logger.info("Training started (CPU, %d clips, %d epochs)…", len(ds), num_train_epochs)
    try:
        trainer.train(resume_from_checkpoint=_find_latest_checkpoint(output_dir))
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Saved to %s", output_dir)


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CPU-only Whisper fine-tuning")
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-clips", type=int, default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if args.stage == 1:
        model_src = _BASE_MODEL
    else:
        if not _STAGE1_CHECKPOINT.exists():
            sys.exit(f"[ERROR] Stage 1 checkpoint not found at '{_STAGE1_CHECKPOINT}'.")
        model_src = str(_STAGE1_CHECKPOINT)

    run_finetuning(
        model_name_or_path=model_src,
        parquet_path=str(Path(args.data_dir) / "clean_dataset.parquet"),
        output_dir=args.model_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_clips=args.max_clips,
    )
