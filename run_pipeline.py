# -*- coding: utf-8 -*-
"""run_pipeline.py

Concrete end‑to‑end pipeline for processing a large Hugging Face audio dataset on a CPU‑only machine.

Steps performed:
1. Stream the metadata table from the Hub (no audio download).
2. Compute per‑clip quality features (duration, SNR, clipping, speech‑rate, script‑ratio, code‑mixing index).
3. Extract speaker embeddings via a pretrained ECAPA‑x‑vector model (pyannote). Cached as .npy.
4. Resample each audio file to 16 kHz mono and compute the Whisper log‑Mel spectrogram (80 bins).
5. Run Whisper‑base (CPU) to obtain a transcript.
6. Clean the transcript (Unicode NFC, number normalization, script policy, tag handling).
7. Assemble a final pandas DataFrame with selected columns and write to CSV/Parquet.

The script is written to be robust on limited RAM: audio is processed one clip at a time and intermediate results are cached on disk.

Replace the placeholder constants (DATASET_ID, SPLIT_NAME, LEXICON_PATH, SCRIPT_POLICY) with your actual values before running.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import tqdm

# Audio / signal processing
import librosa
import soundfile as sf

# Whisper (CPU‑only) and feature extraction
import whisper

# Speaker embedding (ECAPA‑x‑vector) – pyannote works on CPU but is slower; we load the pretrained model once.
from pyannote.audio import Inference

# Regular expressions for text cleaning
import regex as re
import unicodedata

# ---------------------------------------------------------------------------
# ------------------------------ CONFIGURATION ------------------------------
# ---------------------------------------------------------------------------
# <-- USER SHOULD EDIT THESE VALUES ------------------------------------------------
DATASET_ID = "<owner>/<repo>"           # e.g. "myorg/audio-corpus"
SPLIT_NAME = "train"                    # name of the split that contains the metadata table
LEXICON_PATH = "path/to/medical_lexicon.txt"   # one term per line, optional – used for hot‑word prepending
SCRIPT_POLICY = "roman"                 # "roman", "devanagari" or "mixed"
# ---------------------------------------------------------------------------

# Directory layout (all under the current working directory)
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

for d in [DATA_DIR, OUTPUT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# --------------------------- UTILITY FUNCTIONS ----------------------------
# ---------------------------------------------------------------------------

def load_lexicon(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

MEDICAL_LEXICON = load_lexicon(LEXICON_PATH)

def normalize_text(text: str) -> str:
    """Unicode NFC + number normalization + script handling."""
    # NFC
    text = unicodedata.normalize("NFC", text)
    # Simple number normalization: replace Arabic numerals with words (placeholder)
    # You can plug in a more sophisticated library if needed.
    text = re.sub(r"\d+", lambda m: num2words(int(m.group())) if "num2words" in globals() else m.group(), text)
    # Script policy handling
    if SCRIPT_POLICY == "roman":
        # transliterate Devanagari to Roman (basic approach – replace each char with a placeholder)
        # for production use `indic_transliteration` or similar library.
        text = re.sub(r"\p{Devanagari}+", "", text)  # naive removal – keep Roman only
    elif SCRIPT_POLICY == "devanagari":
        text = re.sub(r"[\p{Latin}]+", "", text)
    # Strip optional language tags
    text = re.sub(r"<[/]?hi-en>", "", text)
    return text.strip()

def compute_duration(file_path: str, sr: int = 16000) -> float:
    info = sf.info(file_path)
    return info.duration

def estimate_snr(audio: np.ndarray, sr: int) -> float:
    # Very rough SNR: ratio of signal RMS to noise RMS estimated from non‑speech frames.
    # Here we use a simple energy threshold; for a better estimate use a VAD.
    rms = np.sqrt(np.mean(audio ** 2))
    # Assume the lowest 5% of absolute amplitudes as noise floor
    noise_floor = np.percentile(np.abs(audio), 5)
    snr = 20 * np.log10(rms / (noise_floor + 1e-9))
    return snr

def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    return np.any(np.abs(audio) >= threshold)

def speech_rate(text: str, duration: float) -> float:
    # characters per second (excluding spaces)
    chars = len(text.replace(" ", ""))
    return chars / duration if duration > 0 else 0.0

def script_ratio(text: str) -> Tuple[float, float]:
    devanagari = len(re.findall(r"\p{Devanagari}+", text))
    roman = len(re.findall(r"[A-Za-z]+", text))
    total = devanagari + roman
    if total == 0:
        return 0.0, 0.0
    return devanagari / total, roman / total

def code_mixing_index(text: str) -> float:
    # proportion of English words in a Hinglish utterance
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0
    eng = sum(1 for w in words if re.fullmatch(r"[A-Za-z]+", w))
    return eng / len(words)

# ---------------------------------------------------------------------------
# --------------------------- MAIN PIPELINE --------------------------------
# ---------------------------------------------------------------------------

def main():
    from datasets import load_dataset, Audio

    # -------------------------------------------------------------------
    # 1. Stream metadata (no audio download yet)
    # -------------------------------------------------------------------
    print("Loading metadata from HuggingFace…")
    ds = load_dataset(
        DATASET_ID,
        split=SPLIT_NAME,
        streaming=True,
        # The dataset may already contain an "audio" column with file paths.
    )

    # Convert the streaming dataset to a DataFrame (metadata fits in RAM).
    rows = []
    for ex in tqdm.tqdm(ds, desc="Reading metadata", unit="row"):
        rows.append(ex)
    meta_df = pd.DataFrame(rows)
    print(f"Metadata contains {len(meta_df)} rows, columns: {list(meta_df.columns)}")

    # -------------------------------------------------------------------
    # 2. Prepare columns we need – adjust to actual column names later.
    # -------------------------------------------------------------------
    required_cols = [
        "audio_path",    # relative path on HF hub (or full URL)
        "speaker_id",
        "source_corpus",
        "language",
        "transcript",    # if any, may be empty
    ]
    # Keep only existing columns
    existing = [c for c in required_cols if c in meta_df.columns]
    meta_df = meta_df[existing]

    # -------------------------------------------------------------------
    # 3. Initialise heavy models once
    # -------------------------------------------------------------------
    print("Loading Whisper model (base) for CPU inference…")
    whisper_model = whisper.load_model("base", device="cpu")

    print("Loading ECAPA‑x‑vector speaker encoder (pyannote)…")
    speaker_encoder = Inference("pyannote/embedding", window="whole")  # whole‑clip embedding

    # -------------------------------------------------------------------
    # 4. Process each clip – streaming to keep RAM low
    # -------------------------------------------------------------------
    processed_rows = []
    for idx, row in tqdm.tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing clips"):
        audio_hub_path = row["audio_path"]
        # ----------------------------------------------------------------
        # Download the audio file on‑the‑fly (datasets >= 2.18 supports .download_and_extract())
        # ----------------------------------------------------------------
        audio = Audio()
        # `audio.decode_example` returns a dict with 'array' and 'sampling_rate'
        audio_data = audio.decode_example({"path": audio_hub_path})
        wav = audio_data["array"]
        sr = audio_data["sampling_rate"]

        # ----------------------------------------------------------------
        # a) Resample to 16 kHz mono
        # ----------------------------------------------------------------
        wav_resampled = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        wav_resampled = wav_resampled.astype(np.float32)

        # ----------------------------------------------------------------
        # b) Compute quality features
        # ----------------------------------------------------------------
        duration_sec = len(wav_resampled) / 16000
        snr_db = estimate_snr(wav_resampled, 16000)
        clipping = detect_clipping(wav_resampled)

        # ----------------------------------------------------------------
        # c) Speaker embedding (cached on disk to avoid recompute)
        # ----------------------------------------------------------------
        embed_path = CACHE_DIR / f"embed_{idx}.npy"
        if embed_path.is_file():
            speaker_emb = np.load(embed_path)
        else:
            speaker_emb = speaker_encoder(wav_resampled)
            np.save(embed_path, speaker_emb)

        # ----------------------------------------------------------------
        # d) Whisper log‑Mel (optional: pre‑compute and cache – here we just run inference)
        # ----------------------------------------------------------------
        # Whisper's `transcribe` does the feature extraction internally.
        # To keep everything CPU‑friendly we use the built‑in method.
        result = whisper_model.transcribe(wav_resampled, language="hi", fp16=False)
        transcript_raw = result["text"].strip()

        # ----------------------------------------------------------------
        # e) Text cleaning according to the chosen script policy
        # ----------------------------------------------------------------
        transcript_clean = normalize_text(transcript_raw)

        # ----------------------------------------------------------------
        # f) Additional text‑side metrics
        # ----------------------------------------------------------------
        sp_rate = speech_rate(transcript_clean, duration_sec)
        devanagari_ratio, roman_ratio = script_ratio(transcript_clean)
        code_mix_idx = code_mixing_index(transcript_clean)

        # ----------------------------------------------------------------
        # g) Assemble a single row for the final table
        # ----------------------------------------------------------------
        final_row = {
            "audio_path": audio_hub_path,
            "speaker_id": row.get("speaker_id"),
            "source_corpus": row.get("source_corpus"),
            "language": row.get("language"),
            "duration": duration_sec,
            "snr_db": snr_db,
            "clipping": clipping,
            "speech_rate_cps": sp_rate,
            "devanagari_ratio": devanagari_ratio,
            "roman_ratio": roman_ratio,
            "code_mix_index": code_mix_idx,
            "transcript": transcript_clean,
            # Store speaker embedding as a flattened list for CSV compatibility
            **{f"emb_{i}": float(v) for i, v in enumerate(speaker_emb.squeeze())},
        }
        processed_rows.append(final_row)

    # -------------------------------------------------------------------
    # 5. Build final DataFrame & persist
    # -------------------------------------------------------------------
    final_df = pd.DataFrame(processed_rows)
    csv_out = OUTPUT_DIR / "clean_dataset.csv"
    final_df.to_csv(csv_out, index=False)
    print(f"Saved cleaned dataset with {len(final_df)} rows to {csv_out}")

    # Optional: also write a Parquet version for faster downstream loading
    parquet_out = OUTPUT_DIR / "clean_dataset.parquet"
    final_df.to_parquet(parquet_out, index=False)
    print(f"Parquet version written to {parquet_out}")

if __name__ == "__main__":
    main()
