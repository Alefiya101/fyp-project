# -*- coding: utf-8 -*-
"""run_pipeline.py

Cleaning / profiling pipeline for the agarwalayushi/hinglish speech dataset.
Designed for a CPU-only Windows (or Linux/macOS) machine.

Dataset schema (agarwalayushi/hinglish):
    source   string  speaker / source identifier
    text     string  transcript, <hi-en> tags mark code-switched utterances
    audio    Audio   embedded WAV bytes + sample rate
    quality  string  sample rate (Hz) or MOS score
    duration string  clip duration in seconds

There is no language column and no corpus-name column. Language is inferred
from tags and script inside `text`.

Steps per clip
    1. Read text first (cheap) and apply optional code-switch-only filter.
    2. Decode audio bytes (version-independent, via soundfile), mono, 16 kHz.
    3. Quality gate: duration, silence, clipping, estimated SNR.
    4. Optional speaker embedding (pyannote/embedding, needs a Hugging Face token).
    5. Optional Whisper ASR with medical-lexicon prompt biasing + WER vs reference.
    6. Text normalisation and metrics (script ratios, code-mix index, speech rate).
    7. Training-balance signal (oversample_weight, aug_recommended).
    8. Save 16 kHz WAVs, chunked parquet parts, final CSV/parquet, drop log, summary.

Resumable: progress is stored in output/progress.json (use --resume).

Examples
    python run_pipeline.py --max-clips 200 --no-embeddings
    python run_pipeline.py --max-clips 5000 --cs-only
    python run_pipeline.py --resume --max-clips 5000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import regex as re
import soundfile as sf
import tqdm
from scipy.signal import resample_poly

try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _transliterate
    TRANSLIT_AVAILABLE = True
except ImportError:
    TRANSLIT_AVAILABLE = False

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False


# ---------------------------------------------------------------------------
# CONFIGURATION (defaults; most can be overridden on the command line)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    dataset_id: str = "agarwalayushi/hinglish"
    split: str = "train"
    local_parquet: Optional[str] = None      # for testing with a local file/glob
    lexicon_path: str = "data/medical_lexicon.txt"
    script_policy: str = "mixed"             # "mixed" (keep as is) or "roman"
    spell_numbers: bool = True

    min_duration_sec: float = 1.0
    max_duration_sec: float = 30.0
    min_snr_db: float = 5.0
    max_clip_fraction: float = 0.001         # >0.1% of samples at full scale = clipped
    cs_code_mix_threshold: float = 0.25

    max_clips: Optional[int] = 2000          # stop after this many KEPT clips (None = all)
    cs_only: bool = False                    # keep only clips with a <hi-en> tag
    shuffle: bool = True
    shuffle_buffer: int = 1000               # ~300 MB RAM; raise only if you have memory
    seed: int = 42

    run_asr: bool = True
    whisper_model: str = "base"
    run_embeddings: bool = True
    save_audio: bool = True
    part_size: int = 500                     # rows per parquet part
    resume: bool = False

    output_dir: Path = Path("output")
    log_dir: Path = Path("logs")


# ---------------------------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"</?[A-Za-z]+(?:-[A-Za-z]+)?>")
CS_TAG_RE = re.compile(r"<\s*hi-en\s*>", re.IGNORECASE)
DEVA_RUN_RE = re.compile(r"[\p{Devanagari}\p{M}]+")
DEVA_ANY_RE = re.compile(r"\p{Devanagari}")
WORD_RE = re.compile(r"[\p{L}\p{M}\p{N}]+")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
WS_RE = re.compile(r"\s+")


def load_lexicon(path: str) -> List[str]:
    if not os.path.isfile(path):
        print(f"[WARN] Lexicon not found at '{path}'. Hotword biasing disabled.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        terms = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    print(f"[INFO] Loaded {len(terms)} lexicon terms from '{path}'.")
    return terms


def _spell_number(match: "re.Match") -> str:
    s = match.group()
    if len(s) > 12:                       # phone numbers, IDs: leave untouched
        return s
    try:
        value = float(s) if "." in s else int(s)
        return " " + num2words(value, lang="en").replace("-", " ") + " "
    except Exception:
        return s


def to_roman(text: str) -> str:
    """Transliterate Devanagari runs to Roman (ITRANS). Lossy: use for matching
    and analysis, not as natural Hinglish spelling."""
    if not TRANSLIT_AVAILABLE:
        return text
    return DEVA_RUN_RE.sub(
        lambda m: _transliterate(m.group(), sanscript.DEVANAGARI, sanscript.ITRANS),
        text,
    )


def base_text(raw: str) -> str:
    """NFC + remove language tags + collapse whitespace. Scripts untouched."""
    text = unicodedata.normalize("NFC", raw or "")
    text = TAG_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def normalize_text(raw: str, cfg: Config) -> str:
    text = base_text(raw)
    if cfg.spell_numbers and NUM2WORDS_AVAILABLE:
        text = NUM_RE.sub(_spell_number, text)
    if cfg.script_policy == "roman":
        text = to_roman(text)
    return WS_RE.sub(" ", text).strip()


def script_stats(text: str) -> Tuple[int, int]:
    deva = latin = 0
    for w in WORD_RE.findall(text):
        if DEVA_ANY_RE.search(w):
            deva += 1
        elif LATIN_WORD_RE.fullmatch(w):
            latin += 1
    return deva, latin


def text_metrics(text: str) -> Dict[str, float]:
    """Script ratios and code-mix index on text with scripts intact.

    code_mix_index = share of Latin-script words among Hindi+English words.
    It is only defined when the text contains Devanagari; for fully
    Roman-script text a script-based index cannot separate Hindi from English,
    so it is NaN and the <hi-en> tag is the signal used instead.
    """
    deva, latin = script_stats(text)
    total = deva + latin
    deva_ratio = deva / total if total else 0.0
    roman_ratio = latin / total if total else 0.0
    code_mix = latin / total if deva > 0 else float("nan")
    return {"devanagari_ratio": deva_ratio, "roman_ratio": roman_ratio,
            "code_mix_index": code_mix}


def speech_rate(text: str, duration: float) -> float:
    chars = len(text.replace(" ", ""))
    return chars / duration if duration > 0 else 0.0


def wer_norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", text)
    return WS_RE.sub(" ", text).strip()


def compute_wer(ref: str, hyp: str) -> float:
    """Indicative WER between reference and Whisper output. NaN when it is not
    meaningful (jiwer missing, empty reference, or different scripts)."""
    if not JIWER_AVAILABLE:
        return float("nan")
    r, h = wer_norm(ref), wer_norm(hyp)
    if not r:
        return float("nan")
    if bool(DEVA_ANY_RE.search(r)) != bool(DEVA_ANY_RE.search(h)):
        return float("nan")
    return float(jiwer.wer(r, h))


# ---------------------------------------------------------------------------
# AUDIO UTILITIES
# ---------------------------------------------------------------------------
def decode_audio(field: Any) -> Tuple[np.ndarray, int]:
    """Decode a Hugging Face Audio(decode=False) value into mono float32.
    Independent of the `datasets` decoding backend version."""
    if not isinstance(field, dict):
        raise ValueError(f"unsupported audio field type: {type(field).__name__}")
    if field.get("bytes"):
        data, sr = sf.read(io.BytesIO(field["bytes"]), dtype="float32")
    elif field.get("array") is not None:
        data, sr = np.asarray(field["array"], dtype=np.float32), int(field["sampling_rate"])
    elif field.get("path") and os.path.isfile(field["path"]):
        data, sr = sf.read(field["path"], dtype="float32")
    else:
        raise ValueError("audio field has no bytes, array or readable path")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def resample_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wav
    g = gcd(sr, 16000)
    return resample_poly(wav, 16000 // g, sr // g).astype(np.float32)


def clip_fraction(wav: np.ndarray, threshold: float = 0.999) -> float:
    return float(np.mean(np.abs(wav) >= threshold)) if wav.size else 0.0


def estimate_snr(wav: np.ndarray, frame: int = 400, hop: int = 160) -> float:
    """Rough SNR: energy of loud frames (90th pct) over quiet frames (10th pct).
    A dynamic-range proxy, capped at 60 dB. Not a calibrated SNR."""
    n = 1 + (len(wav) - frame) // hop
    if n < 5:
        return 0.0
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    energy = np.mean(wav[idx].astype(np.float64) ** 2, axis=1)
    sig, noise = np.percentile(energy, 90), np.percentile(energy, 10)
    if sig <= 1e-12:
        return 0.0
    return float(min(10 * math.log10(sig / max(noise, 1e-12)), 60.0))


def quality_gate(duration: float, snr_db: float, clip_frac: float,
                 cfg: Config) -> Tuple[bool, str]:
    if duration < cfg.min_duration_sec:
        return False, f"too_short ({duration:.2f}s)"
    if duration > cfg.max_duration_sec:
        return False, f"too_long ({duration:.1f}s)"
    if clip_frac > cfg.max_clip_fraction:
        return False, f"clipping ({clip_frac:.4f})"
    if snr_db < cfg.min_snr_db:
        return False, f"low_snr ({snr_db:.1f} dB)"
    return True, ""


def balance_signal(code_mix: float, has_cs_tag: bool, cfg: Config) -> Tuple[int, bool]:
    """(oversample_weight, aug_recommended). Code-switched / spontaneous clips
    are upsampled; clean read-speech clips are flagged for augmentation."""
    spontaneous = has_cs_tag or (not math.isnan(code_mix) and code_mix > cfg.cs_code_mix_threshold)
    return (3, False) if spontaneous else (1, True)


# ---------------------------------------------------------------------------
# MODELS (optional, loaded only when enabled)
# ---------------------------------------------------------------------------
class WhisperAsr:
    def __init__(self, model_name: str, lexicon: List[str], max_prompt_tokens: int = 200):
        import whisper
        print(f"[INFO] Loading Whisper-{model_name} on CPU...")
        self.model = whisper.load_model(model_name, device="cpu")
        self.prompt = self._build_prompt(whisper, lexicon, max_prompt_tokens)
        if self.prompt:
            print(f"[INFO] Whisper prompt active ({len(self.prompt.split())} words).")
        else:
            print("[WARN] No Whisper prompt (empty lexicon): medical terms may be mis-transcribed.")

    def _build_prompt(self, whisper_mod, lexicon: List[str], max_tokens: int) -> str:
        if not lexicon:
            return ""
        text = ", ".join(lexicon)
        try:
            tok = whisper_mod.tokenizer.get_tokenizer(self.model.is_multilingual)
            ids = tok.encode(" " + text)[:max_tokens]
            return tok.decode(ids).strip()
        except Exception:
            return " ".join(text.split()[:max_tokens // 2])

    def transcribe(self, wav_16k: np.ndarray) -> str:
        result = self.model.transcribe(
            wav_16k, language=None, fp16=False, temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.prompt or None, verbose=None,
        )
        return result["text"].strip()


class SpeakerEmbedder:
    def __init__(self):
        import torch
        from pyannote.audio import Inference, Model
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        print("[INFO] Loading pyannote/embedding...")
        try:
            model = Model.from_pretrained("pyannote/embedding", token=token)
        except TypeError:                     # older pyannote.audio versions
            model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
        if model is None:
            raise RuntimeError("model could not be loaded (token missing or terms not accepted)")
        self.torch = torch
        self.inference = Inference(model, window="whole")

    def embed(self, wav_16k: np.ndarray) -> List[float]:
        tensor = self.torch.from_numpy(wav_16k).unsqueeze(0)
        out = self.inference({"waveform": tensor, "sample_rate": 16000})
        return [float(x) for x in np.asarray(out).flatten()]


# ---------------------------------------------------------------------------
# DATASET ACCESS
# ---------------------------------------------------------------------------
def open_stream(cfg: Config, skip: int):
    from datasets import Audio, load_dataset
    if cfg.local_parquet:
        ds = load_dataset("parquet", data_files=cfg.local_parquet, split="train", streaming=True)
    else:
        ds = load_dataset(cfg.dataset_id, split=cfg.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    if cfg.shuffle:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    if skip:
        ds = ds.skip(skip)
    return ds


# ---------------------------------------------------------------------------
# OUTPUT HELPERS
# ---------------------------------------------------------------------------
def list_parts(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("part_*.parquet"))


def flush_part(rows: List[dict], out_dir: Path) -> None:
    if not rows:
        return
    part_no = len(list_parts(out_dir)) + 1
    pd.DataFrame(rows).to_parquet(out_dir / f"part_{part_no:05d}.parquet", index=False)
    rows.clear()


def save_progress(out_dir: Path, next_idx: int) -> None:
    (out_dir / "progress.json").write_text(json.dumps({"next_idx": next_idx}))


def load_progress(out_dir: Path) -> int:
    p = out_dir / "progress.json"
    return int(json.loads(p.read_text()).get("next_idx", 0)) if p.is_file() else 0


class DropLog:
    def __init__(self, path: Path, append: bool):
        new = not (append and path.is_file())
        self.fh = open(path, "w" if new else "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["idx", "source", "reason"])

    def add(self, idx: int, source: str, reason: str) -> None:
        self.w.writerow([idx, source, reason])

    def flush(self) -> None:
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(cfg: Config) -> None:
    out_dir, log_dir = cfg.output_dir, cfg.log_dir
    audio_dir = out_dir / "audio"
    for d in (out_dir, log_dir) + ((audio_dir,) if cfg.save_audio else ()):
        d.mkdir(parents=True, exist_ok=True)

    if list_parts(out_dir) and not cfg.resume:
        sys.exit(f"[ERROR] '{out_dir}' already contains results. Use --resume to continue, "
                 "or pick another --output-dir.")
    if cfg.script_policy == "roman" and not TRANSLIT_AVAILABLE:
        print("[WARN] script_policy='roman' needs 'indic-transliteration'; text left unchanged.")
    if cfg.spell_numbers and not NUM2WORDS_AVAILABLE:
        print("[WARN] num2words not installed; digits kept as-is.")

    start_idx = load_progress(out_dir) if cfg.resume else 0
    already_kept = sum(len(pd.read_parquet(p, columns=["clip_id"])) for p in list_parts(out_dir))
    print(f"[INFO] Starting at stream index {start_idx}; {already_kept} clips already kept.")

    # Models
    asr = None
    if cfg.run_asr:
        asr = WhisperAsr(cfg.whisper_model, load_lexicon(cfg.lexicon_path))
    embedder = None
    if cfg.run_embeddings:
        try:
            embedder = SpeakerEmbedder()
        except Exception as exc:
            print(f"[WARN] Speaker embeddings disabled: {exc}\n"
                  "       Needs: accept terms at hf.co/pyannote/embedding and set HF_TOKEN.")

    ds = open_stream(cfg, start_idx)
    drops = DropLog(log_dir / "dropped_clips.csv", append=cfg.resume)
    reasons: Counter = Counter()
    rows: List[dict] = []
    kept = already_kept
    consumed = start_idx
    total_target = cfg.max_clips

    pbar = tqdm.tqdm(total=total_target, initial=kept, desc="Kept clips", unit="clip")
    try:
        for row in ds:
            idx = consumed
            consumed += 1
            if total_target is not None and kept >= total_target:
                consumed -= 1
                break
            source = str(row.get("source") or "")

            # 1. text first (cheap)
            raw_text = row.get("text") or ""
            has_cs_tag = bool(CS_TAG_RE.search(raw_text))
            if cfg.cs_only and not has_cs_tag:
                reasons["not_code_switched"] += 1
                continue

            # 2. audio
            try:
                wav, sr = decode_audio(row.get("audio"))
                if not np.all(np.isfinite(wav)):
                    raise ValueError("non-finite samples")
            except Exception as exc:
                reasons["decode_error"] += 1
                drops.add(idx, source, f"decode_error: {exc}")
                continue

            clip_frac = clip_fraction(wav)           # measured before resampling
            wav16 = resample_16k(wav, sr)
            duration = len(wav16) / 16000.0
            snr_db = estimate_snr(wav16)

            # 3. quality gate
            ok, why = quality_gate(duration, snr_db, clip_frac, cfg)
            if not ok:
                reasons[why.split(" ")[0]] += 1
                drops.add(idx, source, why)
                continue

            # 4. text metrics (scripts intact) and normalised reference
            clean = base_text(raw_text)
            tm = text_metrics(clean)
            reference = normalize_text(raw_text, cfg)
            cps = speech_rate(reference, duration)
            suspect = (not reference) or not (3.0 <= cps <= 30.0)

            deva_ratio_present = tm["devanagari_ratio"] > 0
            if has_cs_tag:
                lang_tag = "hi-en"
            elif deva_ratio_present:
                lang_tag = "hi"
            else:
                lang_tag = "en-IN"      # Roman-only text: Hindi-in-Roman cannot be told apart

            weight, aug = balance_signal(tm["code_mix_index"], has_cs_tag, cfg)

            # 5. optional models
            asr_text, asr_wer = "", float("nan")
            if asr is not None:
                try:
                    asr_text = normalize_text(asr.transcribe(wav16), cfg)
                    asr_wer = compute_wer(reference, asr_text)
                except Exception as exc:
                    print(f"[WARN] Whisper failed on clip {idx}: {exc}")
            embedding = None
            if embedder is not None:
                try:
                    embedding = embedder.embed(wav16)
                except Exception as exc:
                    print(f"[WARN] Embedding failed on clip {idx}: {exc}")

            # 6. save audio + row
            clip_id = f"clip_{idx:09d}"
            audio_path = ""
            if cfg.save_audio:
                audio_path = str((audio_dir / f"{clip_id}.wav").as_posix())
                sf.write(audio_path, wav16, 16000, subtype="PCM_16")

            record = {
                "clip_id": clip_id,
                "audio_path": audio_path,
                "source": source,
                "quality": str(row.get("quality") or ""),
                "lang_tag": lang_tag,
                "has_cs_tag": has_cs_tag,
                "duration": duration,
                "snr_db": snr_db,
                "clip_fraction": clip_frac,
                "speech_rate_cps": cps,
                "suspect_transcript": bool(suspect),
                "devanagari_ratio": tm["devanagari_ratio"],
                "roman_ratio": tm["roman_ratio"],
                "code_mix_index": tm["code_mix_index"],
                "oversample_weight": weight,
                "aug_recommended": aug,
                "reference_text": reference,
                "asr_transcript": asr_text,
                "asr_wer": asr_wer,
            }
            if embedder is not None:
                record["speaker_embedding"] = embedding
            rows.append(record)
            kept += 1
            pbar.update(1)

            if len(rows) >= cfg.part_size:
                flush_part(rows, out_dir)
                save_progress(out_dir, consumed)
                drops.flush()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted: saving what has been processed so far.")
    finally:
        pbar.close()
        flush_part(rows, out_dir)
        save_progress(out_dir, consumed)
        drops.close()

    # Final merge
    parts = list_parts(out_dir)
    if not parts:
        print("[WARN] No clips were kept. See logs/dropped_clips.csv.")
        return
    final_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    final_df.to_parquet(out_dir / "clean_dataset.parquet", index=False)
    csv_df = final_df.drop(columns=[c for c in ["speaker_embedding"] if c in final_df.columns])
    csv_df.to_csv(out_dir / "clean_dataset.csv", index=False, encoding="utf-8-sig")

    # Summary
    print("\n================ SUMMARY ================")
    print(f"Kept clips        : {len(final_df)}  ({final_df['duration'].sum() / 3600:.2f} h)")
    print(f"Drop reasons      : {dict(reasons)}")
    print(f"Language tags     : {final_df['lang_tag'].value_counts().to_dict()}")
    print(f"Oversample weights: {final_df['oversample_weight'].value_counts().to_dict()}")
    print(f"Suspect transcripts: {int(final_df['suspect_transcript'].sum())}")
    if asr is not None:
        print(f"Median ASR WER    : {final_df['asr_wer'].median():.3f}  (indicative, same-script clips only)")
    print(f"CSV     -> {out_dir / 'clean_dataset.csv'}")
    print(f"Parquet -> {out_dir / 'clean_dataset.parquet'}")
    print(f"Drops   -> {log_dir / 'dropped_clips.csv'}")


def parse_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="Hinglish dataset cleaning pipeline")
    d = Config()
    p.add_argument("--max-clips", type=int, default=d.max_clips,
                   help="stop after this many kept clips (0 = no limit)")
    p.add_argument("--cs-only", action="store_true", help="keep only <hi-en> tagged clips")
    p.add_argument("--script-policy", choices=["mixed", "roman"], default=d.script_policy)
    p.add_argument("--keep-digits", action="store_true", help="do not spell digits as words")
    p.add_argument("--no-asr", action="store_true", help="skip Whisper")
    p.add_argument("--no-embeddings", action="store_true", help="skip speaker embeddings")
    p.add_argument("--no-save-audio", action="store_true", help="do not write 16 kHz WAV files")
    p.add_argument("--whisper-model", default=d.whisper_model,
                   help="tiny, base, small ... (base is the CPU-friendly default)")
    p.add_argument("--lexicon", default=d.lexicon_path)
    p.add_argument("--local-parquet", default=None, help="use a local parquet file/glob instead of the Hub")
    p.add_argument("--shuffle-buffer", type=int, default=d.shuffle_buffer)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--part-size", type=int, default=d.part_size)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default=str(d.output_dir))
    p.add_argument("--log-dir", default=str(d.log_dir))
    a = p.parse_args(argv)
    return Config(
        local_parquet=a.local_parquet, lexicon_path=a.lexicon, script_policy=a.script_policy,
        spell_numbers=not a.keep_digits, max_clips=(a.max_clips or None), cs_only=a.cs_only, shuffle=not a.no_shuffle,
        shuffle_buffer=a.shuffle_buffer, run_asr=not a.no_asr, whisper_model=a.whisper_model,
        run_embeddings=not a.no_embeddings, save_audio=not a.no_save_audio,
        part_size=a.part_size, resume=a.resume,
        output_dir=Path(a.output_dir), log_dir=Path(a.log_dir),
    )


if __name__ == "__main__":
    run(parse_args())
