# Run-Pipeline Guide (Rev 2 – post-reviewer update)

This document explains how to **execute the `run_pipeline.py` script**.  
It assumes a Windows machine with the workspace at `c:\FYP`.

---

## What changed in Rev 2

Three concrete gaps were identified against the reviewer's feedback:

| # | Gap | Fix |
|---|-----|-----|
| **1** | Clips under 1 s / over 30 s and low-SNR / clipped audio were computed but never filtered | Quality gate added in `passes_quality_gate()`. Dropped clips are logged to `logs/dropped_clips.csv`. |
| **2** | All corpora were weighted equally – studio TTS / NPTEL dominated and the model would degrade on noisy clinical audio | `oversample_weight` (1 or 3) and `aug_recommended` columns added. Training loop should use `WeightedRandomSampler` and apply noise+reverb+speed augmentation to `aug_recommended=True` clips. |
| **3** | Medical lexicon was loaded but **never used** | `initial_prompt` built from the lexicon is now passed to `whisper_model.transcribe()` for soft hotword biasing – no retraining required. `language=None` replaces hard-coded `"hi"` to auto-detect genuinely code-switched clips. |

---

## 1 — Prerequisites

| Requirement | How to install |
|-------------|----------------|
| **Python 3.11** | Official installer or Anaconda/Miniconda |
| **Conda** | Included with Anaconda/Miniconda |
| **Git** (optional) | `winget install --id Git.Git -e --source winget` |

---

## 2 — Create the isolated environment

```powershell
conda create -n audio_clean python=3.11 -y
conda activate audio_clean
```

---

## 3 — Install packages

```powershell
pip install `
    pandas pyarrow tqdm librosa==0.10.2 soundfile==0.12.1 `
    openai-whisper torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu `
    pyannote.audio==3.1.1 regex==2024.5.15 datasets
```

---

## 4 — Fill in the placeholders

Open `run_pipeline.py` and edit the block near the top:

```python
DATASET_ID    = "<owner>/<repo>"              # e.g. "myorg/audio-corpus"
SPLIT_NAME    = "train"
LEXICON_PATH  = "path/to/medical_lexicon.txt" # one medical term per line
SCRIPT_POLICY = "roman"                       # "roman" | "devanagari" | "mixed"

# FIX 1 thresholds (adjust if needed)
MIN_DURATION_SEC = 1.0    # drop clips shorter than this
MAX_DURATION_SEC = 30.0   # drop clips longer than this
MIN_SNR_DB       = 5.0    # drop clips with estimated SNR below this

# FIX 2 spontaneous corpora (add your corpus names)
SPONTANEOUS_CORPORA = {"mucs", "limmits", "kathbath_cs"}
```

### Medical lexicon format

One term per line, plain UTF-8. Example:

```
metformin
paracetamol
dyspnoea
hypertension
tachycardia
```

These are prepended as an `initial_prompt` so Whisper's decoder is biased toward them without any retraining.

---

## 5 — Run the pipeline

```powershell
python run_pipeline.py
```

The script will:

1. **Stream** metadata from HuggingFace Hub (no audio downloaded yet).
2. Iterate over every clip, downloading **one at a time**.
3. **Quality-gate** each clip (FIX 1) – drop if too short, too long, clipped or low-SNR.
4. Extract a **speaker embedding** (ECAPA-x-vector via pyannote), cached on disk.
5. **Resample** to 16 kHz mono; Whisper computes the 80-bin log-Mel internally.
6. Run **Whisper-base** (CPU) with medical-lexicon `initial_prompt` (FIX 3).
7. **Clean** the transcript (NFC, number norm, script policy, tag strip).
8. Compute text-side metrics and the **training-balance signal** (FIX 2).
9. Write:
   - `output/clean_dataset.csv`
   - `output/clean_dataset.parquet`
   - `logs/dropped_clips.csv` (every rejected clip with reason)

---

## 6 — Columns produced

| Column | Description |
|--------|-------------|
| `audio_path` | HF Hub path |
| `speaker_id` | From metadata |
| `source_corpus` | From metadata |
| `language` | From metadata |
| `duration` | Seconds after resampling |
| `snr_db` | Estimated SNR |
| `clipping` | True if peak ≥ 0.99 |
| `speech_rate_cps` | Chars/sec in cleaned transcript |
| `devanagari_ratio` | Share of Devanagari word-tokens |
| `roman_ratio` | Share of Roman word-tokens |
| `code_mix_index` | English word share per utterance |
| `oversample_weight` | **3** for spontaneous/CS, **1** for studio |
| `aug_recommended` | **True** for studio clips → apply noise+reverb+speed in training loop |
| `transcript` | Cleaned Whisper output |
| `emb_0 … emb_511` | 512-dim ECAPA speaker embedding |

---

## 7 — Training-loop guidance (FIX 2 downstream)

The pipeline **tags** clips; it does not apply augmentation itself (augmentation happens on-the-fly during training). In your training loop (Person B):

```python
from torch.utils.data import WeightedRandomSampler

weights = dataset_df["oversample_weight"].values.tolist()
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# For clips where aug_recommended == True, apply:
#   - Additive noise (e.g. MUSAN)
#   - Room impulse response convolution (e.g. RIRs from openslr.org/28)
#   - Speed perturbation: librosa.effects.time_stretch(wav, rate=r) with r ∈ {0.9, 1.1}
#   - SpecAugment on the log-Mel tensor (time and frequency masking)
```

---

## 8 — Known limitations (document in your FYP)

| Issue | Note |
|-------|-------|
| **Domain mismatch is vocabulary, not acoustics** | Acoustic transfer is fine; the medical lexicon + hotword biasing addresses vocabulary. Do **not** expect more general audio data to fix this. |
| **SNR estimator is coarse** | The percentile method can overestimate SNR on music-contaminated clips. A WebRTC-VAD-based estimator is more accurate. |
| **Diarization not tested** | `pyannote/embedding` produces clip-level embeddings, which are fine for speaker-disjoint splitting. Full diarization needs the multi-speaker `pyannote/speaker-diarization` pipeline. |
| **WER on clinical test set only measures ASR** | It tells you nothing about entity extraction or FHIR output. Annotated simulated dialogues are still needed for end-to-end evaluation. |

---

## 9 — Next actions

1. Replace the placeholder constants and point `LEXICON_PATH` at your medical term list.
2. Add corpus names that are genuinely spontaneous to `SPONTANEOUS_CORPORA`.
3. Run the pipeline; check `logs/dropped_clips.csv` to calibrate the quality thresholds.
4. Hand the `clean_dataset.parquet` + column definitions to Person B for training.
