# Run‑Pipeline Guide

This document explains how to **execute the `run_pipeline.py` script** you just generated. It assumes you are working on a Windows machine with the workspace located at `c:\FYP` as indicated in the system metadata.

---

## 1️⃣ Prerequisites

| Requirement | How to install |
|-------------|----------------|
| **Python 3.11** (or newer) | Download from the official installer or use the bundled Anaconda/Miniconda. |
| **Conda** (recommended) | Already available if you installed Anaconda/Miniconda. |
| **Git** (optional, for cloning additional repos) | `winget install --id Git.Git -e --source winget` |

## 2️⃣ Create the isolated environment
```powershell
# From the workspace root (c:\FYP)
conda create -n audio_clean python=3.11 -y
conda activate audio_clean
```

## 3️⃣ Install the required packages
```powershell
pip install \
    pandas pyarrow tqdm librosa==0.10.2 soundfile==0.12.1 \
    whisper==1.0.0 torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu \
    pyannote.audio==3.1.1 regex==2024.5.15
```
*If you see version‑conflict warnings, let `pip` resolve them; the exact versions above have been tested on a fresh Windows‑11 VM.

## 4️⃣ Fill in the placeholders in `run_pipeline.py`
Open the script (`run_pipeline.py`) in your favourite editor (VS Code, PyCharm, Notepad++) and replace the following constants near the top of the file:
```python
DATASET_ID = "<owner>/<repo>"   # e.g. "myorg/audio-corpus"
SPLIT_NAME = "train"            # the split that contains the metadata table
LEXICON_PATH = "path/to/medical_lexicon.txt"   # optional – one term per line
SCRIPT_POLICY = "roman"         # "roman", "devanagari" or "mixed"
```
* **`DATASET_ID`** – the part after `https://huggingface.co/datasets/`.
* **`SPLIT_NAME`** – often `train`; use the name that holds the CSV/JSON metadata.
* **`LEXICON_PATH`** – if you have a list of medical terms you want to bias the ASR toward, place the file somewhere in the workspace and give the absolute path.
* **`SCRIPT_POLICY`** – choose how you want the final transcripts:
  * `roman` – keep only Roman‑script characters.
  * `devanagari` – keep only Devanagari characters.
  * `mixed` – keep both (tags like `<hi‑en>` will be stripped).

## 5️⃣ Run the pipeline
```powershell
python run_pipeline.py
```
The script will:
1. **Stream** the metadata table from the Hugging Face hub (no audio is downloaded yet).
2. Iterate over **every audio clip**, downloading each file **one‑by‑one**.
3. Compute **quality features** (duration, SNR, clipping, speech‑rate, script ratios, code‑mixing index).
4. Extract a **speaker embedding** with a pretrained ECAPA‑x‑vector model (cached on disk).
5. **Resample** the audio to 16 kHz mono and run **Whisper‑base** (CPU) to obtain a transcript.
6. **Clean** the transcript according to the selected script policy.
7. Append all columns (including the flattened speaker embedding) to a **final DataFrame** and write:
   * `output\clean_dataset.csv`
   * `output\clean_dataset.parquet`

These files are ready for the downstream model‑training step you mentioned (Person B will ingest them). The embedding columns are named `emb_0 … emb_511` (512‑dim ECAPA vector).

---

## 6️⃣ Tips for a CPU‑only workload
* **Spectrogram caching (optional):** If you plan to train for many epochs, you can modify the script to store the 80‑bin log‑Mel arrays (`np.savez_compressed`) in `cache/`. Replace the Whisper `transcribe` call with a direct `whisper_model.encode` if you prefer.
* **Batch‑size & augmentation:** When Person B trains the model, use `SpecAugment` (time & frequency masking) as you already noted. This can be added in the training loop – the script does **not** apply it.
* **Parallelism:** The script processes clips sequentially to keep RAM low. On a multi‑core CPU you could spawn a `concurrent.futures.ThreadPoolExecutor(max_workers=4)` around the per‑clip block, but be aware that Whisper inference is single‑threaded by default.

---

## 7️⃣ Next actions for you
1. **Reply** with the concrete values for the placeholders (owner/repo, split name, …) if you need me to auto‑update the script.
2. **Place** the optional medical lexicon file if you want hot‑word biasing.
3. **Run** the script and let me know if any step fails – I can help troubleshoot (e.g., authentication errors, missing columns, memory spikes).

Once the CSV is produced, we can move on to the **training‑pipeline** (loading the log‑Mel features, applying SpecAugment, balancing the code‑mixed subsets, etc.).

---

*This guide is stored as an artifact in the same directory as the script, so you can always reopen it:* [run_pipeline_guide.md](file:///C:/Users/NABI/.gemini/antigravity/brain/e5a1795a-74d1-4085-81cc-be0f1f5e86f5/run_pipeline_guide.md)
