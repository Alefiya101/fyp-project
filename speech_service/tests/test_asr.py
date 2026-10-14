# -*- coding: utf-8 -*-
"""test_asr.py

Unit tests for:
  - speech_service.asr.baseline_eval  (compute_wer, evaluate_dataset CSV schema)
  - speech_service.asr.finetune       (load_dataset_from_parquet, missing-row exclusion)

All tests mock model loading — no network access required.
# Feature: person-a-speech-pipeline
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from speech_service.asr.baseline_eval import compute_wer, evaluate_dataset
from speech_service.asr.finetune import load_dataset_from_parquet

SR = 16_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_s: float = 0.5, sr: int = SR) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), wav, sr, subtype="PCM_16")


@pytest.fixture()
def tiny_parquet(tmp_path):
    """Write a 3-clip parquet with valid audio files and return its path."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    records = []
    for i in range(3):
        clip_id = f"clip_{i:03d}"
        audio_path = audio_dir / f"{clip_id}.wav"
        _write_wav(audio_path)
        records.append({
            "clip_id": clip_id,
            "audio_path": str(audio_path),
            "reference_text": f"sample text {i}",
            "lang_tag": "hi-en",
            "duration": 0.5,
        })
    df = pd.DataFrame(records)
    parquet_path = tmp_path / "clean_dataset.parquet"
    df.to_parquet(str(parquet_path), index=False)
    return str(parquet_path)


@pytest.fixture()
def parquet_with_missing(tmp_path):
    """Parquet where 1 of 4 rows has a missing audio_path."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    records = []
    for i in range(4):
        clip_id = f"clip_{i:03d}"
        if i == 2:
            # missing path
            audio_path = ""
        else:
            audio_path = str(audio_dir / f"{clip_id}.wav")
            _write_wav(Path(audio_path))
        records.append({
            "clip_id": clip_id,
            "audio_path": audio_path,
            "reference_text": f"text {i}",
            "lang_tag": "hi-en",
            "duration": 0.5,
        })
    df = pd.DataFrame(records)
    p = tmp_path / "clean_dataset.parquet"
    df.to_parquet(str(p), index=False)
    return str(p)


# ===========================================================================
# compute_wer — unit tests
# ===========================================================================

class TestComputeWer:
    def test_identical_strings_return_zero(self):
        assert compute_wer("hello world", "hello world") == pytest.approx(0.0)

    def test_empty_reference_returns_nan(self):
        assert math.isnan(compute_wer("", "anything"))

    def test_empty_hypothesis_returns_nan(self):
        assert math.isnan(compute_wer("something", ""))

    def test_both_empty_returns_nan(self):
        assert math.isnan(compute_wer("", ""))

    def test_completely_wrong_hypothesis_greater_than_zero(self):
        wer = compute_wer("the quick brown fox", "cat sat mat hat")
        assert not math.isnan(wer)
        assert wer > 0.0

    def test_normalisation_case_insensitive(self):
        wer = compute_wer("Hello World", "hello world")
        assert wer == pytest.approx(0.0)


# ===========================================================================
# evaluate_dataset — CSV schema tests
# ===========================================================================

class TestEvaluateDataset:
    def _mock_whisper_model(self):
        model = MagicMock()
        seg = MagicMock()
        seg.text = "sample text"
        seg.avg_logprob = -0.1
        model.transcribe.return_value = ([seg], MagicMock())
        return model

    def test_csv_has_required_columns(self, tiny_parquet, tmp_path):
        output_csv = str(tmp_path / "results.csv")
        mock_model = self._mock_whisper_model()

        with patch("speech_service.asr.baseline_eval.load_model", return_value=mock_model), \
             patch("speech_service.asr.baseline_eval.transcribe_clip", return_value="sample text"):
            evaluate_dataset(tiny_parquet, model_size="base", output_csv=output_csv)

        assert Path(output_csv).exists()
        df = pd.read_csv(output_csv)
        required = {"clip_id", "reference", "hypothesis", "wer"}
        assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

    def test_csv_has_one_row_per_clip(self, tiny_parquet, tmp_path):
        output_csv = str(tmp_path / "results.csv")
        with patch("speech_service.asr.baseline_eval.load_model", return_value=self._mock_whisper_model()), \
             patch("speech_service.asr.baseline_eval.transcribe_clip", return_value="text"):
            evaluate_dataset(tiny_parquet, model_size="base", output_csv=output_csv)

        df = pd.read_csv(output_csv)
        assert len(df) == 3

    def test_empty_hypothesis_records_nan_wer(self, tiny_parquet, tmp_path):
        output_csv = str(tmp_path / "results.csv")
        with patch("speech_service.asr.baseline_eval.load_model", return_value=self._mock_whisper_model()), \
             patch("speech_service.asr.baseline_eval.transcribe_clip", return_value=""):
            evaluate_dataset(tiny_parquet, model_size="base", output_csv=output_csv)

        df = pd.read_csv(output_csv)
        assert df["wer"].isna().all() or (df["wer"].apply(lambda x: math.isnan(float(x)) if pd.notna(x) else True)).all()


# ===========================================================================
# finetune.load_dataset_from_parquet — missing row exclusion
# ===========================================================================

class TestLoadDatasetFromParquet:
    def test_drops_missing_audio_path_rows(self, parquet_with_missing):
        """Property 5: rows with missing audio_path are excluded."""
        ds = load_dataset_from_parquet(parquet_with_missing)
        # 4 rows total, 1 missing path, 1 file doesn't exist on disk → 2 or 3 valid
        assert len(ds) <= 3
        for row in ds:
            assert row.get("audio_path", "").strip() != ""

    def test_returned_dataset_has_required_columns(self, parquet_with_missing):
        ds = load_dataset_from_parquet(parquet_with_missing)
        assert "audio_path" in ds.column_names
        assert "text" in ds.column_names

    def test_max_clips_cap(self, tiny_parquet):
        ds = load_dataset_from_parquet(tiny_parquet, max_clips=2)
        assert len(ds) <= 2

    def test_all_valid_rows_kept(self, tiny_parquet):
        ds = load_dataset_from_parquet(tiny_parquet)
        assert len(ds) == 3


# ===========================================================================
# Property-based tests
# ===========================================================================

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    _HYPOTHESIS = True
except ImportError:
    _HYPOTHESIS = False

if _HYPOTHESIS:
    # Feature: person-a-speech-pipeline, Property 3
    @given(
        text=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
            min_size=1, max_size=50,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_wer_identity_zero(text):
        """Property 3: compute_wer(s, s) == 0.0 for any non-empty normalised string."""
        if not text.strip():
            return
        result = compute_wer(text, text)
        assert result == pytest.approx(0.0, abs=1e-6) or math.isnan(result)

    # Feature: person-a-speech-pipeline, Property 5
    @given(
        n_valid=st.integers(min_value=1, max_value=10),
        n_missing=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_missing_rows_excluded_from_dataset(tmp_path_factory, n_valid, n_missing):
        """Property 5: missing audio_path rows are excluded; count matches."""
        tmp_path = tmp_path_factory.mktemp("data")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        records = []
        for i in range(n_valid):
            p = audio_dir / f"clip_{i}.wav"
            _write_wav(p)
            records.append({"clip_id": f"v{i}", "audio_path": str(p), "reference_text": "text", "lang_tag": "hi-en", "duration": 0.5})
        for i in range(n_missing):
            records.append({"clip_id": f"m{i}", "audio_path": "", "reference_text": "text", "lang_tag": "hi-en", "duration": 0.5})

        df = pd.DataFrame(records)
        p_path = tmp_path / "ds.parquet"
        df.to_parquet(str(p_path), index=False)

        ds = load_dataset_from_parquet(str(p_path))
        assert len(ds) == n_valid
