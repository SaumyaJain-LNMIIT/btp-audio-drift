"""Tests for the audio/feature pipeline: resampling, no-leakage scaling, and
metadata persistence. Uses tiny synthetic waveforms (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest

from features.audio_loader import AudioLoader
from features.feature_extractor import FeatureExtractor


# ------------------------- resampling -------------------------
def test_resample_changes_length_to_target():
    sr_in, sr_out, dur = 8000, 22050, 1.0
    t = np.linspace(0, dur, int(sr_in * dur), endpoint=False)
    sine = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    out = AudioLoader.preprocess_waveform(sine, sr_in, sr_out, max_duration=dur)
    assert len(out) == int(sr_out * dur)


def test_resample_noop_when_rates_match():
    sr, dur = 22050, 0.5
    y = np.random.RandomState(0).normal(size=int(sr * dur)).astype(np.float32)
    out = AudioLoader.preprocess_waveform(y, sr, sr, max_duration=dur)
    assert len(out) == int(sr * dur)
    assert np.allclose(out, y, atol=1e-6)


def test_pad_and_truncate():
    sr = 22050
    short = np.ones(sr, dtype=np.float32)              # 1s, want 2s -> pad
    out = AudioLoader.preprocess_waveform(short, sr, sr, max_duration=2.0)
    assert len(out) == 2 * sr and out[-1] == 0.0
    long = np.ones(4 * sr, dtype=np.float32)           # 4s, want 2s -> truncate
    out2 = AudioLoader.preprocess_waveform(long, sr, sr, max_duration=2.0)
    assert len(out2) == 2 * sr


def test_stereo_is_downmixed():
    sr = 22050
    stereo = np.ones((2, sr), dtype=np.float32)
    out = AudioLoader.preprocess_waveform(stereo, sr, sr, max_duration=1.0)
    assert out.ndim == 1 and len(out) == sr


# ------------------------- no-leakage scaling -------------------------
def test_scaler_fits_on_train_only():
    rng = np.random.RandomState(0)
    train = rng.normal(5.0, 2.0, size=(100, 39)).astype(np.float32)
    test = rng.normal(50.0, 2.0, size=(40, 39)).astype(np.float32)  # very different

    fe = FeatureExtractor()
    fe.fit_scaler(train)
    train_z = fe.transform(train)
    test_z = fe.transform(test)

    # training data is standardised to ~0 mean / ~1 std
    assert abs(train_z.mean()) < 0.1
    assert abs(train_z.std() - 1.0) < 0.1
    # the scaler used TRAIN statistics, so out-of-distribution test data is NOT
    # centred at 0 (this is exactly what "no leakage" looks like)
    assert test_z.mean() > 5.0


def test_transform_before_fit_raises():
    fe = FeatureExtractor()
    with pytest.raises(RuntimeError):
        fe.transform(np.zeros((3, 39), dtype=np.float32))


def test_save_and_load_scaler(tmp_path):
    rng = np.random.RandomState(1)
    train = rng.normal(3.0, 1.5, size=(80, 39)).astype(np.float32)
    fe = FeatureExtractor()
    fe.fit_scaler(train)
    ref = fe.transform(train)

    path = tmp_path / "scaler.npz"
    fe.save_scaler(path)
    fe2 = FeatureExtractor()
    fe2.load_scaler(path)
    assert np.allclose(fe2.transform(train), ref, atol=1e-5)


# ------------------------- extract + persist metadata -------------------------
def _fake_clips(n=6, sr=22050):
    rng = np.random.RandomState(2)
    clips = []
    for i in range(n):
        clips.append({
            "clip_id": f"clip{i}",
            "audio": rng.normal(size=sr).astype(np.float32),
            "sr": sr,
            "orig_sr": 44100,
            "label": f"class{i % 3}",
            "fold": (i % 10) + 1,
        })
    return clips


def test_extract_batch_raw_is_unscaled():
    fe = FeatureExtractor()
    feats, labels, names = fe.extract_batch(_fake_clips(), scale=False)
    assert feats.shape[1] == 39
    assert len(labels) == 6 and len(names) == 3


def test_extract_batch_persists_metadata(tmp_path):
    fe = FeatureExtractor()
    feats, labels, names, meta = fe.extract_batch(
        _fake_clips(), save_path=tmp_path, scale=False, return_meta=True
    )
    assert set(meta["folds"]) and len(meta["clip_ids"]) == 6

    loaded = FeatureExtractor.load_cached_full(tmp_path / "features.npz")
    assert loaded["features"].shape == feats.shape
    assert "folds" in loaded and "clip_ids" in loaded
    assert "orig_srs" in loaded and loaded["orig_srs"][0] == 44100