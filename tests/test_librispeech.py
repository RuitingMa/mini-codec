"""Hermetic tests for ``LibriSpeechSegments``.

These tests build a tiny synthetic LibriSpeech-shaped directory under
``tmp_path`` (a few mono FLAC files at 16 kHz) so they run without the real
~1 GB ``dev-clean`` download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.librispeech import LibriSpeechSegments

NATIVE_SR = 16000


def _write_flac(path: Path, duration_seconds: float, freq: float = 440.0) -> None:
    """Write a mono 16-bit FLAC sine wave at the LibriSpeech native sample rate."""
    n = int(round(duration_seconds * NATIVE_SR))
    t = np.arange(n) / NATIVE_SR
    # 0.5 amplitude keeps things well inside [-1, 1].
    wav = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, NATIVE_SR, subtype="PCM_16", format="FLAC")


@pytest.fixture
def fake_librispeech(tmp_path: Path) -> Path:
    """Build a fake LibriSpeech tree with three utterances of varying length.

    Layout: <tmp_path>/LibriSpeech/dev-clean/<spk>/<chap>/<spk>-<chap>-<utt>.flac
    """
    root = tmp_path
    base = root / "LibriSpeech" / "dev-clean"
    # 1.5s and 2.0s should pass the 1.0s segment filter; 0.5s should be dropped.
    _write_flac(base / "84" / "121123" / "84-121123-0001.flac", 1.5, freq=440)
    _write_flac(base / "84" / "121123" / "84-121123-0002.flac", 2.0, freq=660)
    _write_flac(base / "174" / "50561" / "174-50561-0001.flac", 0.5, freq=220)
    return root


def test_shape_native_sr(fake_librispeech: Path) -> None:
    ds = LibriSpeechSegments(
        root=fake_librispeech, split="dev-clean", sample_rate=16000, segment_seconds=1.0
    )
    assert len(ds) == 2  # the 0.5s file is filtered out
    x = ds[0]
    assert x.dtype == torch.float32
    assert x.shape == (1, 16000)


def test_shape_resampled_to_24k(fake_librispeech: Path) -> None:
    ds = LibriSpeechSegments(
        root=fake_librispeech, split="dev-clean", sample_rate=24000, segment_seconds=1.0
    )
    x = ds[0]
    assert x.shape == (1, 24000)


def test_amplitude_range(fake_librispeech: Path) -> None:
    ds = LibriSpeechSegments(root=fake_librispeech, split="dev-clean")
    for i in range(len(ds)):
        x = ds[i]
        # 16-bit PCM round-trip can nudge values slightly past the source 0.5
        # but must always live in [-1, 1].
        assert x.abs().max() <= 1.0 + 1e-6


def test_short_utterance_filtered(fake_librispeech: Path) -> None:
    # All three utterances qualify for a 0.4s segment.
    ds_short = LibriSpeechSegments(
        root=fake_librispeech, split="dev-clean", segment_seconds=0.4
    )
    assert len(ds_short) == 3

    # But only the 2.0s file qualifies for a 1.8s segment.
    ds_long = LibriSpeechSegments(
        root=fake_librispeech, split="dev-clean", segment_seconds=1.8
    )
    assert len(ds_long) == 1


def test_deterministic_when_random_crop_off(fake_librispeech: Path) -> None:
    ds = LibriSpeechSegments(
        root=fake_librispeech, split="dev-clean", random_crop=False
    )
    assert torch.equal(ds[0], ds[0])


def test_seeded_random_crop_is_reproducible(fake_librispeech: Path) -> None:
    ds_a = LibriSpeechSegments(root=fake_librispeech, split="dev-clean", seed=42)
    ds_b = LibriSpeechSegments(root=fake_librispeech, split="dev-clean", seed=42)
    # Same seed → same crop sequence for the same access pattern.
    a = torch.stack([ds_a[i] for i in range(len(ds_a))])
    b = torch.stack([ds_b[i] for i in range(len(ds_b))])
    assert torch.equal(a, b)


def test_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LibriSpeechSegments(root=tmp_path, split="dev-clean")
