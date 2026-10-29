"""Tests for the trial-audio helpers shared by the Gosselin 2025 and Bagur 2025 loaders."""
from __future__ import annotations

import numpy as np
import pytest

from deepSTRF.datasets.audio import _trial_audio as ta


def test_trial_waveform_onset_param():
    wav = np.ones(100)
    buf = ta.trial_waveform(wav, n_bins=300, hop=960, onset_samples=96000)
    assert buf.shape == (300 * 960,)
    assert np.flatnonzero(buf)[0] == 96000 and buf.sum() == 100


def test_overlap_weights_fractional_and_offset():
    W = ta.overlap_weights(300, 5.0, 31.7, 47)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert W[0, :6].sum() == pytest.approx(30.0 / 31.7)      # bin 0 = [0, 31.7) ms: inputs 0..5
    assert W[0, 6] == pytest.approx(1.7 / 31.7) and W[0, 7] == 0
    assert W[0, 0] == pytest.approx(5.0 / 31.7)
    Ws = ta.overlap_weights(300, 5.0, 31.7, 47, offset_ms=2.5)  # bin 0 = [2.5, 34.2) ms
    assert Ws[0, 0] == pytest.approx(2.5 / 31.7) and Ws[0, 6] == pytest.approx(4.2 / 31.7)


def test_find_gain_copies_sign_and_gain():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    wavs = {"a": x, "b": -0.1 * x, "c": 3.0 * x, "d": rng.standard_normal(1000),
            "e": rng.standard_normal(500)}
    assert ta.find_gain_copies(wavs) == [["a", "b", "c"]]


def test_md5_stamp_skips_rehash(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    calls = []

    def hasher(p):
        calls.append(p.name)
        return ta.md5_of(p)

    good = ta.md5_of(f)
    assert ta.md5_intact(tmp_path, "x.bin", good, hasher=hasher)
    assert ta.md5_intact(tmp_path, "x.bin", good, hasher=hasher)
    assert calls == ["x.bin"]
    f.write_bytes(b"changed!")
    assert not ta.md5_intact(tmp_path, "x.bin", good, hasher=hasher)


def test_fetch_checked_refetches_corrupt(tmp_path):
    (tmp_path / "y.bin").write_bytes(b"corrupt")
    good = b"payload"

    def download(record, name, dest):
        dest.write_bytes(good)

    import hashlib
    ta.fetch_checked(1, "y.bin", tmp_path, hashlib.md5(good).hexdigest(), download=download)
    assert (tmp_path / "y.bin").read_bytes() == good
    with pytest.raises(IOError):
        ta.fetch_checked(1, "z.bin", tmp_path, "0" * 32, download=download)
    assert not (tmp_path / "z.bin").exists()
