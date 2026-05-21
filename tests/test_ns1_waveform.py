"""Tests for the raw-waveform branch of ``NS1Dataset``.

End-to-end tests need the OSF assets unpacked under the user's deepSTRF cache;
they skip automatically when those are missing. The wav→stim-index mapping
verification (the Phase-0 deliverable) is the slowest test — it does a quick
mel-spec correlation against the precomputed ``X_nfht`` for all 20 stims.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch


def _ns1_cache_root() -> str:
    from deepSTRF.utils.data_download import default_cache_dir
    return str(default_cache_dir("NS1"))


def _has_local_ns1_wavs() -> bool:
    root = _ns1_cache_root()
    wav_dir = os.path.join(root, "spikesandwav", "SH.En.C")
    spec_path = os.path.join(root, "test_data_5ms.mat")
    meta_path = os.path.join(root, "MetadataSHEnCneurons.mat")
    return (os.path.isdir(wav_dir)
            and os.path.isfile(spec_path)
            and os.path.isfile(meta_path))


HAS_LOCAL = _has_local_ns1_wavs()
skip_no_data = pytest.mark.skipif(
    not HAS_LOCAL, reason="NS1 OSF + DNet cache missing — skip integration test"
)


def test_filename_constant_layout():
    """The hard-coded wav-name table covers all 20 stims and matches the
    ``source.{1,2}.sound.0.snr.0.token.0.fw.{1,2}.frozen.{1..N}`` pattern."""
    from deepSTRF.datasets.audio.ns1_drc import NS1_WAV_FILENAMES, NS1_NAT_SOUNDS

    assert len(NS1_WAV_FILENAMES) == NS1_NAT_SOUNDS == 20
    for i, fn in enumerate(NS1_WAV_FILENAMES[:12]):
        assert fn == f"source.1.sound.0.snr.0.token.0.fw.2.frozen.{i+1}"
    for i, fn in enumerate(NS1_WAV_FILENAMES[12:]):
        assert fn == f"source.2.sound.0.snr.0.token.0.fw.1.frozen.{i+1}"


def test_load_resampled_mono_wav_unit(tmp_path):
    """Round-trip a synthetic stereo wav through the helper."""
    import torchaudio
    from deepSTRF.utils.audio_io import load_resampled_mono_wav

    fs_native = 48000
    fs_target = 16000
    T_target = 8000  # 0.5 s at target_fs
    stereo = torch.randn(2, fs_native, dtype=torch.float32) * 0.1
    path = str(tmp_path / "tone.wav")
    torchaudio.save(path, stereo, fs_native)

    # exact target_length
    w = load_resampled_mono_wav(path, target_fs=fs_target, target_length=T_target)
    assert w.shape == (1, T_target)
    assert w.dtype == torch.float32

    # no target_length → natural length
    w_nat = load_resampled_mono_wav(path, target_fs=fs_target)
    assert w_nat.shape == (1, fs_target)  # 1 s in, 1 s out at fs_target

    # right-padding when target_length > natural
    w_pad = load_resampled_mono_wav(path, target_fs=fs_target, target_length=fs_target * 2)
    assert w_pad.shape == (1, fs_target * 2)
    assert (w_pad[0, fs_target:] == 0).all()


@skip_no_data
def test_ns1_waveform_shape_and_metadata():
    """``return_waveform=True`` produces ``(1, T_audio)`` stims, advertises
    ``audio_fs`` and ``T_audio = 999 * (audio_fs // 200)`` (= 999 × 240 at
    the default 48 kHz). Also exercise the 16 kHz override path."""
    from deepSTRF.datasets.audio.ns1_drc import NS1Dataset

    # Default audio_fs (48 kHz)
    ds = NS1Dataset(return_waveform=True)
    assert ds.audio_fs == 48000
    assert ds.get_S() == 20
    assert len(ds.stims) == 20
    samples_per_bin = ds.audio_fs // 200  # 200 Hz = 1000 / dt_ms
    for s in range(20):
        assert ds.stims[s].shape == (1, 999 * samples_per_bin), \
            f"stim {s} has wrong shape {ds.stims[s].shape}"
        assert ds.stims[s].dtype == torch.float32

    # Explicit 16 kHz override still works
    ds16 = NS1Dataset(return_waveform=True, audio_fs=16000)
    assert ds16.audio_fs == 16000
    assert ds16.stims[0].shape == (1, 999 * 80)


@skip_no_data
def test_ns1_spec_unchanged():
    """Default ``return_waveform=False`` behaviour is unchanged: ``(1, 34, 999)``
    spec tensors, ``audio_fs is None``."""
    from deepSTRF.datasets.audio.ns1_drc import NS1Dataset

    ds = NS1Dataset()
    assert ds.audio_fs is None
    for s in range(20):
        assert ds.stims[s].shape == (1, 34, 999)


@skip_no_data
def test_ns1_responses_identical_between_modes():
    """Switching to waveform input does not touch the spike/response data."""
    from deepSTRF.datasets.audio.ns1_drc import NS1Dataset

    ds_spec = NS1Dataset()
    ds_wav = NS1Dataset(return_waveform=True)
    assert ds_spec.get_N() == ds_wav.get_N()
    for s in range(0, 20, 5):  # sample a few stims
        for n in range(0, ds_spec.get_N(), 20):
            r_spec = ds_spec.responses[s][n]
            r_wav = ds_wav.responses[s][n]
            assert torch.allclose(r_spec, r_wav, equal_nan=True)


@skip_no_data
def test_ns1_waveform_collate():
    """``neural_collate`` produces ``(B, 1, T_audio)`` stim and
    ``(B, N, R, T_neural)`` responses — the two time axes are independent."""
    from torch.utils.data import DataLoader

    from deepSTRF.datasets.audio.ns1_drc import NS1Dataset
    from deepSTRF.utils import neural_collate

    ds = NS1Dataset(return_waveform=True)
    ds.select_population(list(range(ds.get_N())))
    loader = DataLoader(ds, batch_size=4, collate_fn=neural_collate)
    stims, responses, valid_mask, metas = next(iter(loader))
    expected_T = 999 * (ds.audio_fs // 200)
    assert stims.shape == (4, 1, expected_T)
    assert responses.shape == (4, ds.get_N(), 20, 999)
    assert valid_mask.shape == responses.shape
    assert not torch.isnan(stims).any()


@skip_no_data
def test_ns1_wav_stim_mapping_via_mel_correlation():
    """Phase 0 mapping verification: log-mel of each stim-index wav must be
    more correlated with ``X_nfht[stim_idx]`` than with any other stim's
    spectrogram (best-match diagonal). Tolerates a few off-diagonal ties for
    high-frequency stims (insects buzzing etc.) where our default mel
    parameters clip useful content.
    """
    import scipy.io as sio
    import soundfile as sf
    import torchaudio
    import torchaudio.transforms as T_aud

    from deepSTRF.datasets.audio.ns1_drc import (
        NS1_WAV_FILENAMES, NS1_WAV_DIR_NAME,
    )

    root = _ns1_cache_root()
    X = sio.loadmat(os.path.join(root, "test_data_5ms.mat"))["X_nfht"]
    X_flat = X[:, :, 0, :]  # (20, 34, 999)

    fs_target = 16000
    mel = T_aud.MelSpectrogram(
        sample_rate=fs_target, n_fft=512, win_length=400, hop_length=80,
        f_min=300, f_max=8000, n_mels=34, power=2.0, center=True, mel_scale="htk",
    )

    specs = []
    for fname in NS1_WAV_FILENAMES:
        wav_np, fs_in = sf.read(os.path.join(root, NS1_WAV_DIR_NAME, fname))
        w = torch.tensor(wav_np, dtype=torch.float32)
        if fs_in != fs_target:
            w = torchaudio.functional.resample(w, fs_in, fs_target)
        s = torch.log(mel(w) + 1.0).numpy()
        specs.append(s)
    S = np.stack(specs, axis=0)[..., :999]
    S = (S - S.mean()) / S.std()

    # Best-match diagonal: argmax over stim_idx per wav must hit the diagonal
    # for at least 18/20 stims (allowing two stims of degenerate corr — these
    # are 6 "insects_buzzing" and 7 in the OSF release).
    corr = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            corr[i, j] = np.corrcoef(S[i].ravel(), X_flat[j].ravel())[0, 1]
    best = corr.argmax(axis=1)
    hits = int((best == np.arange(20)).sum())
    assert hits >= 18, (
        f"Only {hits}/20 stims best-match their lexicographic position — "
        f"NS1 wav→stim mapping may be wrong. Diagonal corrs: "
        f"{[float(corr[i, i]) for i in range(20)]}"
    )
