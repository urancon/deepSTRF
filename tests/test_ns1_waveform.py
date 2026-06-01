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
    import soundfile as sf
    from deepSTRF.utils.audio_io import load_resampled_mono_wav

    fs_native = 48000
    fs_target = 16000
    T_target = 8000  # 0.5 s at target_fs
    stereo = torch.randn(2, fs_native, dtype=torch.float32) * 0.1
    path = str(tmp_path / "tone.wav")
    # soundfile.write wants (frames, channels); avoid torchaudio.save (FFmpeg).
    sf.write(path, stereo.t().numpy(), fs_native)

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


@skip_no_data
def test_causal_mel_matches_groundtruth_spectrogram():
    """Spec-fallback equivalence (Phase-B arm a): the shipped strictly-causal
    CausalMelSpectrogram, run on the waveform branch, produces a spectrogram
    that best-matches its *own* stim's precomputed ``X_nfht`` for (almost)
    every stim.

    Absolute per-stim correlation has a high-frequency tail (the htk-mel vs
    voicebox toolchain gap — mean ≈ 0.66, with the 'insect' stim as low as
    ~0.04), so we assert the robust best-match *diagonal*, not a per-stim
    magnitude. This is the spectrogram-level equivalence; the task-level
    equivalence (cc_norm parity) lives in the examples notebook. Currently
    20/20.
    """
    import torch
    from deepSTRF.datasets.audio.ns1_drc import NS1Dataset
    from deepSTRF.models.wav2spec import CausalMelSpectrogram

    ds_wav = NS1Dataset(return_waveform=True)
    ds_spec = NS1Dataset()
    mel = CausalMelSpectrogram(audio_fs=ds_wav.audio_fs, n_mels=ds_wav.F,
                               hop_ms=ds_wav.dt)
    mel.eval()
    S = ds_wav.get_S()
    specs, X = [], []
    with torch.no_grad():
        for s in range(S):
            specs.append(mel(ds_wav.stims[s].unsqueeze(0)).squeeze().numpy())
            X.append(ds_spec.stims[s].squeeze().numpy())
    corr = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            T = min(specs[i].shape[-1], X[j].shape[-1])
            corr[i, j] = np.corrcoef(specs[i][..., :T].ravel(),
                                     X[j][..., :T].ravel())[0, 1]
    hits = int((corr.argmax(axis=1) == np.arange(S)).sum())
    assert hits >= 18, (
        f"CausalMel best-match diagonal only {hits}/{S} — the model-side "
        f"causal mel no longer reproduces the dataset-side spectrogram. "
        f"diag corrs: {[round(float(corr[i, i]), 2) for i in range(S)]}"
    )


# --------------------------------------------------------------------------- #
#  Waveform grid-lock (base-class AudioNeuralDataset.validate) — these run     #
#  without any on-disk data via a minimal in-memory subclass.                  #
# --------------------------------------------------------------------------- #

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset


class _ToyAudioWav(AudioNeuralDataset):
    """Minimal in-memory ``AudioNeuralDataset`` for exercising the base-class
    waveform grid-lock validation without any on-disk assets.

    Builds a clean ``T_audio = T_neural * hop`` grid by default; tests mutate
    ``self.stims`` / ``self.responses`` afterwards to inject violations and then
    call ``validate()`` explicitly.
    """

    def __init__(self, audio_fs=48000, dt_ms=5.0, n_stims=3, T_neural=10, N=2,
                 hearing_range_hz=(200.0, 40000.0)):
        super().__init__(path="toy", dt_ms=dt_ms)
        self.F = 34
        self.audio_fs = audio_fs
        # validate() gates the waveform grid-lock check on return_waveform
        # (not audio_fs — spec-mode datasets like Downer2025 set audio_fs too).
        self.return_waveform = audio_fs is not None
        self.hearing_range_hz = hearing_range_hz
        hop = int(round(audio_fs * dt_ms / 1000)) if audio_fs else 1
        self.stims = [torch.zeros(1, T_neural * hop) for _ in range(n_stims)]
        self.responses = [[torch.zeros(3, T_neural) for _ in range(N)]
                          for _ in range(n_stims)]
        self.stim_meta = [{"name": f"s{i}"} for i in range(n_stims)]
        self.nrn_meta = [{"id": j} for j in range(N)]
        self.N_neurons = N


def test_audio_grid_lock_accepts_clean_toy():
    ds = _ToyAudioWav(audio_fs=48000, dt_ms=5.0, T_neural=10)
    ds.validate()  # must not raise
    assert ds.hop == 240
    assert ds.hearing_range_hz == (200.0, 40000.0)


def test_audio_hop_none_in_spec_mode():
    ds = _ToyAudioWav(audio_fs=None)
    assert ds.hop is None
    ds.validate()  # grid-lock skipped entirely when audio_fs is None


def test_audio_grid_lock_rejects_bad_pad():
    """T_audio not a multiple of hop → reject."""
    ds = _ToyAudioWav(T_neural=10)
    ds.stims[1] = torch.zeros(1, 10 * ds.hop + 1)  # off by one sample
    with pytest.raises(AssertionError, match="multiple of"):
        ds.validate()


def test_audio_grid_lock_rejects_wrong_frame_count():
    """T_audio a multiple of hop but T_audio // hop != T_resp → reject."""
    ds = _ToyAudioWav(T_neural=10)
    ds.stims[2] = torch.zeros(1, 11 * ds.hop)  # 11 frames vs 10 response bins
    with pytest.raises(AssertionError, match="neural"):
        ds.validate()


def test_audio_grid_lock_rejects_spec_shaped_stim():
    """A (1, F, T) spec tensor handed in while audio_fs is set → reject."""
    ds = _ToyAudioWav(T_neural=10)
    ds.stims[0] = torch.zeros(1, 34, 10)  # rank-3, wrong for waveform mode
    with pytest.raises(AssertionError, match=r"\(1, T_audio\)"):
        ds.validate()


def test_audio_grid_lock_rejects_noninteger_hop():
    """audio_fs * dt_ms / 1000 not integer → reject before per-stim checks."""
    ds = _ToyAudioWav(audio_fs=44100, dt_ms=5.0, T_neural=10)  # 220.5 samples/bin
    with pytest.raises(AssertionError, match="integer number of samples"):
        ds.validate()


def test_audio_grid_lock_skips_allsentinel_stim():
    """A stim no neuron heard (all (1,1)-NaN sentinels) can't be aligned, so
    its waveform length is not checked against a response length."""
    ds = _ToyAudioWav(T_neural=10)
    nan = torch.full((1, 1), float("nan"))
    ds.responses[0] = [nan, nan]                 # nobody heard stim 0
    ds.stims[0] = torch.zeros(1, 7 * ds.hop)     # "wrong" length, but unanchored
    ds.validate()  # must not raise


def test_audio_hearing_range_validation():
    ds = _ToyAudioWav(hearing_range_hz=(40000.0, 200.0))  # decreasing → invalid
    with pytest.raises(AssertionError, match="hearing_range_hz"):
        ds.validate()


@skip_no_data
def test_ns1_waveform_validate_and_attrs():
    """The real NS1 waveform dataset passes the new grid-lock validation and
    advertises hop / hearing_range_hz; spec mode leaves hop None."""
    from deepSTRF.datasets.audio.ns1_drc import NS1Dataset

    ds = NS1Dataset(return_waveform=True)
    ds.validate()  # exercised at construction too, but be explicit
    assert ds.hop == 240
    assert ds.hearing_range_hz == (200.0, 40000.0)

    ds_spec = NS1Dataset()
    assert ds_spec.hop is None
    assert ds_spec.hearing_range_hz == (200.0, 40000.0)  # set regardless of mode
