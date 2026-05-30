"""Tests for ``deepSTRF.models.wav2spec`` front-ends.

Every registered wav2spec module is exercised against three parametrised
contracts: shape, eval-mode determinism, and strict causality (a Jacobian
probe that asserts gradient of output frame ``k`` w.r.t. any audio sample
later than ``(k+1) * hop - 1`` is exactly zero).
"""
from __future__ import annotations

import math

import pytest
import torch


# Each entry is (label, ctor) where ctor() -> nn.Module. Add new wav2spec
# modules here as they land and they'll inherit all the parametrised checks.
WAV2SPEC_CASES = [
    ("CausalMelSpectrogram-16kHz-5ms-25ms",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["CausalMelSpectrogram"])
             .CausalMelSpectrogram(audio_fs=16000, n_mels=34, hop_ms=5.0, win_ms=25.0)),
    ("CausalMelSpectrogram-16kHz-5ms-10ms",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["CausalMelSpectrogram"])
             .CausalMelSpectrogram(audio_fs=16000, n_mels=34, hop_ms=5.0, win_ms=10.0)),
    ("SincNet-16kHz-5ms-K251-mel-symlog",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["SincNet"])
             .SincNet(audio_fs=16000, n_filters=34, kernel_size=251, hop_ms=5.0,
                      init="mel", activation="symlog")),
    ("SincNet-16kHz-5ms-K64-linear-logabs",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["SincNet"])
             .SincNet(audio_fs=16000, n_filters=48, kernel_size=64, hop_ms=5.0,
                      init="linear", activation="logabs")),
    ("SincNet-16kHz-5ms-K251-envelope-logabs",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["SincNet"])
             .SincNet(audio_fs=16000, n_filters=34, kernel_size=251, hop_ms=5.0,
                      init="mel", activation="logabs", envelope=True)),
    ("CausalGammatone-16kHz-5ms-log-halfwave",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["CausalGammatone"])
             .CausalGammatone(audio_fs=16000, n_filters=34, hop_ms=5.0,
                              f_min=300.0, f_max=7000.0, kernel_ms=20.0)),
    ("CausalGammatone-16kHz-5ms-cuberoot-full",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["CausalGammatone"])
             .CausalGammatone(audio_fs=16000, n_filters=24, hop_ms=5.0,
                              f_min=300.0, f_max=7000.0, kernel_ms=15.0,
                              rectify="full", compression="cuberoot",
                              env_window_ms=10.0)),
    ("CausalGammatone-16kHz-5ms-pcen",
     lambda: __import__("deepSTRF.models.wav2spec", fromlist=["CausalGammatone"])
             .CausalGammatone(audio_fs=16000, n_filters=24, hop_ms=5.0,
                              f_min=300.0, f_max=7000.0, kernel_ms=15.0,
                              compression="pcen")),
]


@pytest.fixture(params=WAV2SPEC_CASES, ids=[c[0] for c in WAV2SPEC_CASES])
def wav2spec(request):
    return request.param[1]()


def test_factory_dispatch():
    from deepSTRF.models.wav2spec import (
        CausalMelSpectrogram, SincNet, make_wav2spec,
    )

    m = make_wav2spec("mel", audio_fs=16000, dt_ms=5.0)
    assert isinstance(m, CausalMelSpectrogram)
    assert m.audio_fs == 16000
    assert m.hop == 80
    assert m.out_channels == 34

    s = make_wav2spec("sincnet", audio_fs=16000, dt_ms=5.0,
                      n_filters=48, kernel_size=64)
    assert isinstance(s, SincNet)
    assert s.audio_fs == 16000
    assert s.hop == 80
    assert s.out_channels == 48
    assert s.kernel_size == 64

    from deepSTRF.models.wav2spec import CausalGammatone
    g = make_wav2spec("gammatone", audio_fs=16000, dt_ms=5.0, n_filters=24)
    assert isinstance(g, CausalGammatone)
    assert g.audio_fs == 16000
    assert g.hop == 80
    assert g.out_channels == 24

    with pytest.raises(ValueError):
        make_wav2spec("not-a-real-frontend", audio_fs=16000, dt_ms=5.0)


def test_sincnet_constructor_rejects_bad_args():
    from deepSTRF.models.wav2spec import SincNet

    with pytest.raises(ValueError):
        SincNet(audio_fs=0)
    with pytest.raises(ValueError):
        SincNet(audio_fs=16000, activation="not-a-real-activation")
    with pytest.raises(ValueError):
        SincNet(audio_fs=16000, init="not-a-real-init")


def test_sincnet_gradient_flow_through_cutoffs():
    """Backprop populates ``.grad`` on both ``low_hz_`` and ``band_hz_``."""
    from deepSTRF.models.wav2spec import SincNet

    sn = SincNet(audio_fs=16000, n_filters=34, kernel_size=251, hop_ms=5.0)
    y = sn(torch.randn(2, 1, 100 * sn.hop) * 0.1)
    y.pow(2).mean().backward()
    assert sn.low_hz_.grad is not None
    assert sn.band_hz_.grad is not None
    assert sn.low_hz_.grad.abs().sum().item() > 0
    assert sn.band_hz_.grad.abs().sum().item() > 0


def test_sincnet_cutoff_clamps():
    """``f1`` and ``f2`` stay in (0, fs/2] regardless of raw param value."""
    from deepSTRF.models.wav2spec import SincNet

    sn = SincNet(audio_fs=16000, n_filters=34, kernel_size=251, hop_ms=5.0)
    # tamper with the raw params: push low_hz negative and band_hz large
    with torch.no_grad():
        sn.low_hz_.data[:] = -1234.0
        sn.band_hz_.data[:] = 1e6
    f1, f2 = sn.f1, sn.f2
    assert (f1 >= 1.0).all() and (f1 < sn.audio_fs / 2).all()
    assert (f2 > f1).all() and (f2 <= sn.audio_fs / 2).all()


def test_constructor_rejects_bad_args():
    from deepSTRF.models.wav2spec import CausalMelSpectrogram

    with pytest.raises(ValueError):
        CausalMelSpectrogram(audio_fs=0)
    with pytest.raises(ValueError):
        CausalMelSpectrogram(audio_fs=16000, hop_ms=10.0, win_ms=5.0)  # win < hop


def test_wav2spec_shape(wav2spec):
    """Forward ``(B, 1, T_audio)`` waveform → ``(B, 1, n_mels, T_neural)``."""
    T_neural = 200
    T_audio = T_neural * wav2spec.hop
    x = torch.randn(3, 1, T_audio)
    y = wav2spec(x)
    assert y.shape == (3, 1, wav2spec.out_channels, T_neural)


def test_wav2spec_eval_mode_deterministic(wav2spec):
    """Same input gives the same output across forwards in eval mode."""
    wav2spec.eval()
    x = torch.randn(2, 1, 100 * wav2spec.hop)
    with torch.no_grad():
        y0 = wav2spec(x)
        for _ in range(4):
            assert torch.equal(wav2spec(x), y0)


def test_wav2spec_jacobian_causal(wav2spec):
    """Jacobian probe: ``∂y[..., k] / ∂x[..., j] == 0`` for all j strictly
    later than ``(k+1) * hop - 1`` (i.e. any audio sample in neural bin
    ``k+1`` or later)."""
    T_neural = 60
    K = 30  # probe a middle output frame
    x = torch.randn(1, 1, T_neural * wav2spec.hop, requires_grad=True)
    y = wav2spec(x)
    # gradient of all features at frame K wrt every input sample
    y[0, 0, :, K].sum().backward()
    grad = x.grad[0, 0]
    cutoff = (K + 1) * wav2spec.hop  # first audio sample of bin K+1
    future = grad[cutoff:]
    past = grad[:cutoff]
    assert future.abs().max().item() == 0.0, (
        f"causality violation: ∂y[K={K}]/∂x[j>={cutoff}] max abs = "
        f"{future.abs().max().item():.3e}"
    )
    assert past.abs().max().item() > 0.0, (
        f"gradient is identically zero in the past — module forgot to look at the input?"
    )


def test_wav2spec_input_validation(wav2spec):
    """Rejects inputs with wrong rank / wrong channel count."""
    with pytest.raises(ValueError):
        wav2spec(torch.randn(10 * wav2spec.hop))      # 1D
    with pytest.raises(ValueError):
        wav2spec(torch.randn(2, 10 * wav2spec.hop))   # missing channel dim
    with pytest.raises(ValueError):
        wav2spec(torch.randn(2, 2, 10 * wav2spec.hop))  # stereo


def test_wav2spec_rejects_nonhop_divisible_length(wav2spec):
    """Input length not a multiple of hop → clear ValueError. This is the
    guard that surfaces an audio_fs/dt_ms mismatch between the wav2spec and
    its dataset (e.g. SincNet(audio_fs=16000) against a 48 kHz dataset)."""
    x = torch.randn(2, 1, 50 * wav2spec.hop + 1)
    with pytest.raises(ValueError, match="multiple of hop"):
        wav2spec(x)
