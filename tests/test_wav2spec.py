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
]


@pytest.fixture(params=WAV2SPEC_CASES, ids=[c[0] for c in WAV2SPEC_CASES])
def wav2spec(request):
    return request.param[1]()


def test_factory_dispatch():
    from deepSTRF.models.wav2spec import CausalMelSpectrogram, make_wav2spec

    m = make_wav2spec("mel", audio_fs=16000, dt_ms=5.0)
    assert isinstance(m, CausalMelSpectrogram)
    assert m.audio_fs == 16000
    assert m.hop == 80
    assert m.out_channels == 34

    with pytest.raises(ValueError):
        make_wav2spec("not-a-real-frontend", audio_fs=16000, dt_ms=5.0)


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
