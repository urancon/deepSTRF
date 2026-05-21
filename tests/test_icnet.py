"""Shape + causality + stride-factor tests for ``ICNet``."""
from __future__ import annotations

import pytest
import torch


def test_icnet_factor_into_strides():
    from deepSTRF.models.audio.icnet import _factor_into_strides

    # Paper config: 32 samples / bin
    assert _factor_into_strides(32, 5) == [2, 2, 2, 2, 2]
    # NS1 16 kHz / 5 ms config (kept for regression): 80 samples / bin
    assert _factor_into_strides(80, 5) == [2, 2, 2, 2, 5]
    # NS1 48 kHz / 5 ms (new default): 240 samples / bin
    assert _factor_into_strides(240, 5) == [2, 2, 2, 2, 15]
    # NAT4-style config: 10 ms at 16 kHz → 160 samples / bin
    assert _factor_into_strides(160, 5) == [2, 2, 2, 2, 10]
    # Odd total, can't peel any 2s — single layer covers it
    assert _factor_into_strides(7, 1) == [7]

    with pytest.raises(ValueError):
        _factor_into_strides(0, 5)
    with pytest.raises(ValueError):
        _factor_into_strides(80, 0)


def test_icnet_encoder_shape_and_out_channels():
    """ICNet's internal encoder (wav2spec slot value) produces a 64-channel
    latent at the neural rate."""
    from deepSTRF.models.audio import ICNet

    m = ICNet(audio_fs=16000, out_neurons=5, dt_ms=5.0)
    encoder = m.wav2spec
    assert encoder.out_channels == 64
    assert encoder.encoder_strides == [2, 2, 2, 2, 5]
    x = torch.randn(2, 1, 999 * 80) * 0.1
    y = encoder(x)
    assert y.shape == (2, 1, 64, 999)


def test_icnet_encoder_paper_config():
    """Paper-faithful config: 24 414 Hz / 1.31 ms → strides [2,2,2,2,2]."""
    from deepSTRF.models.audio import ICNet

    dt_paper = 32 * 1000 / 24414
    m = ICNet(audio_fs=24414, out_neurons=4, dt_ms=dt_paper)
    assert m.wav2spec.encoder_strides == [2, 2, 2, 2, 2]


def test_icnet_encoder_custom_strides():
    from deepSTRF.models.audio import ICNet

    m = ICNet(audio_fs=16000, out_neurons=5, dt_ms=5.0,
              encoder_strides=[5, 4, 2, 1, 2])  # = 80
    assert m.wav2spec.encoder_strides == [5, 4, 2, 1, 2]

    # mismatch must raise
    with pytest.raises(ValueError):
        ICNet(audio_fs=16000, out_neurons=5, dt_ms=5.0,
              encoder_strides=[2, 2, 2, 2, 4])  # 64 != 80


def test_icnet_forward_shape_and_nonnegative():
    """ICNet output is ``(B, N, 1, T_neural)`` and non-negative (softplus)."""
    from deepSTRF.models.audio import ICNet

    m = ICNet(audio_fs=16000, out_neurons=10, dt_ms=5.0)
    x = torch.randn(2, 1, 100 * 80) * 0.1
    y = m(x)
    assert y.shape == (2, 10, 1, 100)
    assert (y >= 0).all()
    m.validate()  # must not raise


def test_icnet_strict_causality():
    """Perturbing audio after neural bin K must not move output frames ≤ K."""
    from deepSTRF.models.audio import ICNet

    torch.manual_seed(0)
    m = ICNet(audio_fs=16000, out_neurons=10, dt_ms=5.0)
    m.eval()
    K = 30
    T_neural = 100
    x = torch.randn(1, 1, T_neural * 80) * 0.1
    x_pert = x.clone()
    cut_audio = (K + 1) * 80
    x_pert[..., cut_audio:] = torch.randn_like(x_pert[..., cut_audio:]) * 0.5
    with torch.no_grad():
        y = m(x)
        y_pert = m(x_pert)
    diff = (y - y_pert)[..., :K + 1].abs().max().item()
    assert diff < 1e-5, f"ICNet causality violation: max diff = {diff:.3e}"


def test_icnet_gradient_flow():
    """Gradient reaches all encoder parameters."""
    from deepSTRF.models.audio import ICNet

    m = ICNet(audio_fs=16000, out_neurons=10, dt_ms=5.0)
    x = torch.randn(2, 1, 100 * 80) * 0.1
    y = m(x)
    y.mean().backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.abs().sum().item() > 0, f"{name} grad is zero"


def test_icnet_strf_gradmap_still_works():
    """STRF_gradmap returns a sensible-shape tensor even though ICNet's
    'spectrogram' is really a latent representation."""
    from deepSTRF.models.audio import ICNet

    m = ICNet(audio_fs=16000, out_neurons=5, dt_ms=5.0)
    gm = m.STRF_gradmap(T=1)
    # Shape (N, 1, F=bottleneck_channels=64, T=1)
    assert gm.shape == (5, 1, 64, 1)


def test_icnet_not_exposed_in_wav2spec_namespace():
    """ICNet's encoder is module-private — not exported by
    ``deepSTRF.models.wav2spec``."""
    import deepSTRF.models.wav2spec as w
    assert not hasattr(w, "ICNetFrontend")
    assert not hasattr(w, "_ICNetEncoder")
