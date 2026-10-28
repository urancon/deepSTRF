"""Shape + causality + stride-factor tests for ``ICNet``."""
from __future__ import annotations

import pytest
import torch


def test_icnet_factor_into_strides():
    from deepSTRF.models.audio.icnet import _factor_into_strides

    # Paper config: 32 = 2**5 samples / bin -> uniform stride 2
    assert _factor_into_strides(32, 5) == [2, 2, 2, 2, 2]
    # Balanced factorisation (largest first): 80, 240, 160 samples / bin
    assert _factor_into_strides(80, 5) == [5, 2, 2, 2, 2]
    assert _factor_into_strides(240, 5) == [5, 4, 3, 2, 2]
    assert _factor_into_strides(160, 5) == [5, 4, 2, 2, 2]
    # Regression: a NON-2-smooth total must NOT collapse to one huge tail stride.
    # 44.1 kHz / 10 ms -> 441 = 3**2 * 7**2; the old greedy split gave [1,1,1,1,441]
    # (four full-rate conv layers + a stride-441/kernel-64 conv skipping most input).
    assert _factor_into_strides(441, 5) == [7, 7, 3, 3, 1]
    assert _factor_into_strides(960, 3) == [12, 10, 8]     # ac1_mgb S-tier
    # Odd total, single layer covers it
    assert _factor_into_strides(7, 1) == [7]

    # Invariants for arbitrary (total, n_layers): product matches, length matches,
    # descending order, and every stride stays under the encoder kernel size (64).
    for total, nl in [(32, 5), (80, 5), (240, 5), (441, 5), (240, 3),
                      (960, 3), (120, 5), (1, 3), (513, 4)]:
        s = _factor_into_strides(total, nl)
        prod = 1
        for x in s:
            prod *= x
        assert prod == total and len(s) == nl and s == sorted(s, reverse=True)

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
    assert encoder.encoder_strides == [5, 2, 2, 2, 2]   # balanced factorisation of 80
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


def test_icnet_kernel_sizes_fixed_duration():
    """Kernels sized in ms: receptive field <= the target, and the same layer durations
    at any sample rate."""
    from deepSTRF.models.audio.icnet import icnet_kernel_sizes

    # 44.1 kHz / 10 ms (Espejo, NAT4, Wingert) and 48 kHz / 5 ms (NS1)
    assert icnet_kernel_sizes(44100, [7, 7, 3, 3, 1])[:2] == (189, [186, 53, 15, 10, 7, 13])
    assert icnet_kernel_sizes(48000, [5, 4, 3, 2, 2])[:2] == (194, [191, 76, 38, 26, 26, 26])
    for fs, st in [(16000, [5, 4, 2, 2, 2]), (24000, [5, 4, 3, 2, 2]), (44100, [7, 7, 3, 3, 1])]:
        k_sinc, k_enc, rf = icnet_kernel_sizes(fs, st, receptive_field_ms=250.0)
        assert 230.0 < rf <= 250.0
        assert len(k_enc) == len(st) + 1
        assert k_sinc >= 2 and all(k >= s for k, s in zip(k_enc, st))
        # the first conv lasts about as long at every rate (paper: 64 taps at 24 414 Hz)
        assert abs(k_enc[0] / fs * 1000 - 4.2) < 0.3
    # a longer target gives longer kernels
    assert icnet_kernel_sizes(44100, [7, 7, 3, 3, 1], 500.0)[2] > 400.0


def test_icnet_defaults_norm_and_ms_kernels():
    """Default: CausalLayerNorm after each conv and kernels from receptive_field_ms;
    norm=None, receptive_field_ms=None gives the paper's structure."""
    from deepSTRF.models.audio import ICNet
    from deepSTRF.models.audio.icnet import icnet_kernel_sizes
    from deepSTRF.models.layers import CausalLayerNorm

    m = ICNet(audio_fs=44100, out_neurons=3, dt_ms=10.0)
    enc = m.wav2spec
    k_sinc, k_enc, _ = icnet_kernel_sizes(44100, enc.encoder_strides)
    assert enc.sincnet_kernel_size == k_sinc and enc.encoder_kernel_sizes == k_enc
    assert enc.sincnet.kernel_size == k_sinc
    assert [b.conv.kernel_size[0] for b in [*enc.encoder, enc.bottleneck]] == k_enc
    assert all(isinstance(b.norm, CausalLayerNorm) for b in [*enc.encoder, enc.bottleneck])

    p = ICNet(audio_fs=44100, out_neurons=3, dt_ms=10.0, norm=None, receptive_field_ms=None)
    assert p.wav2spec.sincnet_kernel_size == 64 and p.wav2spec.encoder_kernel_sizes == [64] * 6
    assert all(isinstance(b.norm, torch.nn.Identity)
               for b in [*p.wav2spec.encoder, p.wav2spec.bottleneck])

    # explicit kernels win over receptive_field_ms
    e = ICNet(audio_fs=16000, out_neurons=3, dt_ms=5.0, sincnet_kernel_size=31,
              encoder_kernel_size=[9, 8, 7, 6, 5, 4])
    assert e.wav2spec.sincnet_kernel_size == 31
    assert e.wav2spec.encoder_kernel_sizes == [9, 8, 7, 6, 5, 4]
    with pytest.raises(ValueError):
        ICNet(audio_fs=16000, out_neurons=3, dt_ms=5.0, encoder_kernel_size=[9, 8])
    with pytest.raises(ValueError):
        ICNet(audio_fs=16000, out_neurons=3, dt_ms=5.0, norm="batch")



def test_icnet_init_rate():
    """The initial prediction sits at init_rate (softplus of the decoder bias)."""
    import torch.nn.functional as F
    from deepSTRF.models.audio import ICNet

    m = ICNet(audio_fs=16000, out_neurons=3, dt_ms=5.0, init_rate=0.02)
    assert torch.allclose(F.softplus(m.decoder.bias), torch.full((3,), 0.02), atol=1e-6)
    m = ICNet(audio_fs=16000, out_neurons=3, dt_ms=5.0)
    assert torch.allclose(F.softplus(m.decoder.bias), torch.full((3,), 0.05), atol=1e-6)
    with pytest.raises(ValueError):
        ICNet(audio_fs=16000, out_neurons=3, dt_ms=5.0, init_rate=0.0)
