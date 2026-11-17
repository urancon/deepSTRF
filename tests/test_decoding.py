"""Tests for the decoding API: decoders, decoding metrics, Fitter(direction='decoding')."""
from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.metrics import mel_cepstral_distortion, spectrogram_corrcoef
from deepSTRF.metrics.decoding import MCD_DB_SCALE, _dct2_ortho, mel_filterbank
from deepSTRF.models.audio import MeanTargetBaseline, WienerFilter
from deepSTRF.models.audio.decoding import collapse_repeats, valid_time_mask
from deepSTRF.training import Fitter
from deepSTRF.utils import neural_collate

N, FB, L = 5, 4, 3


class _ToyDecodingDataset(AudioNeuralDataset):
    """Spikes drive a spectrogram through a known causal linear filter."""

    direction = "decoding"

    def __init__(self, n_items=12, T=60, lengths=None, seed=0, noise=0.0):
        super().__init__("toy", dt_ms=5.0)
        g = torch.Generator().manual_seed(1234)       # same planted filter for every seed
        self.W = torch.randn(FB, N, L, generator=g)
        self.b = torch.randn(FB, generator=g)
        g = torch.Generator().manual_seed(seed)
        self.F = FB
        for i in range(n_items):
            t = T if lengths is None else lengths[i]
            x = torch.poisson(torch.full((N, t), 2.0), generator=g)
            xp = torch.nn.functional.pad(x, (L - 1, 0))
            y = torch.stack([sum((self.W[:, n, k:k + 1] * xp[n, k:k + t]) for n in range(N))
                             for k in range(L)]).sum(0) + self.b[:, None]
            y = y + noise * torch.randn(y.shape, generator=g)
            self.stims.append(y.unsqueeze(0))                         # (1, F, T)
            self.responses.append([x[n:n + 1] for n in range(N)])     # (1, T) each
            self.stim_meta.append({"name": f"item{i}"})
        self.nrn_meta = [{"cell_id": f"n{n}"} for n in range(N)]
        self.N_neurons = N
        self.validate()


def _loader(ds, bs=4, shuffle=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=neural_collate)


# ------------------------------------------------------------------ helpers

def test_collapse_repeats_nan_aware():
    r = torch.tensor([[[[1.0, 2.0], [3.0, float("nan")]],
                       [[float("nan"), float("nan")], [float("nan"), float("nan")]]]])  # (1,2,2,2)
    x = collapse_repeats(r)
    assert torch.allclose(x, torch.tensor([[[2.0, 2.0], [0.0, 0.0]]]))
    assert valid_time_mask(r).tolist() == [[True, True]]


# ------------------------------------------------------------------ WienerFilter

def test_wiener_shapes_and_direction():
    m = WienerFilter(N, FB, n_lags=L)
    assert m.direction == "decoding" and m.O == FB
    y = m(torch.rand(2, N, 1, 30))
    assert y.shape == (2, 1, FB, 30)


def test_wiener_decode_matches_explicit_lag_sum():
    torch.manual_seed(0)
    m = WienerFilter(N, FB, n_lags=L, lag_offset=1)
    x = torch.rand(1, N, 20)
    y = m.decode(x)[0, 0]
    t = 10
    expect = m.readout.bias.clone()
    for j in range(L):                   # kernel index k = L-1-j  <->  lag offset + j
        expect += m.readout.weight[:, :, L - 1 - j] @ x[0, :, t - 1 - j]
    assert torch.allclose(y[:, t], expect, atol=1e-5)


@pytest.mark.parametrize("offset", [0, 2])
def test_wiener_is_causal_for_nonnegative_offset(offset):
    m = WienerFilter(N, FB, n_lags=L, lag_offset=offset).eval()
    x = torch.rand(1, N, 1, 40)
    x2 = x.clone()
    x2[..., 25:] += 5.0
    y, y2 = m(x), m(x2)
    assert torch.allclose(y[..., :25 + offset], y2[..., :25 + offset])
    assert not torch.allclose(y[..., 25 + offset:], y2[..., 25 + offset:])


def test_wiener_negative_offset_reads_the_future():
    m = WienerFilter(N, FB, n_lags=L, lag_offset=-(L - 1)).eval()
    x = torch.rand(1, N, 1, 40)
    x2 = x.clone()
    x2[..., 25] += 5.0
    d = (m(x) - m(x2)).abs().sum(dim=(0, 1, 2))
    assert torch.all(d[23:26] > 0) and torch.all(d[:23] == 0) and torch.all(d[26:] == 0)


def test_wiener_closed_form_recovers_planted_filter():
    ds = _ToyDecodingDataset(n_items=10, T=80, noise=0.01)
    m = WienerFilter(N, FB, n_lags=L).fit(_loader(ds), alpha=1e-6)
    assert torch.allclose(m.readout.weight, ds.W, atol=1e-2)
    assert torch.allclose(m.readout.bias, ds.b, atol=5e-2)


def test_wiener_fit_ignores_padded_frames():
    ds = _ToyDecodingDataset(n_items=8, lengths=[40, 60, 50, 80, 45, 70, 55, 65])
    m = WienerFilter(N, FB, n_lags=L).fit(_loader(ds, bs=4), alpha=1e-6)
    assert torch.allclose(m.readout.weight, ds.W, atol=1e-3)


# ------------------------------------------------------------------ MeanTargetBaseline

def test_mean_target_baseline_is_input_independent_template():
    ds = _ToyDecodingDataset(n_items=6, T=30)
    m = MeanTargetBaseline(N, FB).fit(_loader(ds))
    tpl = torch.stack([s[0] for s in ds.stims]).mean(0)
    assert torch.allclose(m.template, tpl, atol=1e-5)
    a, b = m(torch.rand(2, N, 1, 30)), m(torch.zeros(2, N, 1, 30))
    assert a.shape == (2, 1, FB, 30) and torch.equal(a, b)
    assert m(torch.rand(1, N, 1, 35)).shape[-1] == 35            # edge-padded


# ------------------------------------------------------------------ metrics

def test_spectrogram_corrcoef_per_band_and_nan_frames():
    g = torch.randn(3, 1, FB, 50)
    assert torch.allclose(spectrogram_corrcoef(g, g, reduction="none"), torch.ones(FB))
    p = g.clone()
    p[:, :, 0] = -p[:, :, 0]
    r = spectrogram_corrcoef(p, g, reduction="none")
    assert torch.isclose(r[0], torch.tensor(-1.0)) and torch.allclose(r[1:], torch.ones(FB - 1))
    g2 = g.clone()
    g2[..., 40:] = float("nan")
    p2 = g.clone()
    p2[..., 40:] = 1e6                                             # garbage on masked frames
    assert torch.isclose(spectrogram_corrcoef(p2, g2), torch.tensor(1.0))


def test_mcd_zero_identical_and_ignores_loudness_offset():
    g = torch.randn(2, 1, 20, 30)
    assert mel_cepstral_distortion(g, g) == 0
    assert mel_cepstral_distortion(g + 3.0, g) < 1e-4             # constant -> c0 only


def test_mcd_known_value_from_one_cepstral_coefficient():
    D = _dct2_ortho(20, 20)
    g = torch.zeros(1, 1, 20, 5)
    p = g + 0.5 * D[2][None, None, :, None]                       # c2 differs by 0.5
    assert torch.isclose(mel_cepstral_distortion(p, g), torch.tensor(0.5 * MCD_DB_SCALE), atol=1e-5)
    assert torch.isclose(mel_cepstral_distortion(p, g, alpha=1.0), torch.tensor(0.5), atol=1e-6)


def test_mcd_with_linear_frequency_input():
    freqs = torch.linspace(300, 8000, 158)
    g = torch.randn(2, 1, 158, 25)
    assert mel_cepstral_distortion(g, g, freqs_hz=freqs, n_mels=32) == 0
    assert mel_cepstral_distortion(g + torch.randn_like(g), g, freqs_hz=freqs) > 0
    fb = mel_filterbank(freqs, 32)
    assert fb.shape == (32, 158) and (fb.sum(1) > 0).all()
    with pytest.raises(ValueError):
        mel_filterbank(freqs, 400)


def test_mcd_frame_mask_excludes_nan_frames():
    g = torch.randn(1, 1, 20, 30)
    p = g.clone()
    p[..., 20:] += torch.randn(1, 1, 20, 10)
    g2 = g.clone()
    g2[..., 20:] = float("nan")
    assert mel_cepstral_distortion(p, g2) == 0


# ------------------------------------------------------------------ Fitter

def test_fitter_reads_decoding_direction_and_masks_padding():
    ds = _ToyDecodingDataset(n_items=4, lengths=[20, 30, 25, 30])
    f = Fitter(WienerFilter(N, FB, n_lags=L), _loader(ds), _loader(ds), max_epochs=1,
               log_fn=lambda d: None)
    assert f.direction == "decoding" and f.monitor == "val_cc"
    batch = next(iter(_loader(ds)))
    model_input, target = f.unpack_batch(batch)
    assert model_input.shape == (4, N, 1, 30) and target.shape == (4, FB, 1, 30)
    assert target[0, :, :, 20:].isnan().all() and not target[0, :, :, :20].isnan().any()
    assert f.predict(model_input).shape == target.shape


def test_fitter_trains_decoder_by_gradient_descent():
    torch.manual_seed(0)
    train, val = _ToyDecodingDataset(n_items=16, seed=1), _ToyDecodingDataset(n_items=4, seed=2)
    m = WienerFilter(N, FB, n_lags=L)
    f = Fitter(m, _loader(train, shuffle=True), _loader(val), max_epochs=200, patience=20,
               optimizer=torch.optim.Adam(m.parameters(), lr=5e-2), log_fn=lambda d: None)
    hist = f.fit()
    assert hist[0]["val_cc"].shape == (FB,)
    assert torch.nanmean(f.evaluate(_loader(val))["cc"]) > 0.99


def test_fitter_encoding_default_unchanged():
    from deepSTRF.models.audio import Linear
    m = Linear(n_frequency_bands=FB, temporal_window_size=3, out_neurons=N)
    ds = _ToyDecodingDataset(n_items=2)
    f = Fitter(m, _loader(ds), _loader(ds), max_epochs=1, log_fn=lambda d: None)
    assert f.direction == "encoding" and f.monitor == "val_cc_norm"
    with pytest.raises(ValueError):
        Fitter(m, _loader(ds), _loader(ds), direction="sideways")
