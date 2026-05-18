"""Tests for ``NeuralDataset.compute_neuron_quality``.

Verifies that the opt-in method writes Sahani-Linden ``'snr'`` and
Hsu/Spearman-Brown ``'ccmax'`` scalars into ``nrn_meta``, with the
R=1 fallback semantics agreed in the design discussion:

- Per-stim contribution requires R >= 2 and T >= 2.
- A neuron with **only** R=1 stims gets ``ccmax=1.0`` (no normalization
  possible) and ``snr=NaN`` (no noise estimate available).
- The numbers match what calling :func:`deepSTRF.metrics.snr` and the
  CCmax helper directly would give on the same stacked tensor.
"""

from __future__ import annotations

import math

import torch


def _fake_audio_dataset(N: int, S: int, T: int, R: int = 3, F: int = 4, seed: int = 0):
    """Minimal concrete AudioNeuralDataset for self-contained tests."""
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass

    ds = _FakeAudio(path="/tmp/nowhere", dt_ms=1.0)
    ds.F = F
    ds.N_neurons = N
    ds.stim_meta = [{"name": f"s{i}", "type": "synthetic"} for i in range(S)]
    ds.stims = [torch.zeros(1, F, T) for _ in range(S)]
    g = torch.Generator().manual_seed(seed)
    # Each (s, n) gets a (R, T) tensor with a per-stim signal + per-trial noise.
    ds.responses = []
    for s in range(S):
        row = []
        for n in range(N):
            sig = torch.sin(torch.linspace(0, 2 * math.pi * (1 + 0.3 * n), T))
            noise = 0.4 * torch.randn(R, T, generator=g)
            row.append(sig.unsqueeze(0) + noise)
        ds.responses.append(row)
    ds.nrn_meta = [{"uid": f"n{i}"} for i in range(N)]
    ds.validate()
    return ds


def test_compute_neuron_quality_writes_both_keys():
    ds = _fake_audio_dataset(N=3, S=4, T=20, R=4)
    assert "snr" not in ds.nrn_meta[0]
    assert "ccmax" not in ds.nrn_meta[0]
    out = ds.compute_neuron_quality()
    assert set(out.keys()) == {"snr", "ccmax"}
    assert out["snr"].shape == (3,)
    assert out["ccmax"].shape == (3,)
    for m in ds.nrn_meta:
        assert "snr" in m and isinstance(m["snr"], float)
        assert "ccmax" in m and isinstance(m["ccmax"], float)


def test_compute_neuron_quality_values_are_finite_and_sensible():
    """Synthetic signal + noise → finite SP/NP, CCmax in (0, 1]."""
    ds = _fake_audio_dataset(N=4, S=6, T=40, R=5, seed=42)
    out = ds.compute_neuron_quality()
    assert torch.isfinite(out["snr"]).all(), "snr should be finite for this clean synthetic data"
    assert torch.isfinite(out["ccmax"]).all(), "ccmax should be finite (R=5, signal > noise)"
    # CCmax is bounded by 1 (Spearman-Brown noise ceiling)
    assert (out["ccmax"] <= 1.0 + 1e-6).all()
    assert (out["ccmax"] > 0.0).all()
    # SNR should be positive
    assert (out["snr"] > 0).all()


def test_compute_neuron_quality_matches_direct_metrics_call():
    """The dataset-level method must agree numerically with calling
    ``deepSTRF.metrics.snr`` directly on the same stacked tensor."""
    from deepSTRF.metrics.performance import (
        _ccmax_per_neuron,
        _sahani_linden_per_neuron,
    )

    ds = _fake_audio_dataset(N=3, S=5, T=30, R=4, seed=7)

    # Build the same stacked tensor the method builds internally.
    S, N = len(ds.responses), ds.N_neurons
    R_max = max(r.shape[0] for row in ds.responses for r in row)
    T_max = max(r.shape[1] for row in ds.responses for r in row)
    stacked = torch.full((S, N, R_max, T_max), float("nan"))
    for s in range(S):
        for n in range(N):
            r = ds.responses[s][n]
            stacked[s, n, : r.shape[0], : r.shape[1]] = r
    valid = ~torch.isnan(stacked)

    sp, np_ = _sahani_linden_per_neuron(stacked, valid)
    expected_snr = sp / np_.clamp(min=1e-12)
    expected_ccmax = _ccmax_per_neuron(stacked, valid, max_iters=126)

    got = ds.compute_neuron_quality()
    assert torch.allclose(got["snr"], expected_snr, equal_nan=True)
    assert torch.allclose(got["ccmax"], expected_ccmax, equal_nan=True)


def test_compute_neuron_quality_r1_only_neuron_gets_ccmax_one():
    """A neuron with R=1 on every stim has no noise-estimation power.

    Contract: ``ccmax = 1.0`` (sentinel: no normalization), ``snr = NaN``.
    """
    ds = _fake_audio_dataset(N=2, S=3, T=20, R=3, seed=0)
    # rewrite neuron 0 so every (s, 0) has R=1 (a single repeat per stim)
    for s in range(3):
        ds.responses[s][0] = ds.responses[s][0][:1]
    ds._invalidate_nrn_masks()

    out = ds.compute_neuron_quality()
    assert math.isnan(out["snr"][0].item()), "snr undefined when no stim has R>=2"
    assert out["ccmax"][0].item() == 1.0, "ccmax=1.0 sentinel when no stim has R>=2"
    # neuron 1 still has R=3 stims → both values finite
    assert torch.isfinite(out["snr"][1])
    assert torch.isfinite(out["ccmax"][1])


def test_compute_neuron_quality_missing_neuron_gets_nan():
    """A neuron with NO real data (all (1,1) NaN sentinels) gets NaN/NaN."""
    ds = _fake_audio_dataset(N=2, S=3, T=20, R=3, seed=0)
    # mark neuron 1 as missing on every stim
    for s in range(3):
        ds.responses[s][1] = torch.full((1, 1), float("nan"))
    ds._invalidate_nrn_masks()

    out = ds.compute_neuron_quality()
    assert torch.isfinite(out["snr"][0])
    assert torch.isfinite(out["ccmax"][0])
    assert math.isnan(out["snr"][1].item())
    # All stims contribute zero R>=2 data, so the R=1 fallback also fires → 1.0.
    # That's the documented contract: "neuron with zero R>=2 stims" includes
    # the all-sentinel case. Filter by snr=NaN to exclude these.
    assert out["ccmax"][1].item() == 1.0


def test_compute_neuron_quality_pairs_with_predicate_filter():
    """End-to-end: write quality keys, then filter by them."""
    ds = _fake_audio_dataset(N=4, S=5, T=30, R=4, seed=1)
    ds.compute_neuron_quality()
    # Every neuron has SP > 0 here, but using a high threshold still gives
    # a deterministic non-empty filter result we can sanity-check.
    sel = ds.select_pop_by_nrn_predicate(lambda n: n["snr"] > 0)
    assert sel, "synthetic dataset should have some neurons above snr=0"
    # And one that should reject everyone
    sel = ds.select_pop_by_nrn_predicate(lambda n: n["snr"] > 1e9)
    assert sel == []
