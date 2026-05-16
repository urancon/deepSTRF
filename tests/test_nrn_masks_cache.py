"""Tests for the lazy-cache + invalidation behavior of ``NeuralDataset.nrn_masks``.

The mask is derived from the ``(1, 1)`` NaN-sentinel positions in
``self.responses`` (see data_paradigm.md §3.1). Computation is now
cached on first access; structural mutations to ``self.responses``
require ``_invalidate_nrn_masks()``.
"""

from __future__ import annotations

import time

import torch


def _fake_audio_dataset(N: int, S: int, T: int = 8, F: int = 4):
    """Minimal concrete AudioNeuralDataset, mirroring tests/test_concat.py."""
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass

    ds = _FakeAudio(path="/tmp/nowhere", dt_ms=1.0)
    ds.F = F
    ds.N_neurons = N
    ds.stim_meta = [{"name": f"s{i}", "type": "synthetic"} for i in range(S)]
    ds.stims = [torch.zeros(1, F, T) for _ in range(S)]
    ds.responses = [[torch.ones(3, T) * (s + 1) for _ in range(N)] for s in range(S)]
    ds.neuron_metadata = [{"uid": f"n{i}"} for i in range(N)]
    ds.validate()
    return ds


# ============================================================
# Correctness
# ============================================================

def test_nrn_masks_all_real_data():
    ds = _fake_audio_dataset(N=4, S=3)
    masks = ds.nrn_masks
    assert masks.shape == (3, 4)
    assert masks.dtype == torch.bool
    assert masks.all()  # all real responses → all True


def test_nrn_masks_with_sentinel():
    ds = _fake_audio_dataset(N=3, S=2)
    # mark (stim 0, cell 1) as missing via the canonical (1, 1) NaN sentinel
    ds.responses[0][1] = torch.full((1, 1), float("nan"))
    ds._invalidate_nrn_masks()
    masks = ds.nrn_masks
    assert masks.shape == (2, 3)
    assert bool(masks[0, 0]) is True
    assert bool(masks[0, 1]) is False, "sentinel cell must be False"
    assert bool(masks[0, 2]) is True
    assert masks[1].all()


def test_nrn_masks_with_nan_in_non_sentinel():
    """Defensive: a non-(1,1) tensor that contains NaN is still flagged missing."""
    ds = _fake_audio_dataset(N=2, S=2, T=8)
    bad = torch.ones(3, 8)
    bad[0, 4] = float("nan")
    ds.responses[1][0] = bad
    ds._invalidate_nrn_masks()
    masks = ds.nrn_masks
    assert bool(masks[1, 0]) is False
    assert bool(masks[1, 1]) is True


def test_nrn_masks_empty_dataset():
    """A bare instance (no populated responses) must return (0, N) bool."""
    # validate() requires S > 0, so we skip it and build the bare-bones case
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass
    bare = _FakeAudio(path="/tmp/nowhere", dt_ms=1.0)
    bare.F = 4
    bare.N_neurons = 3
    masks = bare.nrn_masks
    assert masks.shape == (0, 3)
    assert masks.dtype == torch.bool


# ============================================================
# Caching + invalidation
# ============================================================

def test_nrn_masks_returns_cached_object():
    """Second access must return the same tensor object (no recomputation)."""
    ds = _fake_audio_dataset(N=2, S=2)
    a = ds.nrn_masks
    b = ds.nrn_masks
    assert a is b, "second access should return the cached tensor object"


def test_invalidate_nrn_masks_forces_recompute():
    ds = _fake_audio_dataset(N=2, S=2)
    a = ds.nrn_masks
    # mutate a response to introduce a sentinel
    ds.responses[0][0] = torch.full((1, 1), float("nan"))
    # without invalidation, the cache is stale (this is the contract)
    stale = ds.nrn_masks
    assert stale is a, "no invalidation -> still cached"
    # now invalidate and the next read reflects the mutation
    ds._invalidate_nrn_masks()
    fresh = ds.nrn_masks
    assert fresh is not a
    assert bool(fresh[0, 0]) is False
    assert bool(stale[0, 0]) is True  # the snapshot still shows the old value


def test_smooth_responses_does_not_break_cache():
    """``smooth_responses`` preserves shapes -> the cached mask stays valid."""
    ds = _fake_audio_dataset(N=2, S=2, T=64)
    before = ds.nrn_masks.clone()
    ds.smooth_responses(window_ms=5.0)
    after = ds.nrn_masks
    # cache may or may not have been invalidated by smooth_responses;
    # what matters is that the *content* is still correct.
    assert torch.equal(after, before)


# ============================================================
# Performance guard — caching must give a meaningful speedup at scale
# ============================================================

def test_nrn_masks_caching_is_faster_than_recompute():
    """Cached access must be much faster than recomputation at modest scale.

    Espejo NAT post-AMT-filter is roughly (S=580, N=170). At that scale
    the naive recompute was ~1 s; caching brings it to microseconds.
    Use S=200 / N=100 here to keep the test sub-second on CI.
    """
    ds = _fake_audio_dataset(N=100, S=200, T=16)
    # warm the cache
    ds.nrn_masks

    # cached access
    t0 = time.perf_counter()
    for _ in range(100):
        _ = ds.nrn_masks
    cached_dt = time.perf_counter() - t0

    # full recompute (invalidate each time)
    t0 = time.perf_counter()
    for _ in range(10):
        ds._invalidate_nrn_masks()
        _ = ds.nrn_masks
    recompute_dt = (time.perf_counter() - t0) / 10  # per-call

    cached_per_call = cached_dt / 100
    # require at least 100x speedup — the cache is just a None check + return
    assert cached_per_call * 100 < recompute_dt, (
        f"caching must give >=100x speedup (got cached={cached_per_call*1e6:.1f}us, "
        f"recompute={recompute_dt*1e3:.1f}ms)"
    )
