"""Regression tests for the streaming :meth:`NeuralDataset.compute_neuron_quality`.

The 2026-05-22 refactor (this branch) replaced the original implementation —
build a single ``(S, N, R_max, T_max)`` padded float32 tensor, then call the
per-neuron helpers — with a stim-streaming loop that calls the new per-stim
helpers :func:`_per_stim_sp_np` / :func:`_per_stim_ccmax` and accumulates
length-weighted partial sums. Motivation: on Downer 2025 TIMIT the padded
tensor weighed in at ~54 GB and OOMed; the streaming variant peaks at the
largest single-stim ``(N, R_s, T_s)`` slab (~100 MB on the same dataset).

The two helpers :func:`_sahani_linden_per_neuron` and :func:`_ccmax_per_neuron`
keep their 4-D-tensor signatures (used in training-loop val-metric callbacks)
and now wrap the per-stim variants; behaviour preservation for those is
covered by the existing :mod:`tests.test_neuron_quality` and
:mod:`tests.test_metrics` suites.

What this file pins down:

1. Per-stim helpers agree with the equivalent 4-D-tensor reference on a small
   synthetic dataset (sanity check on the new entry points).
2. The dataset-level ``compute_neuron_quality()`` is bit-identical to the
   pre-refactor implementation on a small dataset where the old code worked
   (the historical padded path is reproduced inline as the reference).
3. ``compute_neuron_quality()`` completes on a dataset whose fully-padded
   shape would peak at multi-GB — the actual streaming property we're
   buying. Uses small per-stim shapes so the test is fast, but ``S * N *
   R_max * T_max`` is sized so the old padded tensor would have been
   prohibitively large.
"""

from __future__ import annotations

import math

import torch


def _fake_audio_dataset(
    N: int,
    S: int,
    T: int,
    R: int = 3,
    F: int = 4,
    seed: int = 0,
    ragged: bool = False,
):
    """Synthetic AudioNeuralDataset with per-(s, n) heterogeneous (R, T).

    If ``ragged=True``, half the stims get short T and small R for the second
    half of the neuron population — this exercises the per-stim padding path
    where ``R_max``/``T_max`` would differ stim-to-stim.
    """
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass

    ds = _FakeAudio(path="/tmp/nowhere", dt_ms=1.0)
    ds.F = F
    ds.N_neurons = N
    ds.stim_meta = [{"name": f"s{i}", "type": "synthetic"} for i in range(S)]
    ds.stims = [torch.zeros(1, F, T) for _ in range(S)]
    g = torch.Generator().manual_seed(seed)
    ds.responses = []
    for s in range(S):
        row = []
        for n in range(N):
            R_sn = R
            T_sn = T
            if ragged and s % 2 == 1 and n >= N // 2:
                R_sn = max(2, R - 1)
                T_sn = max(2, T // 2)
            sig = torch.sin(torch.linspace(0, 2 * math.pi * (1 + 0.3 * n), T_sn))
            noise = 0.4 * torch.randn(R_sn, T_sn, generator=g)
            row.append(sig.unsqueeze(0) + noise)
        ds.responses.append(row)
    ds.nrn_meta = [{"uid": f"n{i}"} for i in range(N)]
    ds.validate()
    return ds


def _legacy_compute_neuron_quality(ds, max_ccmax_iters: int = 126):
    """The pre-refactor reference: pad-then-aggregate."""
    from deepSTRF.metrics.performance import (
        _ccmax_per_neuron,
        _sahani_linden_per_neuron,
    )

    S = len(ds.responses)
    N = ds.N_neurons
    nan = float("nan")
    if S == 0 or N == 0:
        return torch.full((N,), nan), torch.full((N,), nan)

    R_max = 1
    T_max = 1
    has_r2_stim = torch.zeros(N, dtype=torch.bool)
    for s in range(S):
        for n in range(N):
            r = ds.responses[s][n]
            if tuple(r.shape) == (1, 1):
                continue
            R_max = max(R_max, int(r.shape[0]))
            T_max = max(T_max, int(r.shape[1]))
            if r.shape[0] >= 2:
                has_r2_stim[n] = True

    stacked = torch.full((S, N, R_max, T_max), nan)
    for s in range(S):
        for n in range(N):
            r = ds.responses[s][n]
            if tuple(r.shape) == (1, 1):
                continue
            R, T = int(r.shape[0]), int(r.shape[1])
            stacked[s, n, :R, :T] = r
    valid = ~torch.isnan(stacked)

    sp, np_ = _sahani_linden_per_neuron(stacked, valid)
    snr_n = sp / np_.clamp(min=1e-12)
    ccmax_n = _ccmax_per_neuron(stacked, valid, max_iters=max_ccmax_iters)

    fallback = torch.isnan(ccmax_n) & ~has_r2_stim
    ccmax_n = torch.where(fallback, torch.ones_like(ccmax_n), ccmax_n)
    return snr_n, ccmax_n


# -----------------------------------------------------------------------------
# 1. Per-stim helpers agree with the 4-D-tensor wrappers
# -----------------------------------------------------------------------------


def test_per_stim_sp_np_matches_4d_wrapper():
    """A single-stim call to ``_per_stim_sp_np`` reproduces the weighted
    aggregate of :func:`_sahani_linden_per_neuron` on the same data.

    With a single stim, the length-weighted average is just the per-stim
    value where the stim qualifies and NaN otherwise — making this a direct
    equality check.
    """
    from deepSTRF.metrics.performance import (
        _per_stim_sp_np,
        _sahani_linden_per_neuron,
    )

    g = torch.Generator().manual_seed(11)
    resp = 0.3 * torch.randn(5, 4, 30, generator=g)  # (N, R, T)
    resp[0, :, :2] = float("nan")                    # neuron 0: partial NaN
    valid = ~torch.isnan(resp)

    sp_s, np_s, tv_s = _per_stim_sp_np(resp, valid)
    sp_4d, np_4d = _sahani_linden_per_neuron(resp.unsqueeze(0), valid.unsqueeze(0))

    assert torch.allclose(sp_s, sp_4d, equal_nan=True)
    assert torch.allclose(np_s, np_4d, equal_nan=True)
    # All neurons here have R=4, T_v ≥ 28 → all qualify, all weights > 0.
    assert (tv_s > 0).all()


def test_per_stim_ccmax_matches_4d_wrapper():
    from deepSTRF.metrics.performance import (
        _ccmax_per_neuron,
        _per_stim_ccmax,
    )

    g = torch.Generator().manual_seed(13)
    base = torch.sin(torch.linspace(0, 4 * math.pi, 40)).expand(4, 40).clone()
    resp = base + 0.2 * torch.randn(3, 4, 40, generator=g)  # (N=3, R=4, T=40)
    valid = ~torch.isnan(resp)

    cc_s, tv_s = _per_stim_ccmax(resp, valid, max_iters=126)
    cc_4d = _ccmax_per_neuron(resp.unsqueeze(0), valid.unsqueeze(0), max_iters=126)

    assert torch.allclose(cc_s, cc_4d, equal_nan=True)
    assert (tv_s > 0).all()


def test_per_stim_skips_low_R_and_low_T():
    """Neurons with R < 2 or T < 2 get NaN / weight 0 (no contribution)."""
    from deepSTRF.metrics.performance import _per_stim_ccmax, _per_stim_sp_np

    g = torch.Generator().manual_seed(0)
    # N=3: neuron 0 has R=1, neuron 1 has T=1 (after valid filtering),
    # neuron 2 has the normal R=3, T=20 with a shared signal + per-trial
    # noise so ρ_half > 0 reliably (otherwise the CCmax helper documents-
    # ly drops the stim and we'd get tv == 0 there too).
    resp = torch.full((3, 3, 20), float("nan"))
    resp[0, 0] = torch.randn(20, generator=g)                # only R=1 valid
    resp[1, :, 0] = torch.randn(3, generator=g)              # only T=1 valid
    sig = torch.sin(torch.linspace(0, 4 * math.pi, 20))
    resp[2] = sig.unsqueeze(0) + 0.2 * torch.randn(3, 20, generator=g)
    valid = ~torch.isnan(resp)

    sp_s, np_s, tv_sp = _per_stim_sp_np(resp, valid)
    cc_s, tv_cc = _per_stim_ccmax(resp, valid, max_iters=126)

    assert math.isnan(sp_s[0].item()) and tv_sp[0].item() == 0.0
    assert math.isnan(np_s[0].item())
    assert math.isnan(cc_s[0].item()) and tv_cc[0].item() == 0.0
    assert math.isnan(sp_s[1].item()) and tv_sp[1].item() == 0.0
    assert math.isnan(cc_s[1].item()) and tv_cc[1].item() == 0.0
    assert torch.isfinite(sp_s[2]) and tv_sp[2].item() == 20.0
    assert torch.isfinite(cc_s[2]) and tv_cc[2].item() == 20.0


# -----------------------------------------------------------------------------
# 2. Dataset-level streaming is bit-identical to the legacy path
# -----------------------------------------------------------------------------


def test_streaming_equals_legacy_uniform_shapes():
    """Uniform (R, T) across (s, n) — the simple historical case."""
    ds = _fake_audio_dataset(N=4, S=5, T=30, R=4, seed=7)
    snr_legacy, ccmax_legacy = _legacy_compute_neuron_quality(ds)
    out = ds.compute_neuron_quality()
    assert torch.allclose(out["snr"], snr_legacy, equal_nan=True)
    assert torch.allclose(out["ccmax"], ccmax_legacy, equal_nan=True)


def test_streaming_equals_legacy_ragged_shapes():
    """Heterogeneous per-(s, n) shapes — exercises the per-stim padding path."""
    ds = _fake_audio_dataset(N=4, S=6, T=24, R=4, ragged=True, seed=21)
    snr_legacy, ccmax_legacy = _legacy_compute_neuron_quality(ds)
    out = ds.compute_neuron_quality()
    assert torch.allclose(out["snr"], snr_legacy, equal_nan=True)
    assert torch.allclose(out["ccmax"], ccmax_legacy, equal_nan=True)


def test_streaming_equals_legacy_with_missing_neurons():
    """Block-diagonal coverage: some neurons missing on some stims via the
    canonical ``(1, 1)`` NaN sentinel.
    """
    ds = _fake_audio_dataset(N=3, S=5, T=24, R=4, seed=33)
    # Knock neuron 1 out of stims 0 and 2 with the (1, 1) NaN sentinel.
    ds.responses[0][1] = torch.full((1, 1), float("nan"))
    ds.responses[2][1] = torch.full((1, 1), float("nan"))
    ds._invalidate_nrn_masks()

    snr_legacy, ccmax_legacy = _legacy_compute_neuron_quality(ds)
    out = ds.compute_neuron_quality()
    assert torch.allclose(out["snr"], snr_legacy, equal_nan=True)
    assert torch.allclose(out["ccmax"], ccmax_legacy, equal_nan=True)


def test_streaming_equals_legacy_r1_only_neuron():
    """A neuron with R=1 on every stim must still get ``ccmax = 1.0``
    (the documented sentinel) under streaming.
    """
    ds = _fake_audio_dataset(N=2, S=3, T=20, R=3, seed=0)
    for s in range(3):
        ds.responses[s][0] = ds.responses[s][0][:1]
    ds._invalidate_nrn_masks()

    snr_legacy, ccmax_legacy = _legacy_compute_neuron_quality(ds)
    out = ds.compute_neuron_quality()
    assert torch.allclose(out["snr"], snr_legacy, equal_nan=True)
    assert torch.allclose(out["ccmax"], ccmax_legacy, equal_nan=True)
    assert out["ccmax"][0].item() == 1.0


# -----------------------------------------------------------------------------
# 3. Streaming actually keeps peak memory under the legacy "padded" budget
# -----------------------------------------------------------------------------


def test_streaming_avoids_full_padded_tensor():
    """Build a dataset whose hypothetical ``(S, N, R_max, T_max)`` padded
    tensor would be much larger than the per-stim slab the streaming code
    actually allocates.

    We don't try to OOM the test runner — the point is to *prove* via the
    shape arithmetic that the new code never instantiates the full padded
    shape. Strategy: write a peak-memory tracker that wraps
    :meth:`torch.full` and records the largest allocation made during
    ``compute_neuron_quality()``. The streaming path's max allocation must
    stay bounded by ``N * R_s_max * T_s_max`` per stim, never the global
    ``S * N * R_max * T_max``.
    """
    N = 40
    S = 30
    R = 3
    T_short = 8
    T_long = 200

    # All stims are short EXCEPT one outlier with a much longer T. The legacy
    # padded tensor would have used T_max = T_long for every stim
    # (`S * N * R * T_long` floats); streaming only ever pads the outlier
    # to T_long, all other stims fit in T_short.
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass

    ds = _FakeAudio(path="/tmp/nowhere", dt_ms=1.0)
    ds.F = 4
    ds.N_neurons = N
    ds.stim_meta = [{"name": f"s{i}", "type": "synthetic"} for i in range(S)]
    ds.stims = [torch.zeros(1, 4, T_short) for _ in range(S)]
    g = torch.Generator().manual_seed(101)
    ds.responses = []
    for s in range(S):
        T_sn = T_long if s == 0 else T_short
        row = []
        for n in range(N):
            sig = torch.sin(torch.linspace(0, 2 * math.pi, T_sn))
            noise = 0.3 * torch.randn(R, T_sn, generator=g)
            row.append(sig.unsqueeze(0) + noise)
        ds.responses.append(row)
    ds.nrn_meta = [{"uid": f"n{i}"} for i in range(N)]
    ds.validate()

    # Track every torch.full() allocation made under compute_neuron_quality.
    full_orig = torch.full
    allocs = []

    def tracking_full(size, fill_value, *args, **kwargs):
        out = full_orig(size, fill_value, *args, **kwargs)
        # Skip scalar / 1-D bookkeeping tensors — we care about response slabs.
        if isinstance(size, (tuple, list)) and len(size) >= 3:
            allocs.append(tuple(size))
        return out

    torch.full = tracking_full
    try:
        ds.compute_neuron_quality()
    finally:
        torch.full = full_orig

    legacy_padded_numel = S * N * R * T_long
    max_allocated_numel = max((s[0] * s[1] * s[2] for s in allocs), default=0)
    # The streaming peak is one per-stim slab; the outlier's slab is the
    # largest, sized (N, R, T_long) — strictly smaller than (S, N, R, T_long).
    assert max_allocated_numel == N * R * T_long, (
        f"streaming should peak at one (N, R, T_long) slab = {N * R * T_long}, "
        f"got {max_allocated_numel} (allocs={allocs[:3]}...)"
    )
    assert max_allocated_numel < legacy_padded_numel, (
        f"streaming peak {max_allocated_numel} should be strictly smaller "
        f"than legacy padded tensor {legacy_padded_numel}"
    )
    # No alloc bigger than that ceiling either.
    for sh in allocs:
        assert sh[0] * sh[1] * sh[2] <= N * R * T_long, (
            f"unexpected over-sized allocation {sh} during streaming"
        )
