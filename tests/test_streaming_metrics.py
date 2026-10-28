"""StreamingCorrcoef: batch-by-batch accumulation equals the full-set metrics."""
from __future__ import annotations

import math

import pytest
import torch

from deepSTRF.metrics import StreamingCorrcoef, corrcoef, normalized_corrcoef
from deepSTRF.training import Fitter, set_random_seed
from deepSTRF.training.fitter import _pad_and_cat

from tests.test_fitter import _LinearReadout, _make_loaders


def _batches(seed=0, N=5, n_batches=6):
    """Batches with varying B / R / T, NaN padding, a neuron absent from one whole batch,
    a single-repeat batch, and a silent neuron (zero signal)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n_batches):
        B = 1 + i % 3
        R = 1 if i == 2 else 2 + i % 4
        T = 15 + 7 * i
        sig = torch.randn(B, N, 1, T, generator=g)
        resp = sig + 0.7 * torch.randn(B, N, R, T, generator=g)
        resp[:, 4] = 0.0                                   # silent neuron
        pred = 0.6 * sig + 0.5 * torch.randn(B, N, 1, T, generator=g)
        resp[:, :, :, T - 3:] = float("nan")               # ragged tails
        if R > 1:
            resp[0, 1, 0, :5] = float("nan")               # missing trial bins
        if i == 3:
            resp[:, 2] = float("nan")                      # neuron 2 absent from this batch
        out.append((pred, resp))
    return out


@pytest.mark.parametrize("method", [None, "schoppe"])
def test_streaming_equals_full_set(method):
    batches = _batches()
    m = StreamingCorrcoef(method=method)
    for p, r in batches:
        m.update(p, r)
    got = m.compute()
    P = _pad_and_cat([p for p, _ in batches])
    Rr = _pad_and_cat([r for _, r in batches])
    if method is None:
        ref = corrcoef(P, Rr, reduction="none")
    else:
        ref = normalized_corrcoef(P, Rr, method="schoppe", reduction="none")
    assert got.shape == ref.shape
    assert torch.equal(got.isnan(), ref.isnan())
    ok = ~ref.isnan()
    assert torch.allclose(got[ok], ref[ok], atol=1e-5, rtol=1e-4)
    assert bool(got[4].isnan())                            # silent neuron


def test_single_repeat_everywhere_falls_back_to_raw_cc():
    batches = [(p, r[:, :, :1]) for p, r in _batches()]
    a, b = StreamingCorrcoef("schoppe"), StreamingCorrcoef(None)
    for p, r in batches:
        a.update(p, r)
        b.update(p, r)
    x, y = a.compute(), b.compute()
    assert torch.equal(x.isnan(), y.isnan())
    assert torch.allclose(x[~x.isnan()], y[~y.isnan()])


def test_reset_clears_state():
    batches = _batches()
    m = StreamingCorrcoef()
    for p, r in batches:
        m.update(p, r)
    first = m.compute()
    m.reset()
    for p, r in batches:
        m.update(p, r)
    assert torch.allclose(first, m.compute(), equal_nan=True)


def test_fitter_streaming_metrics_match_callables():
    hist = {}
    for kind in ("callable", "streaming"):
        set_random_seed(0)
        train_loader, val_loader = _make_loaders()
        model = _LinearReadout(F=4, N=2)
        if kind == "callable":
            vm = {"cc": lambda p, r: corrcoef(p, r, reduction="none"),
                  "cc_norm": lambda p, r: normalized_corrcoef(p, r, method="schoppe",
                                                              reduction="none")}
        else:
            vm = {"cc": StreamingCorrcoef(), "cc_norm": StreamingCorrcoef("schoppe")}
        hist[kind] = Fitter(model, train_loader, val_loader, val_metrics=vm,
                            monitor="val_cc", mode="max", max_epochs=3, patience=3,
                            log_fn=lambda d: None).fit()
    for hc, hs in zip(hist["callable"], hist["streaming"]):
        for k in ("val_cc", "val_cc_norm", "train_cc", "train_cc_norm"):
            assert torch.allclose(torch.as_tensor(hc[k]), torch.as_tensor(hs[k]),
                                  atol=1e-5, equal_nan=True), k
        # every neuron has data in every batch here, so the streamed loss is exact
        assert math.isclose(hc["val_loss"], hs["val_loss"], rel_tol=1e-5)


def test_batch_sp_np_matches_per_stim_loop():
    from deepSTRF.metrics.performance import _per_stim_sp_np
    from deepSTRF.metrics.streaming import _batch_sp_np
    for p, r in _batches(seed=3):
        if r.shape[2] < 2:
            continue
        sp, np_, tv = _batch_sp_np(r)
        for b in range(r.shape[0]):
            s_ref, n_ref, t_ref = _per_stim_sp_np(r[b], ~r[b].isnan())
            assert torch.equal(sp[b].isnan(), s_ref.isnan())
            ok = ~s_ref.isnan()
            assert torch.allclose(sp[b][ok].float(), s_ref[ok], atol=1e-5)
            assert torch.allclose(np_[b][ok].float(), n_ref[ok], atol=1e-5)
            assert torch.equal(tv[b].float(), t_ref)


def test_neuron_chunking_gives_same_result(monkeypatch):
    import deepSTRF.metrics.streaming as st
    batches = _batches(seed=5)
    full = StreamingCorrcoef("schoppe")
    for p, r in batches:
        full.update(p, r)
    monkeypatch.setattr(st, "_CHUNK_ELEMS", 50)            # forces one neuron per chunk
    chunked = StreamingCorrcoef("schoppe")
    for p, r in batches:
        chunked.update(p, r)
    assert torch.allclose(full.compute(), chunked.compute(), equal_nan=True)
