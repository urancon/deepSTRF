"""Streaming (batch-by-batch) versions of the correlation metrics.

:func:`~deepSTRF.metrics.corrcoef` and :func:`~deepSTRF.metrics.normalized_corrcoef` take the
whole evaluation set at once: predictions ``(S, N, 1, T)`` and responses ``(S, N, R, T)``,
padded to the largest ``R`` and ``T``. On large datasets that tensor alone does not fit in
memory (Wingert 2026: 4072 training stimuli x 2874 neurons x 2200 bins). Both metrics are
exact functions of per-neuron running sums, so they can be accumulated one batch at a time:

- Pearson ``cc``: count, sum p, sum g, sum p^2, sum g^2, sum p*g over the valid PSTH bins
  (accumulated in float64);
- Schoppe ``cc_norm``: the same sums, plus the signal / noise power, which the library already
  computes per stimulus and averages with length weights (same definition as
  :func:`~deepSTRF.metrics.performance._per_stim_sp_np`, vectorised over neurons).

Memory is one batch plus a few ``(N,)`` vectors, whatever the size of the set. Results equal
the full-set functions up to float rounding (tested in ``tests/test_streaming_metrics.py``).

A streaming metric has ``reset()``, ``update(pred, responses)`` and ``compute()``; the
:class:`~deepSTRF.training.Fitter` uses this interface when a ``val_metrics`` value provides it.
"""
from __future__ import annotations

from typing import Optional

import torch

from deepSTRF.metrics.performance import _EPS

__all__ = ["StreamingCorrcoef"]

_CHUNK_ELEMS = 32_000_000     # neurons per signal-power chunk: n * R * T <= this


@torch.no_grad()
def _batch_sp_np(responses: torch.Tensor):
    """Vectorised :func:`~deepSTRF.metrics.performance._per_stim_sp_np` over a whole batch.

    ``responses`` ``(B, N, R, T)``, NaN = invalid. Computed in the input dtype (float32, as the
    loop version), returned as float64 ``(B, N)`` tensors SP, NP
    and the valid-time count ``T_v`` (0 where the (stim, neuron) cell has < 2 valid repeats
    or < 2 valid bins). Same per-cell definition: valid repeats = repeats with any valid bin,
    valid bins = bins with any valid repeat; a NaN left inside that block propagates, as in
    the loop version.
    """
    x = responses                                        # float32 elementwise, like the loop
    v = ~x.isnan()
    vr = v.any(dim=-1)                                   # (B, N, R)
    vt = v.any(dim=-2)                                   # (B, N, T)
    Rv = vr.sum(-1).to(x.dtype)                          # (B, N)
    Tv = vt.sum(-1).to(x.dtype)
    block = vr.unsqueeze(-1) & vt.unsqueeze(-2)          # (B, N, R, T)
    xb = torch.where(block, x, torch.zeros((), dtype=x.dtype, device=x.device))
    xb = torch.where(block & x.isnan(), x, xb)           # keep NaN inside the block
    Tv1, Rv1 = Tv.clamp(min=1), Rv.clamp(min=1)
    psth = xb.sum(-2) / Rv1.unsqueeze(-1)                # (B, N, T); 0 outside valid bins
    m = (psth * vt).sum(-1) / Tv1
    var_psth = (((psth - m.unsqueeze(-1)) ** 2) * vt).sum(-1) / (Tv - 1).clamp(min=1)
    mr = xb.sum(-1) / Tv1.unsqueeze(-1)                  # (B, N, R) per-repeat mean
    var_r = (((xb - mr.unsqueeze(-1)) ** 2) * block).sum(-1) / (Tv - 1).clamp(min=1).unsqueeze(-1)
    tp = (var_r * vr).sum(-1) / Rv1                      # mean over valid repeats
    sp = ((Rv * var_psth - tp) / (Rv - 1).clamp(min=1)).double()
    tp, Tv = tp.double(), Tv.double()
    ok = (Rv >= 2) & (Tv >= 2)
    nan = torch.full_like(sp, float("nan"))
    return (torch.where(ok, sp, nan), torch.where(ok, tp - sp, nan),
            torch.where(ok, Tv, torch.zeros_like(Tv)))


class StreamingCorrcoef:
    """Per-neuron Pearson cc (``method=None``) or Schoppe ``cc_norm`` (``method='schoppe'``),
    accumulated batch by batch.

    Same definitions as ``corrcoef(pred, responses, reduction='none')`` and
    ``normalized_corrcoef(pred, responses, method='schoppe', reduction='none')``:
    the PSTH is the NaN-mean over repeats; invalid (NaN) bins are skipped; a neuron with
    fewer than 2 valid bins, or zero variance, gets NaN.

    Parameters
    ----------
    method : {None, 'schoppe'}, default None
        ``None`` for the raw Pearson correlation with the PSTH, ``'schoppe'`` for the
        noise-corrected one. ``'hsu'`` (split-half CCmax) is not streamable.
    sp_rel_threshold : float, default 1e-6
        As in :func:`~deepSTRF.metrics.normalized_corrcoef`.
    """

    def __init__(self, method: Optional[str] = None, sp_rel_threshold: float = 1e-6):
        if method not in (None, "schoppe"):
            raise ValueError(f"method must be None or 'schoppe', got {method!r}")
        self.method = method
        self.sp_rel_threshold = sp_rel_threshold
        self.reset()

    def reset(self) -> None:
        self._n = None                       # (N,) float64 sums
        self._r_max = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, responses: torch.Tensor) -> None:
        """Add one batch: ``pred`` ``(B, N, 1, T)``, ``responses`` ``(B, N, R, T)``."""
        if pred.dim() != 4 or responses.dim() != 4 or pred.shape[2] != 1:
            raise ValueError(f"expected pred (B, N, 1, T) and responses (B, N, R, T); got "
                             f"{tuple(pred.shape)} and {tuple(responses.shape)}")
        if pred.device.type == "mps":            # float64 accumulators: MPS has no float64
            pred, responses = pred.cpu(), responses.cpu()
        psth = torch.nanmean(responses, dim=2, keepdim=True)          # (B, N, 1, T)
        valid = ~psth.isnan()
        p = torch.where(valid, pred, torch.zeros((), dtype=pred.dtype, device=pred.device)).double()
        g = torch.nan_to_num(psth, nan=0.0).double()
        dims = (0, 2, 3)
        n = valid.sum(dim=dims).double()
        sums = [n, p.sum(dims), g.sum(dims), (p * p).sum(dims), (g * g).sum(dims), (p * g).sum(dims)]
        if self._n is None:
            N = pred.shape[1]
            z = lambda: torch.zeros(N, dtype=torch.float64, device=pred.device)
            self._n, self._sp, self._sg, self._spp, self._sgg, self._spg = (z() for _ in range(6))
            self._w, self._wsp, self._wnp = z(), z(), z()
        for acc, s in zip((self._n, self._sp, self._sg, self._spp, self._sgg, self._spg), sums):
            acc += s
        self._r_max = max(self._r_max, responses.shape[2])
        if self.method == "schoppe" and responses.shape[2] > 1:
            # one stimulus and a chunk of neurons at a time: bounded memory (~32M elements)
            N, R, T = responses.shape[1], responses.shape[2], responses.shape[3]
            step = max(1, int(_CHUNK_ELEMS // max(R * T, 1)))
            for b in range(responses.shape[0]):
                for n0 in range(0, N, step):
                    sl = slice(n0, n0 + step)
                    sp_b, np_b, tv_b = _batch_sp_np(responses[b:b + 1, sl])   # (1, n) each
                    ok = tv_b > 0
                    zero = torch.zeros_like(tv_b)
                    self._w[sl] += tv_b.sum(0)
                    self._wsp[sl] += torch.where(ok, tv_b * sp_b, zero).sum(0)
                    self._wnp[sl] += torch.where(ok, tv_b * np_b, zero).sum(0)

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        """``(N,)`` float32 tensor (on the device of the batches)."""
        if self._n is None:
            raise RuntimeError("compute() before any update()")
        n = self._n
        nan = torch.full_like(n, float("nan"))
        n1 = n.clamp(min=1)
        cov = self._spg - self._sp * self._sg / n1                   # sum of centred products
        vp = self._spp - self._sp ** 2 / n1
        vg = self._sgg - self._sg ** 2 / n1
        enough = n >= 2
        if self.method is None or self._r_max <= 1:
            # corrcoef: NaN if sqrt(sum_c p^2 * sum_c g^2) < eps (as _pearson_1d)
            den = torch.sqrt((vp * vg).clamp(min=0))
            ok = enough & (den >= _EPS)
            return torch.where(ok, cov / den.clamp(min=_EPS), nan).float()
        # Schoppe: cov / sqrt(var_p * SP), Bessel-corrected cov and var (n - 1)
        q = self._w > 0
        sp = torch.where(q, self._wsp / self._w.clamp(min=1), nan)
        np_ = torch.where(q, self._wnp / self._w.clamp(min=1), nan)
        nb = (n - 1).clamp(min=1)
        var_p = vp / nb
        no_signal = sp <= self.sp_rel_threshold * (sp + np_)
        ok = enough & torch.isfinite(sp) & (sp > 0) & ~no_signal & (var_p >= _EPS)
        out = (cov / nb) / torch.sqrt((var_p * sp).clamp(min=_EPS))
        return torch.where(ok, out, nan).float()
