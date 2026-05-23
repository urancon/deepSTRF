"""Performance metrics for deepSTRF: NaN-aware, eval-only, reduction over neurons.

See ``docs/_source/md/metrics_paradigm.md`` for the design rationale, shape
conventions, NaN handling, and per-metric formulas.
"""

from __future__ import annotations

import itertools
from typing import Optional, Tuple

import numpy as np
import scipy.signal
import torch

from deepSTRF.metrics._masking import (
    collapse_to_psth_if_needed,
    reduce_over_neurons,
    resolve_mask,
)

_EPS = 1e-12


# -----------------------------------------------------------------------------
# Shape and helper utilities
# -----------------------------------------------------------------------------


def _check_pred_gt(pred: torch.Tensor, gt: torch.Tensor) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal gt shape {tuple(gt.shape)}"
        )
    if pred.dim() != 4:
        raise ValueError(
            f"expected 4-D tensors (B, N, R, T), got {pred.dim()}"
        )


def _check_responses(responses: torch.Tensor) -> None:
    if responses.dim() != 4:
        raise ValueError(
            f"expected responses with 4 dims (B, N, R, T), got {responses.dim()}"
        )


def _pearson_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between two 1-D tensors. NaN if either has zero variance."""
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    den = torch.sqrt((x_c * x_c).sum() * (y_c * y_c).sum())
    if den.item() < _EPS:
        return x.new_full((), float("nan"))
    return num / den


# -----------------------------------------------------------------------------
# Public: prediction-vs-PSTH metrics
# -----------------------------------------------------------------------------


@torch.no_grad()
def corrcoef(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pearson correlation per neuron over flattened valid (B, T) positions.

    ``gt`` may be passed as either a pre-computed PSTH ``(B, N, 1, T)`` or a
    raw responses tensor ``(B, N, R, T)`` with ``R > 1``; in the latter case
    it is collapsed to PSTH via ``nanmean(dim=2, keepdim=True)`` before the
    correlation is computed.

    See ``metrics_paradigm.md`` §6.3.
    """
    gt = collapse_to_psth_if_needed(gt)
    _check_pred_gt(pred, gt)
    valid = resolve_mask(gt, mask)
    N = pred.shape[1]
    nan = pred.new_full((), float("nan"))
    out = []
    for n in range(N):
        v_n = valid[:, n, ...]
        if int(v_n.sum().item()) < 2:
            out.append(nan)
            continue
        p_n = pred[:, n, ...][v_n]
        g_n = gt[:, n, ...][v_n]
        out.append(_pearson_1d(p_n, g_n))
    return reduce_over_neurons(torch.stack(out), reduction)


@torch.no_grad()
def fve(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Fraction of variance explained (R²) per neuron, over flattened valid positions.

    ``FVE_n = 1 - SS_res / SS_tot`` where SS_tot is the variance of ``gt`` only.
    Negative when the prediction is worse than predicting the mean.

    ``gt`` may be passed as either a pre-computed PSTH ``(B, N, 1, T)`` or a
    raw responses tensor ``(B, N, R, T)`` with ``R > 1``; in the latter case
    it is collapsed to PSTH via ``nanmean(dim=2, keepdim=True)`` before the
    metric is computed.
    """
    gt = collapse_to_psth_if_needed(gt)
    _check_pred_gt(pred, gt)
    valid = resolve_mask(gt, mask)
    N = pred.shape[1]
    nan = pred.new_full((), float("nan"))
    out = []
    for n in range(N):
        v_n = valid[:, n, ...]
        if int(v_n.sum().item()) < 2:
            out.append(nan)
            continue
        p_n = pred[:, n, ...][v_n]
        g_n = gt[:, n, ...][v_n]
        ss_res = ((p_n - g_n) ** 2).sum()
        ss_tot = ((g_n - g_n.mean()) ** 2).sum()
        if ss_tot.item() < _EPS:
            out.append(nan)
            continue
        out.append(1.0 - ss_res / ss_tot)
    return reduce_over_neurons(torch.stack(out), reduction)


# -----------------------------------------------------------------------------
# Public: Sahani–Linden signal/noise/SNR (responses-only)
# -----------------------------------------------------------------------------


def _per_stim_sp_np(
    responses_b: torch.Tensor,
    valid_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-stim SP, NP, and valid-time-count, all shape ``(N,)``.

    Behaviour-preserving extraction of the inner per-``(stim, neuron)`` loop
    from :func:`_sahani_linden_per_neuron` — exposes a stim-streaming entry
    point so callers can accumulate length-weighted partial sums without
    pre-building an ``(S, N, R_max, T_max)`` padded tensor (see
    :meth:`NeuralDataset.compute_neuron_quality`).

    Parameters
    ----------
    responses_b : torch.Tensor
        Per-stim responses, shape ``(N, R, T)``. Invalid entries flagged via
        ``valid_b`` (NaN sentinels or an explicit mask).
    valid_b : torch.Tensor
        Bool mask of the same shape as ``responses_b``.

    Returns
    -------
    sp_b, np_b : torch.Tensor
        ``(N,)`` per-neuron SP / NP for this stim. ``NaN`` for neurons that
        don't qualify (``R_v < 2`` or ``T_v < 2``).
    Tv_b : torch.Tensor
        ``(N,)`` per-neuron valid-time-bin count. ``0`` for non-qualifying
        neurons — used as the weight in length-weighted aggregation.
    """
    if responses_b.dim() != 3:
        raise ValueError(
            f"_per_stim_sp_np expects (N, R, T), got {tuple(responses_b.shape)}"
        )
    N, _, _ = responses_b.shape
    nan = responses_b.new_full((), float("nan"))
    zero = responses_b.new_zeros(())
    sp_out = []
    np_out = []
    tv_out = []
    for n in range(N):
        v_n = valid_b[n]                              # (R, T)
        valid_repeats = v_n.any(dim=-1)               # (R,)
        valid_time = v_n.any(dim=0)                   # (T,)
        R_v = int(valid_repeats.sum().item())
        T_v = int(valid_time.sum().item())
        if R_v < 2 or T_v < 2:
            sp_out.append(nan)
            np_out.append(nan)
            tv_out.append(zero)
            continue
        sub = responses_b[n][valid_repeats][:, valid_time]   # (R_v, T_v)
        psth = sub.mean(dim=0)                                # (T_v,)
        var_psth = psth.var(unbiased=True)                    # scalar
        tp = sub.var(dim=-1, unbiased=True).mean()            # scalar
        sp = (R_v * var_psth - tp) / (R_v - 1)
        sp_out.append(sp)
        np_out.append(tp - sp)
        tv_out.append(responses_b.new_tensor(float(T_v)))
    return torch.stack(sp_out), torch.stack(np_out), torch.stack(tv_out)


def _sahani_linden_per_neuron(
    responses: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (SP, NP) per neuron, both shape ``(N,)``.

    Per-stimulus computation, then **length-weighted** average across stims:

        SP_n = sum_b ( T_b · SP_{b,n} ) / sum_b T_b

    where ``T_b`` is the number of valid time bins for stim ``b``. This matches
    the natural sample-count weighting of ``cov`` and ``var`` over the
    concatenated ``(b, t)`` time series in ``corrcoef`` / ``normalized_corrcoef``,
    so long stims drive the SP estimate proportionally to their information
    content (the BLUE estimator under the assumption that each per-stim
    ``SP_{b,n}`` is unbiased — see ``metrics_paradigm.md`` §6.5).

    A given ``(b, n)`` cell is included iff it has ≥ 2 valid repeats and ≥ 2
    valid time bins. Cells that have no qualifying stim get NaN.

    Thin wrapper over :func:`_per_stim_sp_np`: loops over the batch axis and
    accumulates per-neuron length-weighted partial sums. The streaming-friendly
    helper is the underscore-prefixed per-stim variant — callers handling
    large datasets should use it directly to avoid pre-padding ``(S, N,
    R_max, T_max)`` in memory.
    """
    B, N, _, _ = responses.shape
    nan = responses.new_full((), float("nan"))
    sum_w = responses.new_zeros(N)
    sum_w_sp = responses.new_zeros(N)
    sum_w_np = responses.new_zeros(N)
    for b in range(B):
        sp_b, np_b, tv_b = _per_stim_sp_np(responses[b], valid[b])
        contrib = tv_b > 0
        if not bool(contrib.any()):
            continue
        # tv_b is 0 for non-qualifying neurons → masked multiplications stay 0
        # whether sp_b/np_b are NaN there or not; explicit `where` avoids the
        # 0 * NaN = NaN trap.
        sum_w = sum_w + tv_b
        sum_w_sp = sum_w_sp + torch.where(contrib, tv_b * sp_b, sum_w_sp.new_zeros(()))
        sum_w_np = sum_w_np + torch.where(contrib, tv_b * np_b, sum_w_np.new_zeros(()))
    qualifying = sum_w > 0
    sp_out = torch.where(qualifying, sum_w_sp / sum_w.clamp(min=1), nan.expand(N))
    np_out = torch.where(qualifying, sum_w_np / sum_w.clamp(min=1), nan.expand(N))
    return sp_out, np_out


@torch.no_grad()
def signal_power(
    responses: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sahani-Linden signal power per neuron.

    Computed per-stimulus (with the per-stim valid repeat count) then
    averaged across stimuli (``nanmean``). Cells without any qualifying
    stim — needs ≥ 2 valid repeats and ≥ 2 valid time bins — return NaN.
    """
    _check_responses(responses)
    valid = resolve_mask(responses, mask)
    sp, _ = _sahani_linden_per_neuron(responses, valid)
    return reduce_over_neurons(sp, reduction)


@torch.no_grad()
def noise_power(
    responses: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sahani-Linden noise power per neuron. ``NP = TP - SP``."""
    _check_responses(responses)
    valid = resolve_mask(responses, mask)
    _, np_ = _sahani_linden_per_neuron(responses, valid)
    return reduce_over_neurons(np_, reduction)


@torch.no_grad()
def snr(
    responses: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Signal-to-noise ratio per neuron. ``SNR = SP / NP``.

    Returns ``+inf`` for noiseless cells (NP ≈ 0). Caller decides whether
    to filter.
    """
    _check_responses(responses)
    valid = resolve_mask(responses, mask)
    sp, np_ = _sahani_linden_per_neuron(responses, valid)
    out = sp / np_.clamp(min=_EPS)
    return reduce_over_neurons(out, reduction)


# -----------------------------------------------------------------------------
# Public: noise-corrected prediction correlation
# -----------------------------------------------------------------------------


def _per_stim_ccmax(
    responses_b: torch.Tensor,
    valid_b: torch.Tensor,
    max_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-stim CCmax + valid-time-count, both shape ``(N,)``.

    Behaviour-preserving extraction of the inner per-``(stim, neuron)`` loop
    from :func:`_ccmax_per_neuron`. Returns ``T_v = 0`` for any neuron that
    doesn't contribute on this stim — both the ``R_v < 2 / T_v < 2``
    structural skip and the ``ρ_half ≤ 0`` "too noisy to estimate" skip
    collapse to zero weight in the streaming aggregator.

    Parameters
    ----------
    responses_b : torch.Tensor
        Per-stim responses, shape ``(N, R, T)``.
    valid_b : torch.Tensor
        Bool mask of the same shape as ``responses_b``.
    max_iters : int
        Cap on random half-splits per neuron (``C(R, R/2)`` blows up
        quickly).

    Returns
    -------
    ccmax_b : torch.Tensor
        ``(N,)`` per-neuron CCmax for this stim. ``NaN`` for neurons that
        don't qualify or where ``ρ_half ≤ 0``.
    Tv_b : torch.Tensor
        ``(N,)`` per-neuron valid-time-bin count. ``0`` for non-contributing
        neurons — use as the streaming weight.
    """
    if responses_b.dim() != 3:
        raise ValueError(
            f"_per_stim_ccmax expects (N, R, T), got {tuple(responses_b.shape)}"
        )
    N, _, _ = responses_b.shape
    nan = responses_b.new_full((), float("nan"))
    zero = responses_b.new_zeros(())
    cc_out = []
    tv_out = []
    for n in range(N):
        v_n = valid_b[n]
        valid_repeats = v_n.any(dim=-1)
        valid_time = v_n.any(dim=0)
        R_v = int(valid_repeats.sum().item())
        T_v = int(valid_time.sum().item())
        if R_v < 2 or T_v < 2:
            cc_out.append(nan)
            tv_out.append(zero)
            continue
        sub = responses_b[n][valid_repeats][:, valid_time]
        if R_v % 2 == 1:
            R_v -= 1
            sub = sub[:R_v]
        half_sets = list(itertools.combinations(range(R_v), R_v // 2))
        n_iters = min(len(half_sets) // 2, max_iters)
        cc_halfs = []
        for i in range(n_iters):
            first = sub[list(half_sets[i])].mean(dim=0)
            second = sub[list(half_sets[-1 - i])].mean(dim=0)
            cc_halfs.append(_pearson_1d(first, second))
        if not cc_halfs:
            cc_out.append(nan)
            tv_out.append(zero)
            continue
        rho_half = torch.stack(cc_halfs).mean()
        if rho_half.item() <= 0:
            # too noisy to estimate ceiling; drop this stim from the weighted avg
            cc_out.append(nan)
            tv_out.append(zero)
            continue
        cc_out.append(torch.sqrt(2 * rho_half / (1 + rho_half)))
        tv_out.append(responses_b.new_tensor(float(T_v)))
    return torch.stack(cc_out), torch.stack(tv_out)


def _ccmax_per_neuron(
    responses: torch.Tensor,
    valid: torch.Tensor,
    max_iters: int,
) -> torch.Tensor:
    """Per-neuron CCmax (Hsu / Spearman-Brown), length-weighted across stims.

    Same length-weighting convention as ``_sahani_linden_per_neuron``: each
    per-stim CCmax_{b,n} is weighted by ``T_b`` (its valid time count) so that
    long stims drive the ceiling estimate more than short ones, matching the
    natural sample weighting of ``corrcoef`` over the concatenated time axis.
    Per-stim cells with ``ρ_half ≤ 0`` are dropped (NaN-skip in the weighted
    average). Cells with no qualifying stim get NaN.

    Thin wrapper over :func:`_per_stim_ccmax`. See
    :func:`_sahani_linden_per_neuron` for the equivalent SP/NP streaming
    rationale.
    """
    B, N, _, _ = responses.shape
    nan = responses.new_full((), float("nan"))
    sum_w = responses.new_zeros(N)
    sum_w_cc = responses.new_zeros(N)
    for b in range(B):
        cc_b, tv_b = _per_stim_ccmax(responses[b], valid[b], max_iters=max_iters)
        contrib = tv_b > 0
        if not bool(contrib.any()):
            continue
        sum_w = sum_w + tv_b
        sum_w_cc = sum_w_cc + torch.where(contrib, tv_b * cc_b, sum_w_cc.new_zeros(()))
    qualifying = sum_w > 0
    return torch.where(qualifying, sum_w_cc / sum_w.clamp(min=1), nan.expand(N))


@torch.no_grad()
def normalized_corrcoef(
    pred: torch.Tensor,
    responses: torch.Tensor,
    method: str = "schoppe",
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    ccmax_iters: int = 126,
) -> torch.Tensor:
    """Noise-corrected correlation coefficient per neuron.

    ``method='schoppe'`` divides by ``sqrt(var(pred) · SP)`` (Schoppe 2016).
    ``method='hsu'`` divides the raw Pearson by ``CCmax`` (Hsu/Spearman-Brown).
    Single-trial degenerate case (R = 1 everywhere): falls back to raw
    ``corrcoef(pred, psth)``.

    See ``metrics_paradigm.md`` §6.4.
    """
    if pred.dim() != 4 or responses.dim() != 4:
        raise ValueError(
            f"pred and responses must be 4-D, got {pred.dim()} and "
            f"{responses.dim()}"
        )
    if pred.shape[2] != 1:
        raise ValueError(
            f"pred R-axis must be 1 (model output convention), got "
            f"{pred.shape[2]}"
        )
    if (
        pred.shape[0] != responses.shape[0]
        or pred.shape[1] != responses.shape[1]
        or pred.shape[3] != responses.shape[3]
    ):
        raise ValueError(
            f"pred and responses must agree on B, N, T axes; got "
            f"pred {tuple(pred.shape)} vs responses {tuple(responses.shape)}"
        )

    psth = torch.nanmean(responses, dim=2, keepdim=True)             # (B, N, 1, T)
    valid_psth = resolve_mask(psth, mask)
    valid_resp = resolve_mask(responses, None)

    # Single-trial degenerate case: the noise correction is undefined.
    if responses.shape[2] == 1:
        return corrcoef(pred, psth, mask=mask, reduction=reduction)

    if method == "schoppe":
        sp, _ = _sahani_linden_per_neuron(responses, valid_resp)     # (N,)
        N = pred.shape[1]
        nan = pred.new_full((), float("nan"))
        out = []
        for n in range(N):
            v_n = valid_psth[:, n, ...]
            if int(v_n.sum().item()) < 2:
                out.append(nan)
                continue
            p_n = pred[:, n, ...][v_n]
            g_n = psth[:, n, ...][v_n]
            T_eff = p_n.numel()
            # Bessel-correct cov to match var(unbiased=True) and SP.
            cov_pg = ((p_n - p_n.mean()) * (g_n - g_n.mean())).sum() / max(T_eff - 1, 1)
            var_p = p_n.var(unbiased=True)
            sp_n = sp[n]
            if not torch.isfinite(sp_n) or sp_n.item() <= 0 or var_p.item() < _EPS:
                out.append(nan)
                continue
            out.append(cov_pg / torch.sqrt(var_p * sp_n))
        return reduce_over_neurons(torch.stack(out), reduction)

    if method == "hsu":
        ccmax = _ccmax_per_neuron(responses, valid_resp, ccmax_iters)  # (N,)
        cc_raw = corrcoef(pred, psth, mask=mask, reduction="none")     # (N,)
        cc_norm = cc_raw / ccmax
        return reduce_over_neurons(cc_norm, reduction)

    raise ValueError(
        f"method must be 'schoppe' or 'hsu', got {method!r}"
    )


# -----------------------------------------------------------------------------
# Public: coherence (eval-only, NaN-intolerant)
# -----------------------------------------------------------------------------


@torch.no_grad()
def coherence(
    pred: torch.Tensor,
    gt: torch.Tensor,
    dt_ms: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Magnitude-squared coherence per neuron (mean over frequency bins).

    Eval-only: uses ``scipy.signal.coherence``, no gradient. Does **not**
    tolerate NaN — raises ``ValueError`` if any input element is NaN.
    """
    _check_pred_gt(pred, gt)
    if torch.isnan(pred).any() or torch.isnan(gt).any():
        raise ValueError(
            "coherence does not tolerate NaN; pre-flatten to a NaN-free subset."
        )
    fs = 1.0 / (dt_ms * 1e-3)
    p = pred.squeeze(2).cpu().numpy()                                # (B, N, T)
    g = gt.squeeze(2).cpu().numpy()                                  # (B, N, T)
    _, coh = scipy.signal.coherence(p, g, fs=fs)                     # (B, N, F)
    coh_t = torch.from_numpy(coh).to(pred.device).to(pred.dtype)
    per_neuron = coh_t.mean(dim=(0, -1))                             # (N,)
    return reduce_over_neurons(per_neuron, reduction)


# -----------------------------------------------------------------------------
# Internal helpers (importable for callers like WehrDataset)
# -----------------------------------------------------------------------------


@torch.no_grad()
def compute_CCmax(
    responses: torch.Tensor,
    max_iters: int = 126,
) -> torch.Tensor:
    """CCmax (Hsu / Spearman-Brown) per ``(B,)`` cell.

    Internal helper kept for backward compatibility with ``WehrDataset``.
    Input ``(B, R, T)``; returns ``(B,)``. NaN-aware: drops invalid repeats
    and time bins per ``(b,)``. Returns 1.0 for cells with R = 1, NaN for
    cells with ``ρ_half ≤ 0``.
    """
    if responses.dim() != 3:
        raise ValueError(
            f"compute_CCmax expects (B, R, T), got {tuple(responses.shape)}"
        )
    valid = ~responses.isnan()
    B, R, T = responses.shape
    nan = responses.new_full((), float("nan"))
    out = []
    for b in range(B):
        v_b = valid[b]
        valid_repeats = v_b.any(dim=-1)
        valid_time = v_b.any(dim=0)
        R_v = int(valid_repeats.sum().item())
        T_v = int(valid_time.sum().item())
        if R_v < 2 or T_v < 2:
            out.append(responses.new_ones(()) if R_v == 1 else nan)
            continue
        sub = responses[b][valid_repeats][:, valid_time]
        if R_v % 2 == 1:
            R_v -= 1
            sub = sub[:R_v]
        half_sets = list(itertools.combinations(range(R_v), R_v // 2))
        n_iters = min(len(half_sets) // 2, max_iters)
        cc_halfs = []
        for i in range(n_iters):
            first = sub[list(half_sets[i])].mean(dim=0)
            second = sub[list(half_sets[-1 - i])].mean(dim=0)
            cc_halfs.append(_pearson_1d(first, second))
        if not cc_halfs:
            out.append(nan)
            continue
        rho_half = torch.stack(cc_halfs).mean()
        if rho_half.item() <= 0:
            out.append(nan)
            continue
        out.append(torch.sqrt(2 * rho_half / (1 + rho_half)))
    return torch.stack(out)


@torch.no_grad()
def compute_TTRC(responses: torch.Tensor) -> torch.Tensor:
    """Trial-to-trial response correlation per ``(B,)`` cell.

    Internal helper kept for backward compatibility with ``WehrDataset``.
    Input ``(B, R, T)``; returns ``(B,)``. NaN-aware. Returns 1.0 for
    R = 1, NaN for cells with no valid trial pair.
    """
    if responses.dim() != 3:
        raise ValueError(
            f"compute_TTRC expects (B, R, T), got {tuple(responses.shape)}"
        )
    valid = ~responses.isnan()
    B, R, T = responses.shape
    nan = responses.new_full((), float("nan"))
    out = []
    for b in range(B):
        v_b = valid[b]
        valid_repeats = v_b.any(dim=-1)
        valid_time = v_b.any(dim=0)
        R_v = int(valid_repeats.sum().item())
        T_v = int(valid_time.sum().item())
        if R_v < 2 or T_v < 2:
            out.append(responses.new_ones(()) if R_v == 1 else nan)
            continue
        sub = responses[b][valid_repeats][:, valid_time]
        cc_pairs = []
        for i in range(R_v):
            for j in range(i + 1, R_v):
                cc_pairs.append(_pearson_1d(sub[i], sub[j]))
        if not cc_pairs:
            out.append(nan)
            continue
        out.append(torch.stack(cc_pairs).mean())
    return torch.stack(out)
