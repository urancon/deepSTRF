"""Loss functions for deepSTRF: gradient-bearing, NaN-aware, reduction over neurons.

See ``docs/_source/md/metrics_paradigm.md`` for shape conventions, NaN handling,
and reduction semantics.
"""

from __future__ import annotations

from typing import Optional

import torch

from deepSTRF.metrics._masking import (
    collapse_to_psth_if_needed,
    per_neuron_mean,
    reduce_over_neurons,
    resolve_mask,
)


def mse_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Boolean-masked mean squared error, reduced over the neuron axis.

    Parameters
    ----------
    pred
        Prediction tensor of shape ``(B, N, 1, T)``.
    gt
        Ground-truth tensor of shape ``(B, N, 1, T)`` (PSTH or single-trial
        target) **or** ``(B, N, R, T)`` with ``R > 1`` (raw responses), in
        which case it is collapsed to PSTH via ``nanmean(dim=2, keepdim=True)``
        before the loss is computed. ``gt`` may contain NaN; positions where
        the resulting PSTH is NaN are dropped from the per-neuron mean.
    mask
        Optional bool tensor broadcastable to the post-collapse ``gt`` shape
        ``(B, N, 1, T)``. If None, defaults to ``~gt.isnan()``. If provided,
        REPLACES (does not augment) the NaN-derived mask.
    reduction
        ``'none'`` → ``(N,)``; ``'mean'``/``'sum'`` → scalar via nanmean/nansum.
    """
    gt = collapse_to_psth_if_needed(gt)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal gt shape "
            f"{tuple(gt.shape)}"
        )
    if pred.dim() != 4:
        raise ValueError(
            f"expected pred and gt with 4 dims (B, N, R, T), got {pred.dim()}"
        )
    valid = resolve_mask(gt, mask)
    # Replace NaN positions in gt with 0 BEFORE the residual so the
    # autograd path stays finite. The forward sum is unchanged because
    # per_neuron_mean masks these positions out; without this the
    # gradient ``2*(pred - NaN) * 0`` evaluates to NaN under IEEE 754
    # and contaminates upstream parameters.
    gt_safe = torch.where(valid, gt, torch.zeros_like(gt))
    diff_sq = (pred - gt_safe) ** 2
    per_neuron = per_neuron_mean(diff_sq, valid)
    return reduce_over_neurons(per_neuron, reduction)


def poisson_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    log_input: bool = False,
    validate_input: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative Poisson log-likelihood (without the ``log(gt!)`` constant).

    Two parameterisations of the prediction are supported, matching the
    canonical-link logic of generalised linear models:

    - ``log_input=False`` (default): ``pred`` is interpreted as the rate ``λ``.
      The loss is ``pred − gt · log(pred + eps)`` per element. ``pred`` must
      be non-negative for the log to be meaningful; the implementation
      *silently clamps* ``pred`` to ``≥ eps`` inside the ``log`` to avoid NaN
      propagation (the linear term keeps its sign). For loud failure on
      negative predictions, pass ``validate_input=True`` — at the cost of a
      per-step CPU sync.

    - ``log_input=True``: ``pred`` is interpreted as the log-rate ``η = log(λ)``.
      The loss becomes ``exp(pred) − gt · pred``, which is well-defined for
      any real-valued ``pred``. This is the standard trick for pairing a
      Poisson NLL with an unbounded readout (Linear, ParametricSigmoid that
      is allowed to dip below zero, etc.). See ``metrics_paradigm.md`` §6.2
      for the GLM-canonical-link derivation.

    The ``log(gt!)`` Stirling term is *not* added — for non-integer ``gt``
    (e.g. trial-averaged PSTH binned counts) it is meaningless, and for
    integer ``gt`` it is constant in ``pred`` so it does not affect
    optimisation. Users who want the full likelihood for AIC/BIC can add
    it themselves.

    ``gt`` may be passed as either a pre-computed PSTH ``(B, N, 1, T)`` or a
    raw responses tensor ``(B, N, R, T)`` with ``R > 1``; in the latter case
    it is collapsed to PSTH via ``nanmean(dim=2, keepdim=True)`` before the
    loss is computed.
    """
    gt = collapse_to_psth_if_needed(gt)
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal gt shape "
            f"{tuple(gt.shape)}"
        )
    if pred.dim() != 4:
        raise ValueError(
            f"expected pred and gt with 4 dims (B, N, R, T), got {pred.dim()}"
        )
    valid = resolve_mask(gt, mask)
    # Same NaN-gradient-leak protection as in mse_loss: replace NaN
    # positions in gt with 0 before the loss formula, so the autograd
    # path stays finite.
    gt_safe = torch.where(valid, gt, torch.zeros_like(gt))

    if log_input:
        elem = torch.exp(pred) - gt_safe * pred
    else:
        if validate_input and (pred.masked_fill(~valid, 0.0) < 0).any():
            raise ValueError(
                "poisson_loss(log_input=False): pred has negative values at "
                "masked-in positions. Either use a non-negative output "
                "activation, set log_input=True (interpret pred as log-rate), "
                "or drop validate_input=True to silently clamp inside log."
            )
        elem = pred - gt_safe * torch.log(pred.clamp(min=eps) + eps)

    per_neuron = per_neuron_mean(elem, valid)
    return reduce_over_neurons(per_neuron, reduction)
