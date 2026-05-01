"""Loss functions for deepSTRF: gradient-bearing, NaN-aware, reduction over neurons.

See ``docs/_source/md/metrics_paradigm.md`` for shape conventions, NaN handling,
and reduction semantics.
"""

from __future__ import annotations

from typing import Optional

import torch

from deepSTRF.metrics._masking import (
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
    pred, gt
        Tensors of identical shape ``(B, N, 1, T)``. ``gt`` may contain NaN.
    mask
        Optional bool tensor broadcastable to ``gt``. If None, defaults to
        ``~gt.isnan()``. If provided, REPLACES (does not augment) the
        NaN-derived mask.
    reduction
        ``'none'`` → ``(N,)``; ``'mean'``/``'sum'`` → scalar via nanmean/nansum.
    """
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
    diff_sq = (pred - gt) ** 2
    per_neuron = per_neuron_mean(diff_sq, valid)
    return reduce_over_neurons(per_neuron, reduction)


def poisson_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative Poisson log-likelihood (without the ``log(rate!)`` constant).

    ``loss = pred − gt · log(pred + eps)`` per element, reduced per-neuron over
    valid positions, then over the neuron axis.

    ``pred`` must be non-negative at masked-in positions; otherwise raises
    ``ValueError`` (loud failure consistent with the data paradigm).
    """
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
    if (pred.masked_fill(~valid, 0.0) < 0).any():
        raise ValueError(
            "poisson_loss requires non-negative pred at all masked-in "
            "positions. Use a non-negative output activation (e.g. Softplus)."
        )
    elem = pred - gt * torch.log(pred + eps)
    per_neuron = per_neuron_mean(elem, valid)
    return reduce_over_neurons(per_neuron, reduction)
