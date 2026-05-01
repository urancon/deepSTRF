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

    if log_input:
        elem = torch.exp(pred) - gt * pred
    else:
        if validate_input and (pred.masked_fill(~valid, 0.0) < 0).any():
            raise ValueError(
                "poisson_loss(log_input=False): pred has negative values at "
                "masked-in positions. Either use a non-negative output "
                "activation, set log_input=True (interpret pred as log-rate), "
                "or drop validate_input=True to silently clamp inside log."
            )
        elem = pred - gt * torch.log(pred.clamp(min=eps) + eps)

    per_neuron = per_neuron_mean(elem, valid)
    return reduce_over_neurons(per_neuron, reduction)
