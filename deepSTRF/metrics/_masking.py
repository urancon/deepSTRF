"""Internal NaN-mask helpers for ``deepSTRF.metrics``. Not public API.

See ``docs/_source/md/metrics_paradigm.md`` §4–5 for the design rationale.
"""

from __future__ import annotations

from typing import Optional

import torch

_VALID_REDUCTIONS = ("none", "mean", "sum")


def resolve_mask(gt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Return the effective valid-positions mask for ``gt``.

    If ``mask`` is None, defaults to ``~gt.isnan()``. If ``mask`` is provided,
    it REPLACES (does not augment) the NaN-derived mask — the caller is
    responsible for AND-ing with ``~gt.isnan()`` themselves if NaN protection
    is needed.
    """
    if mask is None:
        return ~gt.isnan()
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be a bool tensor, got dtype={mask.dtype}")
    if mask.shape != gt.shape:
        try:
            mask = mask.expand_as(gt)
        except RuntimeError as exc:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} not broadcastable to gt shape "
                f"{tuple(gt.shape)}"
            ) from exc
    return mask


def reduce_over_neurons(per_neuron: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply the public ``reduction`` kwarg to a 1-D per-neuron tensor.

    NaN-tolerant: empty cells (NaN in ``per_neuron``) are dropped under
    ``mean``/``sum`` so the scalar reduction stays well-defined.
    """
    if reduction == "none":
        return per_neuron
    if reduction == "mean":
        return torch.nanmean(per_neuron)
    if reduction == "sum":
        return torch.nansum(per_neuron)
    raise ValueError(
        f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}"
    )


def per_neuron_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Vectorised, NaN-loud per-neuron mean over (B, R, T).

    Inputs of shape ``(B, N, R, T)``; returns ``(N,)``. Where ``valid`` is False
    the contribution is zeroed out, so masked-out NaNs don't propagate. Where
    ``valid`` is True but ``values`` is NaN (= the user passed an override mask
    that didn't drop NaN positions), NaN propagates into the per-neuron value
    by design — this is the documented loud-failure mode.
    """
    if values.shape != valid.shape:
        raise ValueError(
            f"values shape {tuple(values.shape)} must equal valid shape "
            f"{tuple(valid.shape)}"
        )
    kept = values.masked_fill(~valid, 0.0)
    counts = valid.sum(dim=(0, 2, 3))                     # (N,)
    sums = kept.sum(dim=(0, 2, 3))                        # (N,)
    safe_counts = counts.clamp(min=1).to(sums.dtype)
    per_neuron = sums / safe_counts
    nan = per_neuron.new_full((), float("nan"))
    return torch.where(counts > 0, per_neuron, nan)
