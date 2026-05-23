"""Opt-in WandB logger for ``fit_multi_seed`` (and any Fitter user).

This module is a thin convenience wrapper. The library does not require
WandB for training — ``fit_multi_seed`` only knows about a generic
``logger_factory(seed) -> SeedLogger`` protocol. ``WandbSeedLogger``
implements that protocol with one ``wandb.init`` per seed.

Build a factory with :func:`make_wandb_logger_factory` and pass it as
``logger_factory=`` to :func:`deepSTRF.training.fit_multi_seed`.

Examples
--------
>>> from deepSTRF.training import fit_multi_seed
>>> from deepSTRF.training.wandb_log import make_wandb_logger_factory
>>> results = fit_multi_seed(
...     model_factory=..., loader_factory=..., n_seeds=3,
...     logger_factory=make_wandb_logger_factory(
...         project="deepstrf", entity="urancon",
...         group="ns1-linear", mode="offline",
...     ),
... )

If WandB is not installed, importing this module raises ``ImportError``
with a clear hint; the rest of ``deepSTRF.training`` keeps working.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Optional

import torch

from deepSTRF.training.fitter import _to_scalar


# -----------------------------------------------------------------------------
# Per-neuron summarisation
# -----------------------------------------------------------------------------


def _per_neuron_scalars(name: str, t: torch.Tensor) -> dict:
    """Reduce a 1-d per-neuron tensor to a {mean, p10, p50, p90} scalar dict
    keyed ``{name}``, ``{name}/p10``, ``{name}/p50``, ``{name}/p90``.

    NaN cells are dropped before quantiles; ``nanmean`` is used for the mean.
    Returns ``{name: nan, ...}`` placeholders if every cell is NaN.
    """
    t = t.detach().cpu()
    valid = t[~t.isnan()]
    out: dict = {name: float(torch.nanmean(t).item())}
    if valid.numel():
        out[f"{name}/p10"] = float(torch.quantile(valid, 0.10).item())
        out[f"{name}/p50"] = float(torch.quantile(valid, 0.50).item())
        out[f"{name}/p90"] = float(torch.quantile(valid, 0.90).item())
    else:
        out[f"{name}/p10"] = float("nan")
        out[f"{name}/p50"] = float("nan")
        out[f"{name}/p90"] = float("nan")
    return out


def _make_histogram(t: torch.Tensor):
    """Return a ``wandb.Histogram`` of the non-NaN cells, or ``None`` if empty."""
    import wandb
    valid = t.detach().cpu()
    valid = valid[~valid.isnan()]
    if valid.numel() == 0:
        return None
    return wandb.Histogram(valid.numpy())


# -----------------------------------------------------------------------------
# WandbSeedLogger — one wandb run per seed
# -----------------------------------------------------------------------------


class WandbSeedLogger:
    """One-seed logger backed by a single ``wandb.Run``.

    Implements the duck-typed ``SeedLogger`` protocol expected by
    :func:`fit_multi_seed`:

    - ``__call__(epoch_dict)`` is invoked once per epoch; per-neuron
      tensors are summarised as ``{mean, p10, p50, p90}`` scalars
      plus a per-epoch ``wandb.Histogram``.
    - ``finalize(final_metrics)`` is invoked once at end of seed, after
      ``Fitter.fit()`` and the post-fit val + test ``evaluate`` passes.
      It writes test metrics + per-neuron summaries to
      ``run.summary`` (so they show up as sortable run-level columns).
    - ``close()`` calls ``run.finish()``.

    Parameters
    ----------
    seed
        The seed value. Added to ``wandb.config`` and used to derive the
        run name.
    project, entity, group
        Forwarded to ``wandb.init``. Run name is auto-derived as
        ``<name>-seed{seed}`` if ``name=`` is set, else
        ``<group>-seed{seed}`` if ``group=`` is set, else ``seed{seed}``.
    name, dir, mode, config, **wandb_init_kwargs
        Forwarded to ``wandb.init`` verbatim. ``WANDB_MODE`` defaults to
        ``offline`` if the env var is unset and ``mode=`` is not given,
        so logging works with no account / network.
    """

    def __init__(
        self,
        seed: int,
        *,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        dir: Optional[str] = None,
        mode: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        **wandb_init_kwargs: Any,
    ) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is required for WandbSeedLogger; "
                "`pip install wandb`."
            ) from e

        if mode is None:
            os.environ.setdefault("WANDB_MODE", "offline")
        if name is not None:
            run_name = f"{name}-seed{seed}"
        elif group is not None:
            run_name = f"{group}-seed{seed}"
        else:
            run_name = f"seed{seed}"
        cfg = dict(config or {})
        cfg["seed"] = seed

        init_kwargs = dict(wandb_init_kwargs)
        if mode is not None:
            init_kwargs["mode"] = mode
        if dir is not None:
            init_kwargs["dir"] = dir

        self.seed = seed
        self.run = wandb.init(
            project=project, entity=entity, group=group,
            name=run_name, config=cfg, reinit=True,
            **init_kwargs,
        )

    # SeedLogger protocol --------------------------------------------------

    def __call__(self, epoch_dict: Mapping[str, Any]) -> None:
        step = int(epoch_dict.get("epoch", 0))
        flat: dict = {}
        for k, v in epoch_dict.items():
            if k == "epoch":
                continue
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.numel() > 1:
                flat.update(_per_neuron_scalars(k, v))
                hist = _make_histogram(v)
                if hist is not None:
                    flat[f"{k}/hist"] = hist
            else:
                flat[k] = _to_scalar(v)
        self.run.log(flat, step=step)

    def finalize(self, final_metrics: Mapping[str, Mapping[str, Any]]) -> None:
        """Write ``{prefix}_{metric}`` summary rows + percentile scalars.

        ``final_metrics`` keys are split prefixes (e.g. ``"val"``, ``"test"``)
        each holding a dict of metric_name -> scalar-or-per-neuron-tensor.
        """
        summary: dict = {}
        for prefix, metrics in final_metrics.items():
            for k, v in metrics.items():
                full = f"{prefix}_{k}"
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.numel() > 1:
                    summary.update(_per_neuron_scalars(full, v))
                    hist = _make_histogram(v)
                    if hist is not None:
                        summary[f"{full}/hist"] = hist
                else:
                    summary[full] = _to_scalar(v)
        self.run.summary.update(summary)

    def close(self) -> None:
        self.run.finish()


# -----------------------------------------------------------------------------
# Factory helper
# -----------------------------------------------------------------------------


def make_wandb_logger_factory(**wandb_kwargs: Any) -> Callable[[int], WandbSeedLogger]:
    """Build a ``logger_factory`` that returns one ``WandbSeedLogger`` per seed.

    All kwargs are forwarded to :class:`WandbSeedLogger`. Use as
    ``fit_multi_seed(..., logger_factory=make_wandb_logger_factory(project="...", ...))``.
    """
    def _factory(seed: int) -> WandbSeedLogger:
        return WandbSeedLogger(seed=seed, **wandb_kwargs)
    return _factory
