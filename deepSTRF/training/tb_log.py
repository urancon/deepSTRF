"""Opt-in TensorBoard logger for ``fit_multi_seed`` (and any Fitter user).

Same protocol as :mod:`deepSTRF.training.wandb_log` — implements the
``SeedLogger`` duck-typed interface ``__call__`` / ``finalize`` /
``close`` against ``torch.utils.tensorboard.SummaryWriter`` instead of
``wandb.Run``.

Use TensorBoard when:
- You want a local-only, no-account, single-process viewer.
- You're on a machine with no outbound network.

Use WandB when:
- You want a cross-run table view with sortable columns.
- You want cloud-hosted persistence + sharing.

Browse the logs with ``tensorboard --logdir=<your log_dir>``; the URL
defaults to http://localhost:6006.

Examples
--------
>>> from deepSTRF.training import fit_multi_seed
>>> from deepSTRF.training.tb_log import make_tensorboard_logger_factory
>>> results = fit_multi_seed(
...     model_factory=..., loader_factory=..., n_seeds=3,
...     logger_factory=make_tensorboard_logger_factory(
...         log_dir="tb_logs", group="linear-ns1-T9",
...         config={"model": "Linear", "T": 9, "F": 34},
...     ),
... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import torch

from deepSTRF.training.fitter import _to_scalar


# -----------------------------------------------------------------------------
# Per-neuron helpers
# -----------------------------------------------------------------------------


def _log_per_neuron(writer, name: str, t: torch.Tensor, step: int) -> None:
    """Write {mean, p10, p50, p90, hist} for a 1-d per-neuron tensor.

    Mean uses ``nanmean`` (so a few NaN cells don't poison the value).
    Quantiles and histograms are computed over non-NaN cells only.
    """
    t = t.detach().cpu()
    valid = t[~t.isnan()]
    writer.add_scalar(name, float(torch.nanmean(t).item()), step)
    if valid.numel():
        writer.add_scalar(f"{name}/p10", float(torch.quantile(valid, 0.10).item()), step)
        writer.add_scalar(f"{name}/p50", float(torch.quantile(valid, 0.50).item()), step)
        writer.add_scalar(f"{name}/p90", float(torch.quantile(valid, 0.90).item()), step)
        writer.add_histogram(f"{name}/hist", valid, step)


# -----------------------------------------------------------------------------
# TensorBoardSeedLogger
# -----------------------------------------------------------------------------


class TensorBoardSeedLogger:
    """One-seed logger backed by a ``torch.utils.tensorboard.SummaryWriter``.

    Implements the ``SeedLogger`` protocol expected by
    :func:`fit_multi_seed`. Each seed writes to
    ``{log_dir}/{group or 'default'}/{run_name}/`` so a typical
    ``tensorboard --logdir <log_dir>`` invocation surfaces every group ×
    seed combination side by side.

    Parameters
    ----------
    seed
        Seed value. Used in the run name and the ``hparams/seed`` scalar.
    log_dir
        Root directory for event files. Subdirectories per group / seed
        are created automatically.
    group
        Optional group name. Determines the second-level subdirectory.
    name
        Optional user-supplied run name. Final run dir is
        ``{name}-seed{seed}``; falls back to ``{group}-seed{seed}`` then
        ``seed{seed}``.
    config
        Optional dict of hparams. Each JSON-friendly entry is written as
        ``hparams/<key>`` (numeric values as scalars, others via
        ``add_text``). The full config is also dumped as a single
        ``config`` text entry.
    """

    def __init__(
        self,
        seed: int,
        *,
        log_dir: Union[str, Path] = "tb_logs",
        group: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "TensorBoard support requires `torch.utils.tensorboard` — "
                "install via `pip install tensorboard`."
            ) from e

        if name is not None:
            run_name = f"{name}-seed{seed}"
        elif group is not None:
            run_name = f"{group}-seed{seed}"
        else:
            run_name = f"seed{seed}"
        full_dir = Path(log_dir) / (group if group is not None else "default") / run_name
        full_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.run_dir = full_dir
        self.writer = SummaryWriter(log_dir=str(full_dir))

        # Surface the full config as one text dump + per-key scalars/text
        if config:
            self.writer.add_text("config", _format_config(config), 0)
            for k, v in config.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.writer.add_scalar(f"hparams/{k}", float(v), 0)
                else:
                    self.writer.add_text(f"hparams/{k}", str(v), 0)
        self.writer.add_scalar("hparams/seed", float(seed), 0)

    # SeedLogger protocol --------------------------------------------------

    def __call__(self, epoch_dict: Mapping[str, Any]) -> None:
        step = int(epoch_dict.get("epoch", 0))
        for k, v in epoch_dict.items():
            if k == "epoch":
                continue
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.numel() > 1:
                _log_per_neuron(self.writer, k, v, step)
            else:
                self.writer.add_scalar(k, _to_scalar(v), step)

    def finalize(self, final_metrics: Mapping[str, Mapping[str, Any]]) -> None:
        """Write final ``{prefix}_{metric}`` scalars + percentile / histogram tags
        at step 0 (TensorBoard's "Summary" doesn't exist as a separate concept —
        we use a fixed step so they appear as a single point on the chart)."""
        for prefix, metrics in final_metrics.items():
            for k, v in metrics.items():
                full = f"final/{prefix}_{k}"
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.numel() > 1:
                    _log_per_neuron(self.writer, full, v, step=0)
                else:
                    self.writer.add_scalar(full, _to_scalar(v), 0)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def _format_config(config: Mapping[str, Any]) -> str:
    """Pretty-format a config dict as a markdown table for ``add_text``."""
    lines = ["| key | value |", "| --- | --- |"]
    for k, v in config.items():
        lines.append(f"| {k} | {v!r} |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Factory helper
# -----------------------------------------------------------------------------


def make_tensorboard_logger_factory(
    **tb_kwargs: Any,
) -> Callable[[int], TensorBoardSeedLogger]:
    """Build a ``logger_factory`` that returns one ``TensorBoardSeedLogger`` per seed.

    All kwargs are forwarded to :class:`TensorBoardSeedLogger`.
    """
    def _factory(seed: int) -> TensorBoardSeedLogger:
        return TensorBoardSeedLogger(seed=seed, **tb_kwargs)
    return _factory
