"""``deepSTRF.training.fit_multi_seed`` — multi-seed init-variance training.

See ``docs/_source/md/fitter.md`` §4.2 for the full design.

This wrapper runs the same Fitter configuration K times under different
seeds and aggregates per-neuron val and test metrics across seeds. It
addresses *initialization variance* — the same data split, the same
hyperparameters, but different initial weights and shuffle order. It is
NOT k-fold or leave-one-stim-out cross-validation; those are separate
TODOs that need a split-factory rather than a seed sweep.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepSTRF.training.fitter import Fitter, _to_scalar
from deepSTRF.training.seed import set_random_seed


LoaderTriple = Tuple[DataLoader, DataLoader, DataLoader]


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------


def _nanstd(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Population std along ``dim``, ignoring NaN entries."""
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    sq = (x - mean) ** 2
    return torch.sqrt(torch.nanmean(sq, dim=dim))


def _stack_metric(values: Sequence[Any]) -> torch.Tensor:
    """Stack a sequence of per-neuron tensors / scalars into ``(n_seeds, ...)``.

    Each entry is either a ``(N,)`` tensor (per-neuron metric) or a scalar
    (already-reduced metric, e.g. ``loss``). Scalars are upcast to 1-d
    length-1 tensors before stacking. All entries must share shape.
    """
    tensors: List[torch.Tensor] = []
    for v in values:
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().float()
            if t.dim() == 0:
                t = t.unsqueeze(0)
        else:
            t = torch.tensor([float(v)])
        tensors.append(t)
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) > 1:
        raise ValueError(
            f"metric tensors across seeds have inconsistent shapes: {shapes}"
        )
    return torch.stack(tensors, dim=0)


# -----------------------------------------------------------------------------
# WandB integration (optional)
# -----------------------------------------------------------------------------


def _start_wandb_run(wandb_kwargs: Mapping[str, Any], seed: int) -> Any:
    """Init one ``wandb`` run for one seed.

    Defaults ``WANDB_MODE=offline`` when the env var is unset, so a fresh
    user with no wandb account can still get local file logging. Run name is
    auto-derived: ``<name>-seed{i}`` if the user passed ``name=``,
    ``<group>-seed{i}`` if they passed ``group=``, else just ``seed{i}``.
    """
    import wandb

    os.environ.setdefault("WANDB_MODE", "offline")
    init_kwargs = dict(wandb_kwargs)
    user_name = init_kwargs.pop("name", None)
    group = init_kwargs.get("group")
    if user_name is not None:
        name = f"{user_name}-seed{seed}"
    elif group is not None:
        name = f"{group}-seed{seed}"
    else:
        name = f"seed{seed}"
    config = dict(init_kwargs.pop("config", {}))
    config["seed"] = seed
    return wandb.init(name=name, config=config, reinit=True, **init_kwargs)


def _make_wandb_log_fn(run) -> Callable[[Mapping[str, Any]], None]:
    """Build a ``log_fn`` that reduces per-neuron tensors to scalars and logs to wandb."""
    def _log(epoch_dict: Mapping[str, Any]) -> None:
        step = int(epoch_dict.get("epoch", 0))
        flat: Dict[str, float] = {}
        for k, v in epoch_dict.items():
            if k == "epoch":
                continue
            flat[k] = _to_scalar(v)
        run.log(flat, step=step)
    return _log


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def fit_multi_seed(
    model_factory: Callable[[int], nn.Module],
    loader_factory: Callable[[int], LoaderTriple],
    n_seeds: int = 5,
    *,
    seeds: Optional[Sequence[int]] = None,
    fitter_kwargs: Optional[Mapping[str, Any]] = None,
    wandb_kwargs: Optional[Mapping[str, Any]] = None,
    set_seed_strict: bool = False,
) -> Dict[str, Any]:
    """Run the same Fitter configuration ``n_seeds`` times under different seeds.

    For each seed: call ``set_random_seed(seed)``, instantiate model and
    ``(train, val, test)`` loaders via the factories, fit a Fitter with
    ``fitter_kwargs``, then re-evaluate post-restore on val and test.

    This is **multi-seed init variance** — same split, different init. NOT
    k-fold CV (that needs a split-factory, separate TODO).

    Parameters
    ----------
    model_factory
        Callable ``seed -> nn.Module``. Called fresh each seed, after
        ``set_random_seed(seed)``, so weight init draws from the seeded RNG.
    loader_factory
        Callable ``seed -> (train_loader, val_loader, test_loader)``. Required
        3-tuple. Called fresh each seed so each run gets a deterministic
        shuffle generator.
    n_seeds
        Number of seeds to run. Ignored if ``seeds`` is given. Default 5.
    seeds
        Explicit seed list. Overrides ``n_seeds``. Default
        ``[0, 1, ..., n_seeds-1]``.
    fitter_kwargs
        Forwarded to ``Fitter(...)``. May not include ``'model'``,
        ``'train_loader'``, or ``'val_loader'``. If ``'ckpt_path'`` is set,
        each seed's checkpoint is saved to ``<stem>_seed{seed}<suffix>`` so
        seeds don't overwrite each other.
    wandb_kwargs
        If given, spawn one wandb run per seed. Keys forwarded to
        ``wandb.init``. ``WANDB_MODE`` defaults to ``offline`` when unset, so
        pure local file logging works with no account / network. Pass
        ``wandb_kwargs={'mode': 'disabled'}`` to no-op (for tests, smoke).
    set_seed_strict
        Forwarded to ``set_random_seed(strict=...)``. Default ``False``.

    Returns
    -------
    results : dict
        Keys:
            - ``'seeds'`` : ``list[int]``
            - ``'per_seed_histories'`` : ``list[list[dict]]`` — raw Fitter
              histories, one list per seed.
            - For each metric ``m`` (default ``cc``, ``cc_norm``, ``loss``):
                - ``'per_seed_val_<m>'``  : ``(n_seeds, N)`` for per-neuron,
                  ``(n_seeds, 1)`` for scalar (``loss``)
                - ``'mean_val_<m>'``      : ``(N,)`` nanmean across seeds
                - ``'std_val_<m>'``       : ``(N,)`` population nanstd across seeds
                - same triple for ``test``.
            - ``'best_seed'`` : ``int`` — seed whose post-fit val ``monitor``
              metric was best (max or min depending on ``mode``).
            - ``'best_state_dict'`` : ``OrderedDict[str, Tensor]`` — deep copy
              of the best seed's post-fit model state.

    Notes
    -----
    "Best" uses the same ``(monitor, mode)`` pair as the per-seed Fitters
    (default ``val_cc_norm`` / ``max``), reduced to a scalar via ``nanmean``
    over the neuron axis.
    """
    if seeds is None:
        seeds = list(range(n_seeds))
    else:
        seeds = list(seeds)
        n_seeds = len(seeds)
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")

    fitter_kwargs = dict(fitter_kwargs or {})
    for k in ("model", "train_loader", "val_loader"):
        if k in fitter_kwargs:
            raise ValueError(
                f"fitter_kwargs[{k!r}] is managed by fit_multi_seed; remove it"
            )
    user_log_fn = fitter_kwargs.pop("log_fn", None)
    base_ckpt = fitter_kwargs.pop("ckpt_path", None)

    monitor = fitter_kwargs.get("monitor", "val_cc_norm")
    mode = fitter_kwargs.get("mode", "max")
    monitor_key = monitor.removeprefix("val_") if monitor.startswith("val_") else monitor

    histories: List[List[Dict[str, Any]]] = []
    val_per_seed: List[Dict[str, Any]] = []
    test_per_seed: List[Dict[str, Any]] = []
    state_dicts: List[Dict[str, torch.Tensor]] = []

    for seed in seeds:
        set_random_seed(seed, strict=set_seed_strict)
        model = model_factory(seed)
        train_loader, val_loader, test_loader = loader_factory(seed)

        run = None
        if wandb_kwargs is not None:
            run = _start_wandb_run(wandb_kwargs, seed=seed)
            log_fn: Callable[[Mapping[str, Any]], None] = _make_wandb_log_fn(run)
        elif user_log_fn is not None:
            log_fn = user_log_fn
        else:
            log_fn = lambda d: None  # silent default for multi-seed sweeps

        fk = dict(fitter_kwargs)
        fk["log_fn"] = log_fn
        if base_ckpt is not None:
            p = Path(base_ckpt)
            fk["ckpt_path"] = p.with_name(f"{p.stem}_seed{seed}{p.suffix}")

        fitter = Fitter(model, train_loader, val_loader, **fk)
        history = fitter.fit()

        val_post = fitter.evaluate(val_loader)
        test_post = fitter.evaluate(test_loader)

        if run is not None:
            run.finish()

        histories.append(history)
        val_per_seed.append(val_post)
        test_per_seed.append(test_post)
        state_dicts.append(copy.deepcopy(fitter.model.state_dict()))

    # Aggregation
    results: Dict[str, Any] = {
        "seeds": list(seeds),
        "per_seed_histories": histories,
    }
    metric_names = list(val_per_seed[0].keys())
    for name in metric_names:
        for prefix, per_seed in (("val", val_per_seed), ("test", test_per_seed)):
            stacked = _stack_metric([p[name] for p in per_seed])
            results[f"per_seed_{prefix}_{name}"] = stacked
            results[f"mean_{prefix}_{name}"] = torch.nanmean(stacked, dim=0)
            results[f"std_{prefix}_{name}"] = _nanstd(stacked, dim=0)

    # Best-of-N: highest (or lowest) post-fit val-monitor scalar.
    if monitor_key not in metric_names:
        raise KeyError(
            f"monitor {monitor!r} (key {monitor_key!r}) not produced by val "
            f"metrics; available: {sorted(metric_names)}"
        )
    monitor_per_seed = torch.tensor(
        [_to_scalar(p[monitor_key]) for p in val_per_seed]
    )
    best_idx = int(
        torch.argmax(monitor_per_seed).item()
        if mode == "max"
        else torch.argmin(monitor_per_seed).item()
    )
    results["best_seed"] = seeds[best_idx]
    results["best_state_dict"] = state_dicts[best_idx]

    return results
