"""``deepSTRF.training.Fitter`` — a thin, opt-in PyTorch training loop.

See ``docs/_source/md/fitter.md`` for the full design contract. This module
implements the canonical 3-line training step from
``metrics_paradigm.md`` §7, plus early stopping, checkpoint selection, and
cross-batch metric accumulation. The class is intentionally short — when
something doesn't fit (multi-GPU, mixed-precision, curricula, ...) the
recommended path is to write the loop, not to extend the Fitter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from deepSTRF.metrics import corrcoef, mse_loss, normalized_corrcoef


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _default_val_metrics() -> Dict[str, Callable]:
    """Canonical val-metric pair: raw and noise-corrected Pearson correlation."""
    return {
        "cc": lambda pred, responses: corrcoef(pred, responses, reduction="none"),
        "cc_norm": lambda pred, responses: normalized_corrcoef(
            pred, responses, method="schoppe", reduction="none"
        ),
    }


def _to_scalar(x: Any) -> float:
    """Reduce a per-neuron tensor to a scalar via ``nanmean``; pass scalars through."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        if x.numel() == 1:
            return float(x.detach().item())
        return float(torch.nanmean(x.detach()).item())
    return float(x)


def _format_epoch(epoch_dict: Mapping[str, Any]) -> None:
    """Default ``log_fn``: ``epoch | k=v | k=v | ...`` to stdout."""
    parts = []
    for k, v in epoch_dict.items():
        if k == "epoch":
            parts.append(f"epoch {int(v):4d}")
        else:
            parts.append(f"{k}={_to_scalar(v):.4f}")
    print(" | ".join(parts))


def _pad_and_cat(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Right-pad each tensor along its last two axes (R, T) with NaN, cat along dim 0.

    All tensors must share dims (B_i, N, R_i, T_i) with the same ``N``. ``B_i``
    may vary (e.g. last partial batch); ``R_i`` and ``T_i`` may vary per batch
    and are padded to the global max with NaN. The NaN pads are dropped by
    every ``deepSTRF.metrics`` function via its NaN-derived mask
    (``metrics_paradigm.md`` §4).
    """
    if not tensors:
        raise ValueError("cannot concatenate an empty list of tensors")
    max_R = max(t.shape[2] for t in tensors)
    max_T = max(t.shape[3] for t in tensors)
    padded = []
    for t in tensors:
        pad_R = max_R - t.shape[2]
        pad_T = max_T - t.shape[3]
        if pad_R or pad_T:
            # F.pad takes (left_T, right_T, left_R, right_R) — back-to-front.
            t = F.pad(t, (0, pad_T, 0, pad_R), value=float("nan"))
        padded.append(t)
    return torch.cat(padded, dim=0)


# -----------------------------------------------------------------------------
# Fitter
# -----------------------------------------------------------------------------


class Fitter:
    """Opt-in training loop for a deepSTRF :class:`NeuralModel`.

    See ``docs/_source/md/fitter.md`` for the full design.

    Parameters
    ----------
    model
        Any ``nn.Module`` whose ``forward`` emits ``(B, N, 1, T)`` predictions.
        Stateful models may implement ``model.detach()`` (no-op by default on
        ``deepSTRF.models.NeuralModel``); the Fitter calls it after every step.
    train_loader, val_loader
        ``DataLoader`` instances built with ``deepSTRF.utils.data.neural_collate``.
        ``val_loader`` with ``batch_size=1`` is the simplest case but any
        batch size works thanks to NaN-pad-and-cat (§6).
    loss_fn
        Callable ``(pred, responses) -> Tensor``. Default ``mse_loss``. The
        deepSTRF losses auto-collapse ``responses`` to PSTH internally
        (``metrics_paradigm.md`` §2), so no caller-side ``nanmean`` is needed.
    val_metrics
        Mapping ``name -> callable(pred, responses) -> per-neuron Tensor``.
        Default: the canonical ``{'cc', 'cc_norm'}`` pair. Stored under
        ``f'val_{name}'`` in the epoch dict.
    optimizer
        Any ``torch.optim.Optimizer``. Default: ``AdamW(model.parameters(),
        lr=1e-3, weight_decay=1e-4)``.
    device
        Where to place the model and per-batch tensors.
    max_epochs
        Hard cap on training epochs.
    patience
        Early-stop patience: number of epochs without improvement on
        ``monitor`` before the loop terminates.
    monitor
        Key in the per-epoch dict to track for early stopping. Default
        ``'val_cc_norm'``. Use ``'val_loss'``, ``'val_cc'``, or any custom
        key you added via ``val_metrics``.
    mode
        ``'max'`` or ``'min'`` — direction of improvement on ``monitor``.
        Default ``'max'`` (paired with ``'val_cc_norm'``).
    ckpt_path
        If given, save the best-on-``monitor`` ``state_dict`` to this path
        and restore it at the end of ``fit()``.
    log_fn
        Called as ``log_fn(epoch_dict)`` once per epoch. Default: a small
        formatter that prints ``epoch | k=v | ...``. Override to log to
        WandB, MLflow, a file, etc.
    track_train_metrics
        If ``True`` (default), recompute ``val_metrics`` over the training
        predictions accumulated this epoch and add them to the epoch dict
        as ``'train_<name>'``. Useful for diagnosing overfitting but
        expensive on large datasets — accumulating ``(B, N, R, T)``
        responses across all train batches is the dominant per-epoch cost
        when ``N × R × T`` is in the millions (e.g. AA2's 494-cell
        population). Set to ``False`` to skip; ``train_loss`` is always
        reported.
    track_per_cell_best
        If ``True``, maintain a per-cell best-on-``monitor`` snapshot of
        the readout's per-N parameter and buffer slices throughout
        training. At end-of-fit, after the global ``ckpt_path`` restore,
        each cell's slice is overlaid with its individual-best snapshot.
        On no-shared-params models this is **strictly** at least as good
        as the vanilla restore on the validation set, cell-by-cell, by
        construction — every cell ends up at its individual val peak.
        The training trajectory itself is unchanged (no gradient masking,
        no per-cell stopping); the only difference is which checkpoint
        is restored at end. Requires ``val_metrics[monitor.removeprefix
        ('val_')]`` to return a ``(N,)`` per-cell tensor (the default
        :func:`_default_val_metrics` does this). Default ``False``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
        val_metrics: Optional[Dict[str, Callable]] = None,
        optimizer: Optional[Optimizer] = None,
        device: Union[str, torch.device] = "cpu",
        max_epochs: int = 1000,
        patience: int = 10,
        monitor: str = "val_cc_norm",
        mode: str = "max",
        ckpt_path: Optional[Union[str, Path]] = None,
        log_fn: Callable[[Mapping[str, Any]], None] = _format_epoch,
        track_train_metrics: bool = True,
        track_per_cell_best: bool = False,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.val_metrics = (
            val_metrics if val_metrics is not None else _default_val_metrics()
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        )
        self.max_epochs = max_epochs
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self.log_fn = log_fn
        self.track_train_metrics = track_train_metrics
        self.track_per_cell_best = track_per_cell_best

    # ------------------------------------------------------------------
    # Hooks (subclass and override, or pass kwargs at construction time)
    # ------------------------------------------------------------------

    def compute_loss(
        self, pred: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        """Default: delegate to ``self.loss_fn(pred, responses)`` (auto-PSTH inside)."""
        return self.loss_fn(pred, responses)

    def on_epoch_end(self, epoch: int, epoch_dict: Dict[str, Any]) -> None:
        """Default: log the epoch dict via ``self.log_fn``."""
        self.log_fn(epoch_dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> List[Dict[str, Any]]:
        """Train until ``max_epochs`` or early-stop on ``monitor``.

        Returns
        -------
        history : list of dict
            One dict per completed epoch, with keys ``'epoch'``, ``'train_*'``,
            and ``'val_*'``.
        """
        history: List[Dict[str, Any]] = []
        best_score = -float("inf") if self.mode == "max" else float("inf")
        better = (
            (lambda new, best: new > best)
            if self.mode == "max"
            else (lambda new, best: new < best)
        )
        epochs_no_improvement = 0

        if self.track_per_cell_best:
            N = self.model.O
            self._per_cell_best_score = torch.full(
                (N,),
                -float("inf") if self.mode == "max" else float("inf"),
                device=self.device,
            )
            self._per_cell_snapshots: Dict[int, List[torch.Tensor]] = {}

        for epoch in range(self.max_epochs):
            train = self._train_one_epoch()
            val = self._evaluate(self.val_loader)
            epoch_dict: Dict[str, Any] = {"epoch": epoch}
            epoch_dict.update({f"train_{k}": v for k, v in train.items()})
            epoch_dict.update({f"val_{k}": v for k, v in val.items()})
            history.append(epoch_dict)
            self.on_epoch_end(epoch, epoch_dict)

            if self.monitor not in epoch_dict:
                raise KeyError(
                    f"monitor key {self.monitor!r} not in epoch dict; "
                    f"available keys: {sorted(epoch_dict)}"
                )

            # Per-cell snapshot update happens BEFORE the global best/patience
            # update so that snapshots track each cell's individual best
            # regardless of population-level early-stop behaviour. Needs the
            # per-cell monitor tensor; raises if it isn't one.
            if self.track_per_cell_best:
                self._update_per_cell_best(epoch_dict[self.monitor])

            score = _to_scalar(epoch_dict[self.monitor])
            if better(score, best_score):
                best_score = score
                if self.ckpt_path is not None:
                    self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.ckpt_path)
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= self.patience:
                break

        if self.ckpt_path is not None and self.ckpt_path.exists():
            self.model.load_state_dict(
                torch.load(self.ckpt_path, map_location=self.device)
            )

        # Overlay per-cell snapshots on top of the global-ckpt restore.
        # Order is intentional: the global restore resets every parameter
        # (including non-readout ones like the model's core) to the
        # population-best state; the per-cell overlay then replaces each
        # cell's readout slice with its individual-best.
        if self.track_per_cell_best:
            self._restore_per_cell_snapshots()

        return history

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Run loss + ``val_metrics`` on a loader (no backprop, no key prefix).

        Returns a dict with keys ``'loss'`` plus each entry of ``self.val_metrics``.
        For test-set evaluation after training: ``fitter.evaluate(test_loader)``.
        """
        return self._evaluate(loader)

    # ------------------------------------------------------------------
    # Per-cell snapshot bookkeeping (only used when track_per_cell_best=True)
    # ------------------------------------------------------------------

    def _per_cell_readout_tensors(self):
        """Yield every readout tensor whose leading axis is the neuron axis.

        Iterates both ``parameters()`` and ``buffers()`` under
        ``self.model.readout``, filtering for ``shape[0] == self.model.O``.
        On a no-shared-params readout (STRF kernel + per-neuron BN +
        per-neuron activation, the post-2026-05-19 audio convention)
        this yields every learnable scalar in the readout.
        """
        N = self.model.O
        for p in self.model.readout.parameters():
            if p.dim() >= 1 and p.shape[0] == N:
                yield p
        for b in self.model.readout.buffers():
            # skip 0-d scalar buffers (e.g. BN's num_batches_tracked)
            if b.dim() >= 1 and b.shape[0] == N:
                yield b

    def _update_per_cell_best(self, score: Any) -> None:
        N = self.model.O
        if not isinstance(score, torch.Tensor) or score.shape != (N,):
            raise ValueError(
                f"track_per_cell_best=True requires the {self.monitor!r} "
                f"val metric to return a per-cell tensor of shape ({N},); "
                f"got {type(score).__name__} with shape "
                f"{tuple(score.shape) if isinstance(score, torch.Tensor) else None!r}. "
                f"Use val_metrics callables with reduction='none'."
            )
        score = score.to(self.device)
        better = (
            (lambda new, best: new > best)
            if self.mode == "max"
            else (lambda new, best: new < best)
        )
        improved = better(score, self._per_cell_best_score) & ~score.isnan()
        for n in torch.nonzero(improved, as_tuple=True)[0].tolist():
            self._per_cell_snapshots[n] = [
                p.data[n].detach().clone()
                for p in self._per_cell_readout_tensors()
            ]
        self._per_cell_best_score = torch.where(
            improved, score, self._per_cell_best_score
        )

    def _restore_per_cell_snapshots(self) -> None:
        for n, snap in self._per_cell_snapshots.items():
            for p, s in zip(self._per_cell_readout_tensors(), snap):
                p.data[n] = s

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> Dict[str, Any]:
        self.model.train()
        loss_sum = 0.0
        n_batches = 0
        preds_list: List[torch.Tensor] = []
        responses_list: List[torch.Tensor] = []

        for batch in self.train_loader:
            stims, responses, _valid_mask, _stim_metas = batch
            stims = stims.to(self.device)
            responses = responses.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(stims)
            loss = self.compute_loss(pred, responses)
            loss.backward()
            self.optimizer.step()
            if hasattr(self.model, "detach"):
                self.model.detach()

            loss_sum += float(loss.detach().item())
            n_batches += 1
            # Accumulate on CPU so the concatenated metrics tensor does not
            # have to fit in GPU memory — for large datasets (494 cells × 81
            # train stims × 20 trials × 511 frames on AA2) the GPU concat
            # exceeds typical visible memory by several gigabytes. Skip the
            # accumulation entirely when train-side metrics are disabled —
            # halves wall time on large datasets where users only care
            # about val metrics.
            if self.track_train_metrics:
                preds_list.append(pred.detach().cpu())
                responses_list.append(responses.detach().cpu())

        out: Dict[str, Any] = {"loss": loss_sum / max(n_batches, 1)}
        if self.track_train_metrics:
            with torch.no_grad():
                preds_cat = _pad_and_cat(preds_list)
                responses_cat = _pad_and_cat(responses_list)
                for name, fn in self.val_metrics.items():
                    out[name] = fn(preds_cat, responses_cat)
        return out

    def _evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        preds_list: List[torch.Tensor] = []
        responses_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                stims, responses, _valid_mask, _stim_metas = batch
                stims = stims.to(self.device)
                responses = responses.to(self.device)

                pred = self.model(stims)
                if hasattr(self.model, "detach"):
                    self.model.detach()

                # Same CPU-accumulation as in _train_one_epoch — see comment
                # there for the AA2-scale memory rationale.
                preds_list.append(pred.cpu())
                responses_list.append(responses.cpu())

            preds_cat = _pad_and_cat(preds_list)
            responses_cat = _pad_and_cat(responses_list)

            out: Dict[str, Any] = {
                "loss": float(self.compute_loss(preds_cat, responses_cat).item()),
            }
            for name, fn in self.val_metrics.items():
                out[name] = fn(preds_cat, responses_cat)
        return out
