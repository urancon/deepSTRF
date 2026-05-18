"""``PerNeuronFitter`` — per-cell early-stopping variant of :class:`Fitter`.

Tailored for Linear-family models (``Linear`` / ``LinearNonlinear`` /
``NetworkReceptiveField``) whose readout has independent per-neuron
parameter slices and whose core is small / cheap. Each neuron tracks
its own monitor history and patience counter; when its patience
expires, the neuron's readout parameters are snapshotted and its
contribution to the training-loss gradient is masked to zero.
At the end of ``fit()``, snapshots are restored so any optimizer
drift on frozen parameters is undone.

**Not recommended for models with a non-trivial shared core** (DNet,
ConvNet2D, Transformer, StateNet): the core continues to evolve
after a neuron is frozen, so the neuron's predictions at the end of
training can differ from those at the moment of freezing — only the
*readout* state is snapshotted, not the core that produces its input.
For those models the regular :class:`Fitter` with global early
stopping (and explicit readout regularization, eventually) is the
right tool.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from deepSTRF.training.fitter import Fitter, _pad_and_cat


class PerNeuronFitter(Fitter):
    """Per-cell early stopping for L/LN-family models.

    Same constructor as :class:`Fitter`. The behavioural differences:

    1. The ``monitor`` key must resolve to a per-neuron tensor of
       shape ``(N,)`` (typical for ``val_metrics`` callables that
       use ``reduction='none'`` — the canned default already does
       so).
    2. Each cell has its own best-score history, patience counter,
       and frozen flag. Cells freeze independently when their
       patience expires.
    3. Frozen cells contribute zero to the per-cell training loss.
       The Fitter's per-batch loss is the mean of the per-cell
       MSE/Poisson/etc. residuals over **active** (non-frozen) cells.
    4. **Every** cell's readout parameter slices are snapshotted at
       its best-on-``monitor`` epoch (mirroring :class:`Fitter`'s
       ``ckpt_path`` semantics, but per-cell). At the end of
       ``fit()`` every cell is restored to its snapshot, so the
       returned model is the population of per-cell best-on-val
       states — *not* whatever the optimizer drifted to before the
       patience window or after freezing.

    Training stops when all cells are frozen, or when ``max_epochs``
    is reached.
    """

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _per_cell_param_slices(self):
        """Yield every readout *tensor* whose leading axis is the neuron
        axis ``N = self.model.O`` — both ``Parameter``s and ``Buffer``s.

        Convention: deepSTRF readouts (``LinearReadout``, ``STRFReadout``)
        store per-neuron parameters with ``N`` as the leading axis. We
        skip tensors that don't satisfy this — the only realistic
        candidates filtered out are scalar buffers (e.g. BatchNorm's
        ``num_batches_tracked``) and shared regularization terms.

        Yielding buffers as well as parameters matters for readouts that
        embed normalization layers with per-neuron running statistics
        (e.g. ``BatchNorm1d(N)`` after the STRF kernel): without buffer
        snapshotting, restored per-cell parameters would sit on top of
        running statistics that kept drifting after the cell froze.
        """
        N = self.model.O
        for p in self.model.readout.parameters():
            if p.dim() >= 1 and p.shape[0] == N:
                yield p
        for b in self.model.readout.buffers():
            # skip 0-d scalar buffers (e.g. BN's num_batches_tracked)
            if b.dim() >= 1 and b.shape[0] == N:
                yield b

    def _snapshot_cell(self, n: int) -> List[torch.Tensor]:
        return [p.data[n].detach().clone() for p in self._per_cell_param_slices()]

    def _restore_cell(self, n: int, snap: List[torch.Tensor]) -> None:
        for p, s in zip(self._per_cell_param_slices(), snap):
            p.data[n] = s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> List[Dict[str, Any]]:
        N = self.model.O
        device = self.device

        better = (
            (lambda new, best: new > best)
            if self.mode == "max"
            else (lambda new, best: new < best)
        )
        # Per-cell bests start at the worst possible value
        best_score = torch.full(
            (N,),
            -float("inf") if self.mode == "max" else float("inf"),
            device=device,
        )
        no_improve = torch.zeros(N, dtype=torch.long, device=device)
        self._frozen_cells = torch.zeros(N, dtype=torch.bool, device=device)
        snapshots: Dict[int, List[torch.Tensor]] = {}

        history: List[Dict[str, Any]] = []

        for epoch in range(self.max_epochs):
            train = self._train_one_epoch_masked()
            val = self._evaluate(self.val_loader)

            # Build the in-progress epoch_dict so that monitor lookup below
            # finds e.g. 'val_cc_norm'. The frozen-cell counter is added
            # *after* the freeze update for this epoch.
            epoch_dict: Dict[str, Any] = {"epoch": epoch}
            epoch_dict.update({f"train_{k}": v for k, v in train.items()})
            epoch_dict.update({f"val_{k}": v for k, v in val.items()})

            if self.monitor not in epoch_dict:
                raise KeyError(
                    f"monitor key {self.monitor!r} not in epoch dict; "
                    f"available keys: {sorted(epoch_dict)}"
                )
            score = epoch_dict[self.monitor]
            if not isinstance(score, torch.Tensor) or score.shape != (N,):
                raise ValueError(
                    f"PerNeuronFitter requires a per-cell monitor tensor of shape "
                    f"({N},); got {type(score).__name__} with shape "
                    f"{tuple(score.shape) if isinstance(score, torch.Tensor) else None!r}. "
                    f"Use val_metrics callables with reduction='none'."
                )
            score = score.to(device)
            # Treat NaN as 'no improvement' (cell hasn't been measured)
            improved = better(score, best_score) & ~score.isnan()

            # Snapshot every still-active cell that improved on this
            # epoch. The snapshot is taken AFTER the train step that
            # produced the new ``score``, so restoring it at the end
            # of fit() lands the cell exactly where it was when it
            # achieved its best ``monitor`` value — mirroring
            # ``Fitter``'s ``ckpt_path`` semantics, but per-cell.
            for n in torch.nonzero(
                improved & ~self._frozen_cells, as_tuple=True
            )[0].tolist():
                snapshots[n] = self._snapshot_cell(n)

            best_score = torch.where(improved, score, best_score)
            no_improve = torch.where(
                improved, torch.zeros_like(no_improve), no_improve + 1
            )

            # Newly frozen cells: patience exhausted and not already frozen.
            # No snapshot taken here — the snapshot from the cell's
            # best-improvement epoch is what we want to restore at end.
            newly_frozen = (no_improve >= self.patience) & ~self._frozen_cells
            self._frozen_cells |= newly_frozen

            # Now record the post-update frozen-cell count and append.
            epoch_dict["frozen_cells"] = int(self._frozen_cells.sum().item())
            history.append(epoch_dict)
            self.on_epoch_end(epoch, epoch_dict)

            if bool(self._frozen_cells.all().item()):
                break

        # Restore every cell that has a snapshot (frozen or still active
        # at max_epochs / all-frozen termination) to its best-on-monitor
        # state. Cells whose monitor was NaN throughout never improved
        # past the -inf init and have no snapshot — leave them untouched.
        for n, snap in snapshots.items():
            self._restore_cell(n, snap)

        return history

    # ------------------------------------------------------------------
    # Training step with masked per-cell loss
    # ------------------------------------------------------------------

    def _train_one_epoch_masked(self) -> Dict[str, Any]:
        self.model.train()
        loss_sum = 0.0
        n_batches = 0
        preds_list: List[torch.Tensor] = []
        responses_list: List[torch.Tensor] = []

        active_mask = (~self._frozen_cells).float()                # (N,)
        n_active = max(int(active_mask.sum().item()), 1)

        for batch in self.train_loader:
            stims, responses, _vm, _meta = batch
            stims = stims.to(self.device)
            responses = responses.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(stims)
            # Per-cell loss vector (N,)
            per_cell = self.loss_fn(pred, responses, reduction="none")
            # Mask frozen cells out of the gradient signal
            loss = (per_cell * active_mask).sum() / n_active
            loss.backward()
            self.optimizer.step()
            if hasattr(self.model, "detach"):
                self.model.detach()

            loss_sum += float(loss.detach().item())
            n_batches += 1
            preds_list.append(pred.detach())
            responses_list.append(responses.detach())

        out: Dict[str, Any] = {"loss": loss_sum / max(n_batches, 1)}
        with torch.no_grad():
            preds_cat = _pad_and_cat(preds_list)
            responses_cat = _pad_and_cat(responses_list)
            for name, fn in self.val_metrics.items():
                out[name] = fn(preds_cat, responses_cat)
        return out
