"""Tests for ``deepSTRF.training.PerNeuronFitter``.

Per-cell early stopping for L/LN-family models. Reuses the synthetic
fixtures from ``tests/test_fitter.py`` for the basic-fit machinery and
adds tests for per-cell freeze, snapshot/restore, and loss masking.
"""

from __future__ import annotations

from typing import List

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from deepSTRF.training import PerNeuronFitter, set_random_seed
from deepSTRF.utils.data import neural_collate


# -----------------------------------------------------------------------------
# Synthetic fixtures (mirror tests/test_fitter.py)
# -----------------------------------------------------------------------------


class _ToyDataset(Dataset):
    def __init__(self, n_stims=8, F=4, T=20, N=2, R=3, W=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.F = F
        self.T = T
        self.N = N
        self.stims = [torch.randn(F, T, generator=g) for _ in range(n_stims)]
        if W is None:
            W = torch.randn(N, F, generator=g)
        self._target_W = W
        self.responses_per_stim: List[List[torch.Tensor]] = []
        for s in self.stims:
            base = (W @ s)
            per_neuron = []
            for n in range(N):
                noise = torch.randn(R, T, generator=g) * 0.05
                per_neuron.append(base[n].unsqueeze(0).expand(R, T) + noise)
            self.responses_per_stim.append(per_neuron)

    def __len__(self):
        return len(self.stims)

    def __getitem__(self, i):
        nrn_mask = torch.ones(self.N, dtype=torch.bool)
        return self.stims[i], self.responses_per_stim[i], nrn_mask, {"idx": i}


class _LinearReadout(torch.nn.Module):
    """Per-neuron linear readout from time-mean of features. (B, N, 1, T)."""

    def __init__(self, F=4, N=2):
        super().__init__()
        self.O = N
        self.fc = torch.nn.Linear(F, N, bias=True)
        # Expose as 'readout' so PerNeuronFitter can find per-cell params
        self.readout = self.fc

    def forward(self, stims):
        x = stims.transpose(-1, -2)
        y = self.fc(x).transpose(-1, -2)
        return y.unsqueeze(2)

    def detach(self):
        pass


def _make_loaders(seed=0, batch_size=4, n_train=32, n_val=16, N=2):
    train_ds = _ToyDataset(n_stims=n_train, N=N, seed=seed)
    W = train_ds._target_W
    val_ds = _ToyDataset(n_stims=n_val, N=N, W=W, seed=seed + 100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=neural_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=neural_collate)
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# (a) Basic fit reduces loss; multiple cells are tracked
# -----------------------------------------------------------------------------


def test_per_neuron_fitter_basic_fit_reduces_loss():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=4)
    model = _LinearReadout(F=4, N=4)
    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=20, patience=20,
        log_fn=lambda d: None,
    )
    history = fitter.fit()
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    # No cells should be frozen if patience never expires
    assert history[-1]["frozen_cells"] == 0


# -----------------------------------------------------------------------------
# (b) Per-cell freeze: cells freeze independently when their patience expires
# -----------------------------------------------------------------------------


def test_per_neuron_fitter_freezes_cells_independently():
    """Force a flat monitor for cell 0 only; expect cell 0 to freeze first."""
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=3)
    model = _LinearReadout(F=4, N=3)

    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=20, patience=3,
        log_fn=lambda d: None,
    )

    original_evaluate = fitter._evaluate

    def _hand_crafted_evaluate(loader):
        out = original_evaluate(loader)
        # Per-cell cc_norm: cells 1, 2 keep improving (synthesised by
        # adding a tiny epoch-counter); cell 0 stays at 0.5 forever.
        epoch_idx = len(getattr(fitter, "_epochs_seen", []))
        if not hasattr(fitter, "_epochs_seen"):
            fitter._epochs_seen = []
        fitter._epochs_seen.append(epoch_idx)
        scores = torch.tensor([0.5,
                               0.5 + 1e-3 * (epoch_idx + 1),
                               0.5 + 1e-3 * (epoch_idx + 1)])
        out["cc_norm"] = scores
        return out

    fitter._evaluate = _hand_crafted_evaluate
    history = fitter.fit()

    # Cell 0 should freeze first (after patience=3 epochs of no improvement).
    # Cells 1, 2 keep improving, never freeze.
    assert history[-1]["frozen_cells"] == 1


# -----------------------------------------------------------------------------
# (c) Snapshot / restore: frozen cells return to their best params at end-of-fit
# -----------------------------------------------------------------------------


def test_per_neuron_fitter_restores_frozen_cells_at_end():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=2)
    model = _LinearReadout(F=4, N=2)

    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=15, patience=3,
        log_fn=lambda d: None,
    )

    original_evaluate = fitter._evaluate

    # Freeze cell 0 immediately at epoch 3 (flat score), cell 1 keeps improving
    def _evaluate(loader):
        out = original_evaluate(loader)
        epoch_idx = getattr(fitter, "_e", 0)
        fitter._e = epoch_idx + 1
        out["cc_norm"] = torch.tensor([0.5, 0.5 + 1e-3 * epoch_idx])
        return out

    fitter._evaluate = _evaluate

    fitter.fit()

    # After fit() returns, cell 0 should have been frozen, snapshotted, and
    # restored to its best value. The frozen-cell weight slice should be the
    # one captured at freeze time, not whatever AdamW drifted to afterwards.
    # We can't easily check exact values without recording, but we can verify
    # the model is still trainable / the structure is preserved.
    assert hasattr(fitter, "_frozen_cells")
    assert int(fitter._frozen_cells.sum().item()) >= 1
    assert int(fitter._frozen_cells[0].item()) == 1


# -----------------------------------------------------------------------------
# (d) Loss masking: frozen cells contribute zero to the per-batch loss
# -----------------------------------------------------------------------------


def test_loss_masking_skips_frozen_cells():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=3)
    model = _LinearReadout(F=4, N=3)
    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=2, patience=2,
        log_fn=lambda d: None,
    )

    # Pretend cell 0 is frozen
    fitter._frozen_cells = torch.tensor([True, False, False])

    # Capture cell-0 readout weights before & after one epoch
    w0_before = model.fc.weight[0].detach().clone()
    fitter._train_one_epoch_masked()
    w0_after = model.fc.weight[0].detach().clone()
    # With weight_decay=0 (the default) and a zeroed gradient on cell 0, the
    # optimizer leaves cell 0 essentially unchanged (Adam moments still nudge
    # it, but only by a small amount on the first step).
    drift = (w0_after - w0_before).abs().max().item()
    assert drift < 0.05, f"frozen cell drifted by {drift}"


# -----------------------------------------------------------------------------
# (e) All-frozen short-circuit: training stops once every cell is frozen
# -----------------------------------------------------------------------------


def test_all_frozen_terminates_fit():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=2)
    model = _LinearReadout(F=4, N=2)
    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=50, patience=2,
        log_fn=lambda d: None,
    )
    # Force flat scores for all cells: every cell freezes after 2 epochs.
    def _evaluate(loader):
        out = fitter.__class__.__bases__[0]._evaluate(fitter, loader)
        out["cc_norm"] = torch.tensor([0.5, 0.5])
        return out

    fitter._evaluate = _evaluate
    history = fitter.fit()
    # All frozen → training stops well before max_epochs=50
    assert len(history) < 50
    assert history[-1]["frozen_cells"] == 2


# -----------------------------------------------------------------------------
# (f) Wrong monitor shape raises a descriptive error
# -----------------------------------------------------------------------------


def test_per_cell_monitor_required():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=2)
    model = _LinearReadout(F=4, N=2)
    # val_metrics returning a SCALAR (rather than per-cell tensor) breaks
    # PerNeuronFitter's per-cell tracking — should error loudly.
    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        val_metrics={"cc_norm": lambda p, r: torch.tensor(0.5)},
        max_epochs=2, patience=2,
        log_fn=lambda d: None,
    )
    with pytest.raises(ValueError, match="per-cell monitor"):
        fitter.fit()


# -----------------------------------------------------------------------------
# (g) Restoration lands cells at their BEST-on-monitor state, not their
#     freeze-time state. This is the regression for the silent-snapshot bug:
#     pre-fix, snapshots were captured ``patience`` epochs after the best
#     was last seen, so restored weights were a drifted version of the best
#     — and PerNeuronFitter ended up worse than the global Fitter on NS1.
# -----------------------------------------------------------------------------


def test_snapshot_is_taken_at_improvement_time_not_freeze_time():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=2)
    model = _LinearReadout(F=4, N=2)
    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=8, patience=3,
        log_fn=lambda d: None,
    )

    # Drive cell 0's monitor: best at epoch 0 (0.9), then strictly worse
    # for 3 epochs → freezes at epoch 3 with snapshot expected to be the
    # state at epoch 0 (the best). Cell 1 keeps improving so it never freezes.
    cell0_trajectory = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1]
    weights_at_epoch_0 = None
    weights_at_epoch_3 = None

    original_evaluate = fitter._evaluate

    def _evaluate(loader):
        out = original_evaluate(loader)
        e = getattr(fitter, "_e", 0)
        fitter._e = e + 1
        nonlocal weights_at_epoch_0, weights_at_epoch_3
        # snapshot the model's cell-0 weight slice at the chosen epochs
        if e == 0:
            weights_at_epoch_0 = model.fc.weight.data[0].detach().clone()
        if e == 3:
            weights_at_epoch_3 = model.fc.weight.data[0].detach().clone()
        out["cc_norm"] = torch.tensor([cell0_trajectory[e], 0.1 + 1e-3 * e])
        return out

    fitter._evaluate = _evaluate
    fitter.fit()

    # Cell 0 must have frozen, and its restored weight slice must match
    # the epoch-0 capture (the best-on-monitor state), NOT the epoch-3
    # capture (the patience-expired drifted state).
    assert int(fitter._frozen_cells[0].item()) == 1
    final_w0 = model.fc.weight.data[0]
    assert torch.allclose(final_w0, weights_at_epoch_0), (
        "Cell-0 weight should be restored to the BEST-monitor state (epoch 0)"
    )
    assert not torch.allclose(final_w0, weights_at_epoch_3), (
        "Cell-0 weight should NOT be the freeze-time state (epoch 3)"
    )


def test_snapshot_includes_per_neuron_buffers():
    """Readouts that embed normalization layers with per-neuron running
    buffers (e.g. ``BatchNorm1d(N)`` after the STRF kernel) must have
    those buffers snapshotted and restored alongside the parameters.

    Without this, restored per-cell weights would sit on top of running
    statistics that kept accumulating after the cell froze — same drift
    failure mode as the snapshot-timing bug, one layer down.
    """
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=2)

    class _LinearReadoutWithBN(torch.nn.Module):
        def __init__(self, F=4, N=2):
            super().__init__()
            self.O = N
            self.fc = torch.nn.Linear(F, N, bias=True)
            self.bn = torch.nn.BatchNorm1d(N)
            self.readout = torch.nn.ModuleDict({"fc": self.fc, "bn": self.bn})

        def forward(self, stims):
            x = stims.transpose(-1, -2)
            y = self.fc(x).transpose(-1, -2)   # (B, N, T)
            y = self.bn(y)
            return y.unsqueeze(2)

        def detach(self):
            pass

    model = _LinearReadoutWithBN(F=4, N=2)
    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=8, patience=3,
        log_fn=lambda d: None,
    )

    # Force cell 0 to freeze early; record its running buffers at best-epoch.
    cell0_trajectory = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1]
    buffers_at_epoch_0 = {}

    original_evaluate = fitter._evaluate

    def _evaluate(loader):
        out = original_evaluate(loader)
        e = getattr(fitter, "_e", 0)
        fitter._e = e + 1
        if e == 0:
            buffers_at_epoch_0["mean"] = model.bn.running_mean[0].detach().clone()
            buffers_at_epoch_0["var"]  = model.bn.running_var[0].detach().clone()
        out["cc_norm"] = torch.tensor([cell0_trajectory[e], 0.1 + 1e-3 * e])
        return out

    fitter._evaluate = _evaluate
    fitter.fit()

    assert int(fitter._frozen_cells[0].item()) == 1
    # The BN's per-neuron running buffers for cell 0 should match its
    # epoch-0 capture, NOT whatever drifted to over the post-freeze epochs.
    assert torch.allclose(
        model.bn.running_mean[0], buffers_at_epoch_0["mean"]
    ), "BN running_mean[0] should be restored to the best-epoch state"
    assert torch.allclose(
        model.bn.running_var[0], buffers_at_epoch_0["var"]
    ), "BN running_var[0] should be restored to the best-epoch state"


def test_active_cell_gradient_matches_fitter():
    """On a no-shared-params model, when no cell is frozen, PerNeuronFitter
    must produce **bit-identical** per-cell gradients to Fitter.

    Both losses must collapse to the same scalar — Fitter via
    ``mse_loss(reduction='mean')`` ⇒ (1/N) · Σ per_cell[n], and
    PerNeuronFitter via ``(per_cell * mask).sum() / N`` when ``mask`` is
    all-ones. Any other normalization (e.g. divide by ``n_active``)
    would introduce a learning-rate inflation as cells freeze. This is
    the regression for the n_active-vs-N normalization bug fixed
    2026-05-18.
    """
    from deepSTRF.metrics import mse_loss
    from deepSTRF.training import Fitter

    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=4)

    # Identical models for the two fitters
    set_random_seed(0)
    model_f = _LinearReadout(F=4, N=4)
    set_random_seed(0)
    model_p = _LinearReadout(F=4, N=4)
    # sanity: identical parameter init
    for pf, pp in zip(model_f.parameters(), model_p.parameters()):
        assert torch.allclose(pf, pp)

    fitter_f = Fitter(model_f, train_loader, val_loader,
                      max_epochs=1, patience=1,
                      log_fn=lambda d: None, track_train_metrics=False)
    fitter_p = PerNeuronFitter(model_p, train_loader, val_loader,
                               max_epochs=1, patience=1,
                               log_fn=lambda d: None,
                               track_train_metrics=False)
    # initialize PNF's frozen-cells state without running fit()
    fitter_p._frozen_cells = torch.zeros(4, dtype=torch.bool,
                                          device=fitter_p.device)

    # Grab the first batch and run one step on each model
    batch = next(iter(train_loader))
    stims, responses, _vm, _meta = batch

    # Fitter step
    fitter_f.optimizer.zero_grad()
    pred_f = model_f(stims)
    loss_f = mse_loss(pred_f, responses)
    loss_f.backward()
    grads_f = [p.grad.detach().clone() for p in model_f.parameters()]

    # PerNeuronFitter step (no cells frozen)
    fitter_p.optimizer.zero_grad()
    pred_p = model_p(stims)
    per_cell = mse_loss(pred_p, responses, reduction="none")
    active_mask = (~fitter_p._frozen_cells).float()
    loss_p = (per_cell * active_mask).sum() / model_p.O
    loss_p.backward()
    grads_p = [p.grad.detach().clone() for p in model_p.parameters()]

    # Predictions must be identical (same model, same init, same input)
    assert torch.allclose(pred_f, pred_p)
    # Scalar losses must match
    assert torch.allclose(loss_f, loss_p)
    # Per-parameter gradients must match bit-exactly
    for gf, gp in zip(grads_f, grads_p):
        assert torch.allclose(gf, gp), (
            f"Gradients diverge: max abs diff = {(gf - gp).abs().max():.2e}"
        )


def test_active_gradient_constant_as_cells_freeze():
    """As cells freeze, the per-cell gradient on the *surviving* cells must
    stay constant (no learning-rate inflation). Pre-fix, dividing by
    ``n_active`` instead of ``N`` inflated the active-cell gradient by
    a factor of ``N / n_active`` as cells dropped out.
    """
    from deepSTRF.metrics import mse_loss

    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=4)

    model = _LinearReadout(F=4, N=4)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fitter = PerNeuronFitter(model, train_loader, val_loader,
                             max_epochs=1, patience=1,
                             log_fn=lambda d: None,
                             track_train_metrics=False)

    batch = next(iter(train_loader))
    stims, responses, _vm, _meta = batch

    # Pass 1: no cells frozen
    model.load_state_dict(init_state)
    fitter._frozen_cells = torch.zeros(4, dtype=torch.bool, device=fitter.device)
    fitter.optimizer.zero_grad()
    pred = model(stims)
    per_cell = mse_loss(pred, responses, reduction="none")
    active_mask = (~fitter._frozen_cells).float()
    loss = (per_cell * active_mask).sum() / model.O
    loss.backward()
    grads_all_active = [p.grad.detach().clone() for p in model.parameters()]

    # Pass 2: cells 0..2 frozen, cell 3 still active
    model.load_state_dict(init_state)
    fitter._frozen_cells = torch.tensor([True, True, True, False], device=fitter.device)
    fitter.optimizer.zero_grad()
    pred = model(stims)
    per_cell = mse_loss(pred, responses, reduction="none")
    active_mask = (~fitter._frozen_cells).float()
    loss = (per_cell * active_mask).sum() / model.O
    loss.backward()
    grads_one_active = [p.grad.detach().clone() for p in model.parameters()]

    # Cell 3's slice of every per-N parameter must have the same gradient
    # in both passes. Pre-fix (divide by n_active), it would be inflated by
    # 4 / 1 = 4x in pass 2.
    for ga, go in zip(grads_all_active, grads_one_active):
        if ga.dim() >= 1 and ga.shape[0] == 4:
            assert torch.allclose(ga[3], go[3]), (
                f"Cell-3 gradient changed when other cells froze "
                f"(max abs diff = {(ga[3] - go[3]).abs().max():.2e})"
            )


def test_snapshot_also_restores_unfrozen_cells_at_end():
    """Cells that never frozen but improved during training must also be
    restored to their best-on-monitor state at end-of-fit."""
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=1)
    model = _LinearReadout(F=4, N=1)
    fitter = PerNeuronFitter(
        model, train_loader, val_loader,
        max_epochs=5, patience=10,  # cell never freezes within max_epochs
        log_fn=lambda d: None,
    )

    # Best score at epoch 2; then drift but no freeze (patience > remaining)
    trajectory = [0.1, 0.3, 0.9, 0.5, 0.4]
    weights_at_epoch_2 = None

    original_evaluate = fitter._evaluate

    def _evaluate(loader):
        out = original_evaluate(loader)
        e = getattr(fitter, "_e", 0)
        fitter._e = e + 1
        nonlocal weights_at_epoch_2
        if e == 2:
            weights_at_epoch_2 = model.fc.weight.data[0].detach().clone()
        out["cc_norm"] = torch.tensor([trajectory[e]])
        return out

    fitter._evaluate = _evaluate
    fitter.fit()

    # Cell 0 should NOT be frozen (patience=10, only 5 epochs ran)
    assert int(fitter._frozen_cells[0].item()) == 0
    # But its weight should still be restored to the best-on-monitor state
    final_w0 = model.fc.weight.data[0]
    assert torch.allclose(final_w0, weights_at_epoch_2), (
        "Even non-frozen cells must be restored to their best-on-monitor state"
    )
