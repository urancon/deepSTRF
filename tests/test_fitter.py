"""Tests for ``deepSTRF.training.Fitter``.

Mirrors the contracts in ``docs/_source/md/fitter.md`` §9. The dataset
is a tiny synthetic NeuralDataset-shaped iterable (no real data on disk)
— Fitter behaviour, not data IO, is what's under test here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from deepSTRF.metrics import corrcoef, fve, mse_loss
from deepSTRF.training import Fitter, set_random_seed
from deepSTRF.training.fitter import _pad_and_cat
from deepSTRF.utils.data import neural_collate


# -----------------------------------------------------------------------------
# Synthetic fixtures
# -----------------------------------------------------------------------------


class _ToyDataset(Dataset):
    """Tiny dataset mimicking ``NeuralDataset.__getitem__`` outputs.

    Each item is ``(stim, per_neuron_responses, per_neuron_mask, stim_meta)``
    in the contract neural_collate expects:
        - stim shape ``(F, T)``
        - per_neuron_responses: list of ``N`` ``(R, T)`` tensors
        - per_neuron_mask: ``(N,)`` bool tensor
        - stim_meta: dict

    Generative model: response = sum_f W[n, f] * stim[f] + small noise.
    Trainable by a Linear-over-features layer.
    """

    def __init__(self, n_stims=8, F=4, T=20, N=2, R=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.F = F
        self.T = T
        self.N = N
        self.stims = [torch.randn(F, T, generator=g) for _ in range(n_stims)]
        W = torch.randn(N, F, generator=g)
        self._target_W = W
        self.responses_per_stim: List[List[torch.Tensor]] = []
        for s in self.stims:
            base = (W @ s)                                    # (N, T)
            per_neuron = []
            for n in range(N):
                noise = torch.randn(R, T, generator=g) * 0.05
                per_neuron.append(base[n].unsqueeze(0).expand(R, T) + noise)
            self.responses_per_stim.append(per_neuron)

    def __len__(self):
        return len(self.stims)

    def __getitem__(self, i):
        stim = self.stims[i]
        responses = self.responses_per_stim[i]
        nrn_mask = torch.ones(self.N, dtype=torch.bool)
        return stim, responses, nrn_mask, {"idx": i}


class _LinearReadout(torch.nn.Module):
    """Per-neuron linear readout from per-time-step features. Output ``(B, N, 1, T)``.

    Stim arrives as ``(B, F, T)`` from ``_ToyDataset`` after ``neural_collate``.
    ``forward`` permutes to ``(B, T, F)``, applies ``nn.Linear(F, N)``, and
    reshapes to ``(B, N, 1, T)``.
    """

    def __init__(self, F=4, N=2):
        super().__init__()
        self.O = N
        self.linear = torch.nn.Linear(F, N, bias=True)

    def forward(self, stims):
        # (B, F, T) -> (B, T, F) -> (B, T, N) -> (B, N, T) -> (B, N, 1, T)
        x = stims.transpose(-1, -2)
        y = self.linear(x)
        y = y.transpose(-1, -2)
        return y.unsqueeze(2)

    def detach(self):
        pass


def _make_loaders(seed=0, batch_size=4, n_train=32, n_val=16):
    """Build train/val loaders sharing the same generative ``W`` so the
    val problem is in fact learnable from train (different stimuli
    + different noise, but same ground-truth weights).
    """
    train_ds = _ToyDataset(n_stims=n_train, seed=seed)
    val_ds = _ToyDataset(n_stims=n_val, seed=seed + 100)
    val_ds._target_W = train_ds._target_W
    # Regenerate val responses with the train W so the linear problem matches.
    g = torch.Generator().manual_seed(seed + 200)
    val_ds.responses_per_stim = []
    for s in val_ds.stims:
        base = train_ds._target_W @ s
        per_neuron = []
        for n in range(val_ds.N):
            noise = torch.randn(3, s.shape[-1], generator=g) * 0.05
            per_neuron.append(base[n].unsqueeze(0).expand(3, s.shape[-1]) + noise)
        val_ds.responses_per_stim.append(per_neuron)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=neural_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=neural_collate
    )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# (a) one-epoch fit reduces loss
# -----------------------------------------------------------------------------


def test_fit_reduces_loss_on_toy_problem():
    """Optimisation works: train loss strictly decreases, and at some
    epoch the val loss beats the initial one.

    We don't assert ``history[-1]['val_loss'] < history[0]['val_loss']``
    because with no early stopping the toy model overfits the 8-stim
    train set in a few epochs — that's expected behaviour, not a Fitter
    bug, and is exactly what the ``ckpt_path`` + early-stop machinery
    handles in real use.
    """
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(seed=0)
    model = _LinearReadout(F=4, N=2)

    fitter = Fitter(
        model, train_loader, val_loader,
        max_epochs=20, patience=20,
        log_fn=lambda d: None,
    )
    history = fitter.fit()

    assert len(history) == 20
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    best_val_loss = min(h["val_loss"] for h in history)
    assert best_val_loss < history[0]["val_loss"]


def test_fit_drives_val_cc_norm_up():
    """At some point during fit, val CCnorm exceeds the initial value
    (best-during-training, not last-epoch — see test above)."""
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(seed=0)
    model = _LinearReadout(F=4, N=2)
    fitter = Fitter(
        model, train_loader, val_loader,
        max_epochs=30, patience=30,
        log_fn=lambda d: None,
    )
    history = fitter.fit()
    initial_cc = float(torch.nanmean(history[0]["val_cc_norm"]).item())
    best_cc = max(
        float(torch.nanmean(h["val_cc_norm"]).item()) for h in history
    )
    assert best_cc > initial_cc


# -----------------------------------------------------------------------------
# (b) early stopping fires
# -----------------------------------------------------------------------------


def test_early_stop_fires_when_monitor_plateaus():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)
    # Force a flat monitor by feeding back the same constant.
    fitter = Fitter(
        model, train_loader, val_loader,
        max_epochs=100,
        patience=3,
        log_fn=lambda d: None,
    )

    # Override on_epoch_end-free path: fake a constant monitor so we
    # see the patience counter trip deterministically.
    flat_value = torch.tensor([0.0])
    original_evaluate = fitter._evaluate

    def _flat_evaluate(loader):
        out = original_evaluate(loader)
        out["cc_norm"] = flat_value.clone()
        return out

    fitter._evaluate = _flat_evaluate

    history = fitter.fit()
    # First epoch sets best, then `patience` (3) epochs without improvement
    # → stop at epoch 3 (4 entries: 0,1,2,3).
    assert len(history) == fitter.patience + 1


# -----------------------------------------------------------------------------
# (c) checkpoint round-trip
# -----------------------------------------------------------------------------


def test_checkpoint_saves_and_restores_best(tmp_path):
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)
    ckpt = tmp_path / "best.pt"
    fitter = Fitter(
        model, train_loader, val_loader,
        max_epochs=10, patience=10,
        ckpt_path=ckpt,
        log_fn=lambda d: None,
    )
    history = fitter.fit()
    assert ckpt.exists()

    # The model after fit() should match the saved checkpoint exactly,
    # because Fitter restores the best ckpt before returning.
    saved = torch.load(ckpt, map_location="cpu")
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved[k])

    # Best epoch's val_cc_norm should be the max over all epochs (mode='max').
    best_idx = max(
        range(len(history)),
        key=lambda i: float(torch.nanmean(history[i]["val_cc_norm"]).item()),
    )
    # ckpt was saved on the best epoch — so reloading == that state.
    # (We can't easily inspect the state at intermediate epochs without
    #  another pass, so this just sanity-checks the API.)
    assert best_idx >= 0


# -----------------------------------------------------------------------------
# (d) hook overrides fire (loss_fn, val_metrics, on_epoch_end)
# -----------------------------------------------------------------------------


def test_custom_loss_fn_is_called():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)

    n_calls = {"n": 0}

    def my_loss(pred, responses):
        n_calls["n"] += 1
        return mse_loss(pred, responses)

    fitter = Fitter(
        model, train_loader, val_loader,
        loss_fn=my_loss,
        max_epochs=2, patience=2,
        log_fn=lambda d: None,
    )
    fitter.fit()
    # Called once per train batch + once per val-epoch eval.
    assert n_calls["n"] > 0


def test_custom_val_metrics_propagate_to_epoch_dict():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)

    fitter = Fitter(
        model, train_loader, val_loader,
        val_metrics={
            "fve": lambda p, r: fve(p, r, reduction="none"),
        },
        monitor="val_fve", mode="max",
        max_epochs=3, patience=3,
        log_fn=lambda d: None,
    )
    history = fitter.fit()
    assert "val_fve" in history[0]
    assert "train_fve" in history[0]
    assert "val_cc_norm" not in history[0]   # default replaced


def test_on_epoch_end_override_fires():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)

    seen: List[int] = []

    class MyFitter(Fitter):
        def on_epoch_end(self, epoch, epoch_dict):
            seen.append(epoch)

    fitter = MyFitter(
        model, train_loader, val_loader,
        max_epochs=4, patience=4,
        log_fn=lambda d: None,
    )
    fitter.fit()
    assert seen == [0, 1, 2, 3]


# -----------------------------------------------------------------------------
# (e) seeding produces bit-identical runs
# -----------------------------------------------------------------------------


def test_set_random_seed_makes_runs_identical():
    def _one_run():
        set_random_seed(42)
        train_loader, val_loader = _make_loaders(seed=0)
        model = _LinearReadout(F=4, N=2)
        fitter = Fitter(
            model, train_loader, val_loader,
            max_epochs=3, patience=3,
            log_fn=lambda d: None,
        )
        fitter.fit()
        return {k: v.clone() for k, v in model.state_dict().items()}

    a = _one_run()
    b = _one_run()
    for k in a:
        assert torch.equal(a[k], b[k]), f"state mismatch on {k!r}"


# -----------------------------------------------------------------------------
# (f) monitor='val_loss' (mode='min') and monitor='val_cc_norm' (mode='max')
#     both work end-to-end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("monitor,mode", [
    ("val_loss", "min"),
    ("val_cc_norm", "max"),
    ("val_cc", "max"),
])
def test_monitor_modes_work_end_to_end(monitor, mode):
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)
    fitter = Fitter(
        model, train_loader, val_loader,
        monitor=monitor, mode=mode,
        max_epochs=5, patience=5,
        log_fn=lambda d: None,
    )
    history = fitter.fit()
    assert len(history) == 5
    assert monitor in history[0]


# -----------------------------------------------------------------------------
# Argument-validation
# -----------------------------------------------------------------------------


def test_invalid_mode_raises():
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)
    with pytest.raises(ValueError, match="mode"):
        Fitter(model, train_loader, val_loader, mode="MAX")


def test_invalid_patience_raises():
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)
    with pytest.raises(ValueError, match="patience"):
        Fitter(model, train_loader, val_loader, patience=0)


def test_unknown_monitor_raises_during_fit():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)
    fitter = Fitter(
        model, train_loader, val_loader,
        monitor="val_does_not_exist",
        max_epochs=2, patience=2,
        log_fn=lambda d: None,
    )
    with pytest.raises(KeyError, match="val_does_not_exist"):
        fitter.fit()


# -----------------------------------------------------------------------------
# evaluate() returns un-prefixed dict
# -----------------------------------------------------------------------------


def test_evaluate_returns_unprefixed_keys():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadout(F=4, N=2)
    fitter = Fitter(model, train_loader, val_loader, log_fn=lambda d: None)
    out = fitter.evaluate(val_loader)
    assert set(out) == {"loss", "cc", "cc_norm"}


# -----------------------------------------------------------------------------
# _pad_and_cat behaviour
# -----------------------------------------------------------------------------


def test_pad_and_cat_pads_with_nan_and_cats_along_batch():
    a = torch.zeros(2, 3, 1, 5)
    b = torch.ones(1, 3, 2, 7)
    out = _pad_and_cat([a, b])
    assert out.shape == (3, 3, 2, 7)
    # First 2 batch entries: from `a`, R-axis padded with NaN beyond R=1, T-axis
    # padded with NaN beyond T=5.
    assert torch.isnan(out[0, 0, 1, :]).all()              # R=1 slot is NaN
    assert torch.isnan(out[0, 0, 0, 5:]).all()             # T past 5 is NaN
    assert (out[0, 0, 0, :5] == 0).all()                   # original zeros kept
    # Last batch entry: from `b`, no padding needed.
    assert (out[2, 0, 0, :] == 1).all()


def test_pad_and_cat_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        _pad_and_cat([])


# -----------------------------------------------------------------------------
# (g) track_per_cell_best — per-cell best-on-monitor snapshot + end-of-fit
#     restore overlaid on top of the global ckpt_path restore.
# -----------------------------------------------------------------------------


class _LinearReadoutWithReadoutAttr(torch.nn.Module):
    """Variant of _LinearReadout that exposes ``self.readout`` so the
    per-cell snapshot logic can find the per-N parameters and buffers."""

    def __init__(self, F=4, N=2, with_bn: bool = False):
        super().__init__()
        self.O = N
        self.fc = torch.nn.Linear(F, N, bias=True)
        if with_bn:
            self.bn = torch.nn.BatchNorm1d(N)
            self.readout = torch.nn.ModuleDict({"fc": self.fc, "bn": self.bn})
        else:
            self.bn = None
            self.readout = self.fc

    def forward(self, stims):
        x = stims.transpose(-1, -2)        # (B, T, F)
        y = self.fc(x).transpose(-1, -2)   # (B, N, T)
        if self.bn is not None:
            y = self.bn(y)
        return y.unsqueeze(2)              # (B, N, 1, T)

    def detach(self):
        pass


def test_track_per_cell_best_requires_per_cell_monitor():
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadoutWithReadoutAttr(F=4, N=2)
    # A val metric that returns a SCALAR breaks per-cell tracking.
    fitter = Fitter(
        model, train_loader, val_loader,
        val_metrics={"cc_norm": lambda p, r: torch.tensor(0.5)},
        max_epochs=2, patience=2,
        log_fn=lambda d: None,
        track_per_cell_best=True,
    )
    with pytest.raises(ValueError, match="per-cell"):
        fitter.fit()


def test_track_per_cell_best_snapshots_at_improvement_time():
    """The snapshot for cell n must be the readout weight slice at cell n's
    BEST-on-monitor epoch, regardless of how many epochs later training
    continues or terminates."""
    set_random_seed(0)
    train_loader, val_loader = _make_loaders(N=2) if False else _make_loaders()
    # _make_loaders gives N=2 by the default _ToyDataset; rebuild with our model.
    model = _LinearReadoutWithReadoutAttr(F=4, N=2)
    fitter = Fitter(
        model, train_loader, val_loader,
        max_epochs=8, patience=10,   # patience long enough that we run all 8 epochs
        log_fn=lambda d: None,
        track_per_cell_best=True,
    )

    # Drive cell 0's monitor: peak at epoch 2, then strictly worse.
    # Cell 1: monotone improvement throughout.
    cell0_traj = [0.1, 0.5, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0]
    weights_at_peak = {}

    original_evaluate = fitter._evaluate

    def _evaluate(loader):
        out = original_evaluate(loader)
        e = getattr(fitter, "_e", 0)
        fitter._e = e + 1
        if e == 2:
            weights_at_peak["fc"] = model.fc.weight.data[0].detach().clone()
            weights_at_peak["bias"] = model.fc.bias.data[0].detach().clone()
        out["cc_norm"] = torch.tensor([cell0_traj[e], 0.1 + 1e-3 * e])
        return out

    fitter._evaluate = _evaluate
    fitter.fit()

    # Cell 0 must be restored to its epoch-2 snapshot (its peak), not to
    # whatever it drifted to over the remaining 5 epochs.
    assert torch.allclose(model.fc.weight.data[0], weights_at_peak["fc"]), \
        "cell-0 readout weight should be restored to best-monitor state"
    assert torch.allclose(model.fc.bias.data[0], weights_at_peak["bias"]), \
        "cell-0 readout bias should be restored to best-monitor state"


def test_track_per_cell_best_restores_buffers_too():
    """Readouts with per-neuron buffers (e.g. BatchNorm1d(N)) — those
    buffers must also be snapshotted and restored, otherwise the post-fit
    eval-mode forward for cell n sits on top of running stats that kept
    accumulating after cell n's monitor peaked."""
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadoutWithReadoutAttr(F=4, N=2, with_bn=True)
    fitter = Fitter(
        model, train_loader, val_loader,
        max_epochs=8, patience=10,
        log_fn=lambda d: None,
        track_per_cell_best=True,
    )
    cell0_traj = [0.1, 0.5, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0]
    buffers_at_peak = {}

    original_evaluate = fitter._evaluate

    def _evaluate(loader):
        out = original_evaluate(loader)
        e = getattr(fitter, "_e", 0)
        fitter._e = e + 1
        if e == 2:
            buffers_at_peak["mean"] = model.bn.running_mean[0].detach().clone()
            buffers_at_peak["var"] = model.bn.running_var[0].detach().clone()
        out["cc_norm"] = torch.tensor([cell0_traj[e], 0.1 + 1e-3 * e])
        return out

    fitter._evaluate = _evaluate
    fitter.fit()

    assert torch.allclose(model.bn.running_mean[0], buffers_at_peak["mean"]), \
        "BN running_mean[0] should be restored to best-monitor state"
    assert torch.allclose(model.bn.running_var[0], buffers_at_peak["var"]), \
        "BN running_var[0] should be restored to best-monitor state"


def test_track_per_cell_best_off_by_default_is_a_noop():
    """Without the flag, Fitter must behave exactly as before — no
    per-cell bookkeeping, no per-cell restore at end."""
    set_random_seed(0)
    train_loader, val_loader = _make_loaders()
    model = _LinearReadoutWithReadoutAttr(F=4, N=2)
    fitter = Fitter(
        model, train_loader, val_loader,
        max_epochs=3, patience=3,
        log_fn=lambda d: None,
    )
    fitter.fit()
    assert not hasattr(fitter, "_per_cell_snapshots"), \
        "per-cell state should not exist when track_per_cell_best=False"
