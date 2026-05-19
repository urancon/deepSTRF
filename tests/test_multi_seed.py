"""Tests for ``deepSTRF.training.fit_multi_seed``.

Reuses the synthetic ``_ToyDataset`` / ``_LinearReadout`` /
``_make_loaders`` fixtures from ``tests/test_fitter.py`` (same package,
direct import) so the multi-seed sweep is tested on a tiny in-memory
problem with no real data on disk.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from deepSTRF.training import fit_multi_seed
from deepSTRF.training.wandb_log import make_wandb_logger_factory
from deepSTRF.utils.data import neural_collate

from tests.test_fitter import (
    _LinearReadoutWithReadoutAttr,
    _make_loaders,
    _ToyDataset,
)


# -----------------------------------------------------------------------------
# Factories for the toy problem
# -----------------------------------------------------------------------------


def _model_factory(seed: int):
    return _LinearReadoutWithReadoutAttr(F=4, N=2)


def _build_test_loader(train_W: torch.Tensor, seed: int) -> DataLoader:
    """Build a held-out test loader whose responses are generated under the
    *same* target W as the train set — so train/val/test all share the
    learnable problem and the metrics are comparable across splits.
    """
    test_ds = _ToyDataset(n_stims=8, seed=seed + 500)
    g = torch.Generator().manual_seed(seed + 600)
    test_ds.responses_per_stim = []
    for s in test_ds.stims:
        base = train_W @ s
        per_neuron = []
        for n in range(test_ds.N):
            noise = torch.randn(3, s.shape[-1], generator=g) * 0.05
            per_neuron.append(base[n].unsqueeze(0).expand(3, s.shape[-1]) + noise)
        test_ds.responses_per_stim.append(per_neuron)
    return DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=neural_collate
    )


def _loader_factory(seed: int):
    """Return ``(train, val, test)`` for the toy problem under ``seed``.

    Train and val are built by ``_make_loaders(seed=seed)``; test is a
    fresh held-out dataset whose targets share the same ``W`` as train.
    """
    train, val = _make_loaders(seed=seed)
    train_W = train.dataset._target_W
    test = _build_test_loader(train_W, seed)
    return train, val, test


# -----------------------------------------------------------------------------
# Same seed -> identical, different seeds -> differ
# -----------------------------------------------------------------------------


def test_same_seed_yields_identical_results():
    def _run():
        return fit_multi_seed(
            model_factory=_model_factory,
            loader_factory=_loader_factory,
            seeds=[42],
            fitter_kwargs={"max_epochs": 3, "patience": 3},
        )

    a = _run()
    b = _run()
    assert torch.equal(a["per_seed_test_cc_norm"], b["per_seed_test_cc_norm"])
    assert torch.equal(a["per_seed_val_cc"], b["per_seed_val_cc"])
    # Best state dict round-trips bit-identically too.
    for k, v in a["best_state_dict"].items():
        assert torch.equal(v, b["best_state_dict"][k])


def test_different_seeds_yield_different_results():
    results = fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0, 1],
        fitter_kwargs={"max_epochs": 3, "patience": 3},
    )
    a = results["per_seed_val_cc"][0]
    b = results["per_seed_val_cc"][1]
    assert not torch.allclose(a, b), \
        "different seeds should produce different per-cell cc trajectories"


# -----------------------------------------------------------------------------
# Aggregation shape contract
# -----------------------------------------------------------------------------


def test_aggregation_shape_contract():
    results = fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        n_seeds=3,
        fitter_kwargs={"max_epochs": 2, "patience": 2},
    )
    N = 2  # _LinearReadoutWithReadoutAttr default
    assert results["seeds"] == [0, 1, 2]
    assert results["per_seed_test_cc_norm"].shape == (3, N)
    assert results["per_seed_test_cc"].shape == (3, N)
    assert results["mean_test_cc_norm"].shape == (N,)
    assert results["std_test_cc_norm"].shape == (N,)
    assert results["per_seed_val_cc_norm"].shape == (3, N)
    assert results["mean_val_cc_norm"].shape == (N,)
    assert len(results["per_seed_histories"]) == 3
    assert all(len(h) == 2 for h in results["per_seed_histories"])

    # Loss is a scalar per seed -> per_seed_*_loss is (n_seeds, 1).
    assert results["per_seed_val_loss"].shape == (3, 1)
    assert results["per_seed_test_loss"].shape == (3, 1)


# -----------------------------------------------------------------------------
# best_seed / best_state_dict
# -----------------------------------------------------------------------------


def test_best_state_dict_loads_into_fresh_model():
    """``best_state_dict`` is a deep copy of the post-fit state of the best
    seed's model; loading it into a freshly-built model should succeed."""
    results = fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        n_seeds=3,
        fitter_kwargs={"max_epochs": 2, "patience": 2},
    )
    assert results["best_seed"] in [0, 1, 2]
    fresh = _model_factory(seed=999)
    missing, unexpected = fresh.load_state_dict(
        results["best_state_dict"], strict=True
    )
    assert missing == [] and unexpected == []


def test_best_seed_matches_argmax_of_val_monitor():
    """The best seed must be argmax (mode='max') over the nanmean'd
    post-fit val_cc_norm of each seed."""
    results = fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        n_seeds=3,
        fitter_kwargs={"max_epochs": 3, "patience": 3},
    )
    per_seed = results["per_seed_val_cc_norm"]   # (3, N)
    scalars = torch.nanmean(per_seed, dim=1)     # (3,)
    expected = int(torch.argmax(scalars).item())
    assert results["best_seed"] == expected


def test_best_seed_uses_argmin_when_mode_min():
    """``monitor='val_loss'`` with ``mode='min'`` should pick the lowest
    val_loss seed as best, not the highest."""
    results = fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        n_seeds=3,
        fitter_kwargs={
            "max_epochs": 2, "patience": 2,
            "monitor": "val_loss", "mode": "min",
        },
    )
    per_seed_loss = results["per_seed_val_loss"].squeeze(-1)   # (3,)
    expected = int(torch.argmin(per_seed_loss).item())
    assert results["best_seed"] == expected


# -----------------------------------------------------------------------------
# Argument validation
# -----------------------------------------------------------------------------


def test_fitter_kwargs_blocks_managed_keys():
    import pytest

    for k in ("model", "train_loader", "val_loader"):
        with pytest.raises(ValueError, match="managed by fit_multi_seed"):
            fit_multi_seed(
                model_factory=_model_factory,
                loader_factory=_loader_factory,
                seeds=[0],
                fitter_kwargs={k: object()},
            )


def test_ckpt_path_is_suffixed_per_seed(tmp_path):
    base = tmp_path / "best.pt"
    fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0, 1],
        fitter_kwargs={"max_epochs": 2, "patience": 2, "ckpt_path": base},
    )
    assert not base.exists(), "base ckpt path must not be written (would clobber)"
    assert (tmp_path / "best_seed0.pt").exists()
    assert (tmp_path / "best_seed1.pt").exists()


# -----------------------------------------------------------------------------
# output_dir auto-save (logger-agnostic)
# -----------------------------------------------------------------------------


def test_output_dir_creates_per_seed_tree(tmp_path):
    """Every seed gets its own subdir with history.json, final.json,
    final_neurons.pt, best.pt; the root has summary.json + summary_neurons.pt."""
    out = tmp_path / "runs"
    fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0, 1],
        fitter_kwargs={"max_epochs": 3, "patience": 3},
        output_dir=out,
    )
    assert (out / "summary.json").exists()
    assert (out / "summary_neurons.pt").exists()
    for s in (0, 1):
        sd = out / f"seed{s}"
        assert (sd / "history.json").exists()
        assert (sd / "final.json").exists()
        assert (sd / "final_neurons.pt").exists()
        assert (sd / "best.pt").exists()


def test_output_dir_history_json_is_loadable_and_well_formed(tmp_path):
    out = tmp_path / "runs"
    fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0],
        fitter_kwargs={"max_epochs": 3, "patience": 3},
        output_dir=out,
    )
    with open(out / "seed0" / "history.json") as f:
        history = json.load(f)
    assert len(history) == 3
    assert history[0]["epoch"] == 0
    # Per-neuron metrics summarised to {mean, p10, p50, p90, n_valid}
    cc = history[0]["val_cc_norm"]
    assert isinstance(cc, dict)
    assert set(cc) >= {"mean", "p10", "p50", "p90", "n_valid"}
    # Scalars stay scalars
    assert isinstance(history[0]["val_loss"], float)


def test_output_dir_final_neurons_round_trip_via_torch_load(tmp_path):
    out = tmp_path / "runs"
    fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0],
        fitter_kwargs={"max_epochs": 2, "patience": 2},
        output_dir=out,
    )
    N = 2
    saved = torch.load(out / "seed0" / "final_neurons.pt", weights_only=False)
    assert set(saved) == {"val", "test"}
    assert saved["val"]["cc_norm"].shape == (N,)
    assert saved["test"]["cc_norm"].shape == (N,)
    # 'loss' is scalar -> excluded from final_neurons.pt
    assert "loss" not in saved["val"]


def test_output_dir_best_pt_loads_into_fresh_model(tmp_path):
    out = tmp_path / "runs"
    results = fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0, 1, 2],
        fitter_kwargs={"max_epochs": 2, "patience": 2},
        output_dir=out,
    )
    best = results["best_seed"]
    sd = torch.load(out / f"seed{best}" / "best.pt", weights_only=False)
    fresh = _model_factory(seed=999)
    fresh.load_state_dict(sd, strict=True)


def test_output_dir_summary_json_includes_best_seed_and_monitor(tmp_path):
    out = tmp_path / "runs"
    fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0, 1, 2],
        fitter_kwargs={"max_epochs": 2, "patience": 2,
                        "monitor": "val_loss", "mode": "min"},
        output_dir=out,
    )
    with open(out / "summary.json") as f:
        summary = json.load(f)
    assert summary["seeds"] == [0, 1, 2]
    assert summary["best_seed"] in [0, 1, 2]
    assert summary["monitor"] == "val_loss"
    assert summary["mode"] == "min"
    # mean_/std_ entries for per-neuron metrics are dict summaries
    assert isinstance(summary["mean_test_cc_norm"], dict)
    assert "p50" in summary["mean_test_cc_norm"]


# -----------------------------------------------------------------------------
# WandB integration
# -----------------------------------------------------------------------------


def test_wandb_disabled_creates_no_directory(tmp_path, monkeypatch):
    """``mode='disabled'`` must be a true no-op — no ``./wandb/`` directory
    under the (chdir'd) cwd, no run files."""
    monkeypatch.chdir(tmp_path)
    results = fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0],
        fitter_kwargs={"max_epochs": 1, "patience": 1},
        logger_factory=make_wandb_logger_factory(
            mode="disabled", project="deepstrf-test",
        ),
    )
    assert "per_seed_val_cc_norm" in results
    assert not (tmp_path / "wandb").exists()


def test_wandb_offline_writes_run_files(tmp_path):
    """``mode='offline'`` writes one ``offline-run-*`` directory per seed
    under ``dir=``."""
    fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_loader_factory,
        seeds=[0, 1],
        fitter_kwargs={"max_epochs": 1, "patience": 1},
        logger_factory=make_wandb_logger_factory(
            mode="offline", project="deepstrf-test",
            group="smoke", dir=str(tmp_path),
        ),
    )
    wandb_dir = tmp_path / "wandb"
    assert wandb_dir.exists()
    offline_runs = sorted(wandb_dir.glob("offline-run-*"))
    assert len(offline_runs) == 2, \
        f"expected 2 offline-run-* dirs, got {len(offline_runs)}: {offline_runs}"
