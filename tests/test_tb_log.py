"""Tests for ``deepSTRF.training.tb_log``.

We test by inspecting the on-disk event files via TensorBoard's own
``EventAccumulator`` — no need to actually launch the TensorBoard UI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from deepSTRF.training import fit_multi_seed
from deepSTRF.training.tb_log import (
    TensorBoardSeedLogger,
    make_tensorboard_logger_factory,
)
from deepSTRF.utils.data import neural_collate

from tests.test_fitter import _LinearReadoutWithReadoutAttr, _ToyDataset, _make_loaders


def _factory(seed: int):
    train, val = _make_loaders(seed=seed)
    return train, val, val


def _model_factory(seed: int):
    return _LinearReadoutWithReadoutAttr(F=4, N=2)


# -----------------------------------------------------------------------------
# Construction + directory layout
# -----------------------------------------------------------------------------


def test_tb_logger_creates_event_file_under_group_run_subdir(tmp_path):
    logger = TensorBoardSeedLogger(
        seed=0, log_dir=str(tmp_path), group="experiment-A",
    )
    logger({"epoch": 0, "val_loss": 0.5})
    logger.close()

    run_dir = tmp_path / "experiment-A" / "experiment-A-seed0"
    assert run_dir.exists()
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    assert event_files, f"no event file in {run_dir}"


def test_tb_logger_run_name_falls_back_to_seed_only_when_no_group(tmp_path):
    logger = TensorBoardSeedLogger(seed=3, log_dir=str(tmp_path))
    logger({"epoch": 0, "val_loss": 0.5})
    logger.close()
    assert (tmp_path / "default" / "seed3").exists()


# -----------------------------------------------------------------------------
# Per-epoch scalars + per-neuron percentiles
# -----------------------------------------------------------------------------


def _read_scalars(run_dir: Path) -> dict:
    """Return ``{tag: [(step, value), ...]}`` for all scalar tags in run_dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)]
            for tag in acc.Tags()["scalars"]}


def test_tb_logger_writes_per_neuron_percentile_scalars(tmp_path):
    logger = TensorBoardSeedLogger(seed=0, log_dir=str(tmp_path), group="g")
    logger({
        "epoch": 0, "val_loss": 0.5,
        "val_cc_norm": torch.tensor([0.3, 0.5, 0.7, 0.9]),
    })
    logger({
        "epoch": 1, "val_loss": 0.4,
        "val_cc_norm": torch.tensor([0.4, 0.6, 0.8, 1.0]),
    })
    logger.close()

    scalars = _read_scalars(tmp_path / "g" / "g-seed0")
    for tag in ("val_loss", "val_cc_norm",
                 "val_cc_norm/p10", "val_cc_norm/p50", "val_cc_norm/p90"):
        assert tag in scalars, f"missing scalar {tag!r} (have {sorted(scalars)})"
    # Two points (steps 0 and 1) on each tag.
    assert len(scalars["val_loss"]) == 2
    assert len(scalars["val_cc_norm/p50"]) == 2


def test_tb_logger_finalize_writes_final_scalars(tmp_path):
    logger = TensorBoardSeedLogger(seed=0, log_dir=str(tmp_path), group="g")
    logger({"epoch": 0, "val_loss": 0.5,
             "val_cc_norm": torch.tensor([0.4, 0.6])})
    logger.finalize({
        "val":  {"loss": 0.42, "cc_norm": torch.tensor([0.55, 0.65])},
        "test": {"loss": 0.50, "cc_norm": torch.tensor([0.50, 0.60])},
    })
    logger.close()

    scalars = _read_scalars(tmp_path / "g" / "g-seed0")
    for tag in ("final/val_loss", "final/val_cc_norm",
                 "final/val_cc_norm/p50",
                 "final/test_loss", "final/test_cc_norm",
                 "final/test_cc_norm/p10", "final/test_cc_norm/p50",
                 "final/test_cc_norm/p90"):
        assert tag in scalars, f"missing final scalar {tag!r}"


def test_tb_logger_config_lands_as_hparams_scalars(tmp_path):
    TensorBoardSeedLogger(
        seed=7, log_dir=str(tmp_path), group="g",
        config={"n_frequency_bands": 34, "temporal_window_size": 9,
                 "model": "Linear"},
    ).close()
    scalars = _read_scalars(tmp_path / "g" / "g-seed7")
    assert "hparams/n_frequency_bands" in scalars
    assert "hparams/temporal_window_size" in scalars
    assert "hparams/seed" in scalars
    # Numeric values land as their float
    assert scalars["hparams/temporal_window_size"][0][1] == pytest.approx(9.0)


# -----------------------------------------------------------------------------
# fit_multi_seed end-to-end with the TB factory
# -----------------------------------------------------------------------------


def test_fit_multi_seed_with_tb_logger_writes_one_event_per_seed(tmp_path):
    fit_multi_seed(
        model_factory=_model_factory,
        loader_factory=_factory,
        seeds=[0, 1],
        fitter_kwargs={"max_epochs": 2, "patience": 2,
                        "track_train_metrics": True},
        logger_factory=make_tensorboard_logger_factory(
            log_dir=str(tmp_path), group="multi-seed-smoke",
            config={"model": "Linear", "n_frequency_bands": 4},
        ),
    )
    group_dir = tmp_path / "multi-seed-smoke"
    assert (group_dir / "multi-seed-smoke-seed0").exists()
    assert (group_dir / "multi-seed-smoke-seed1").exists()
    s0 = _read_scalars(group_dir / "multi-seed-smoke-seed0")
    s1 = _read_scalars(group_dir / "multi-seed-smoke-seed1")
    for tags in (s0, s1):
        for t in ("val_loss", "val_cc_norm", "val_cc_norm/p50",
                   "train_loss", "train_cc_norm", "hparams/seed"):
            assert t in tags
