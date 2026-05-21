"""Tests for ``deepSTRF.training.auto_config`` + ``slug_from_config``."""

from __future__ import annotations

import torch.nn as nn

from deepSTRF.training import auto_config, slug_from_config


class _ToyEncodingModel(nn.Module):
    """Mimics the deepSTRF audio encoding-model shape: F / T / O attrs,
    prefiltering / core / readout submodules."""

    def __init__(self, F=34, T=9, O=119):
        super().__init__()
        self.F = F
        self.T = T
        self.O = O
        self.prefiltering = nn.Identity()
        self.core = nn.Identity()
        self.readout = self._make_readout()

    def _make_readout(self):
        r = nn.Module()
        r.kernel = nn.Conv2d(1, 1, (3, 3))
        r.activation = nn.ReLU()
        return r


def test_auto_config_extracts_model_class_and_audio_attrs():
    cfg = auto_config(_ToyEncodingModel(F=34, T=9, O=119))
    assert cfg["model"] == "_ToyEncodingModel"
    assert cfg["n_frequency_bands"] == 34
    assert cfg["temporal_window_size"] == 9
    assert cfg["out_neurons"] == 119


def test_auto_config_extracts_submodule_class_names():
    cfg = auto_config(_ToyEncodingModel())
    assert cfg["prefiltering"] == "Identity"
    assert cfg["core"] == "Identity"
    assert cfg["readout"] == "Module"
    assert cfg["readout.kernel"] == "Conv2d"
    assert cfg["readout.activation"] == "ReLU"


def test_auto_config_dataset_name_propagates():
    cfg = auto_config(_ToyEncodingModel(), dataset_name="NS1")
    assert cfg["dataset"] == "NS1"


def test_auto_config_merges_fitter_kwargs_when_json_friendly():
    cfg = auto_config(
        _ToyEncodingModel(),
        fitter_kwargs={"patience": 10, "monitor": "val_cc_norm",
                        "track_per_cell_best": True, "max_epochs": 30},
    )
    assert cfg["patience"] == 10
    assert cfg["monitor"] == "val_cc_norm"
    assert cfg["track_per_cell_best"] is True
    assert cfg["max_epochs"] == 30


def test_auto_config_repr_truncates_non_json_friendly_values():
    obj = object()
    cfg = auto_config(_ToyEncodingModel(), fitter_kwargs={"optimizer": obj})
    assert isinstance(cfg["optimizer"], str)
    assert len(cfg["optimizer"]) <= 60


def test_auto_config_extra_overrides_default_fields():
    cfg = auto_config(
        _ToyEncodingModel(F=34),
        extra={"n_frequency_bands": 999, "extra_key": "hello"},
    )
    assert cfg["n_frequency_bands"] == 999
    assert cfg["extra_key"] == "hello"


def test_auto_config_is_json_serialisable():
    import json
    cfg = auto_config(
        _ToyEncodingModel(),
        dataset_name="NS1",
        fitter_kwargs={"patience": 10, "max_epochs": 30, "mode": "max"},
    )
    json.dumps(cfg)  # raises if any value is not JSON-friendly


def test_slug_from_config_default_fields():
    cfg = {
        "model": "Linear", "dataset": "NS1",
        "temporal_window_size": 9, "n_frequency_bands": 34,
    }
    assert slug_from_config(cfg) == "linear-ns1-T9-F34"


def test_slug_from_config_missing_keys_silently_skipped():
    cfg = {"model": "Linear", "temporal_window_size": 9}
    assert slug_from_config(cfg) == "linear-T9"


def test_slug_from_config_custom_fields():
    cfg = {"a": "x", "b": 5}
    slug = slug_from_config(cfg, fields=[("a", str.upper), ("b", lambda v: f"b{v}")])
    assert slug == "X-b5"
