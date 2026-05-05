"""Local round-trip tests for the HF Hub interface (no network calls).

The HF Hub side (download / upload) is exercised manually when publishing
a checkpoint; CI only validates the local save / load contract.
"""

import json
from pathlib import Path

import pytest
import torch

from deepSTRF.models.audio import Linear, StateNet
from deepSTRF.models.prefiltering import make_prefiltering
from deepSTRF.utils.hub import (
    CONFIG_FILE,
    METADATA_FILE,
    MODEL_CARD_FILE,
    WEIGHTS_FILE,
    load_pretrained_from_dir,
    save_pretrained_to_dir,
)


@pytest.fixture
def statenet_pop():
    """Small population StateNet GRU."""
    return StateNet(
        n_frequency_bands=34, hidden_channels=4, kernel_size=7, stride=3,
        connectivity="LC", rnn_type="GRU", out_neurons=8,
    )


# ---------------------------------------------------------------------------
# Auto-capture of __init__ kwargs
# ---------------------------------------------------------------------------

def test_init_kwargs_captured(statenet_pop):
    kw = statenet_pop._init_kwargs
    # All scalar kwargs should be present.
    assert kw["n_frequency_bands"] == 34
    assert kw["hidden_channels"] == 4
    assert kw["rnn_type"] == "GRU"
    assert kw["out_neurons"] == 8


def test_non_json_kwargs_are_dropped():
    """nn.Module kwargs (prefiltering, output_activation) must NOT be in
    _init_kwargs — they would crash the json.dumps(config) call otherwise.
    """
    m = Linear(n_frequency_bands=34, temporal_window_size=9, out_neurons=5,
               prefiltering=make_prefiltering("adaptrans", 34, dt=5.0))
    # 'prefiltering' is an nn.Module; auto-capture must skip it.
    assert "prefiltering" not in m._init_kwargs
    # The scalar kwargs are still there.
    assert m._init_kwargs["n_frequency_bands"] == 34
    assert m._init_kwargs["temporal_window_size"] == 9
    assert m._init_kwargs["out_neurons"] == 5


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_pretrained_writes_expected_files(tmp_path, statenet_pop):
    save_dir = tmp_path / "ckpt"
    statenet_pop.save_pretrained(save_dir, metadata={"dataset": "NS1"})
    files = {p.name for p in save_dir.iterdir()}
    assert {CONFIG_FILE, WEIGHTS_FILE, METADATA_FILE, MODEL_CARD_FILE} <= files


def test_config_records_model_class(tmp_path, statenet_pop):
    save_dir = tmp_path / "ckpt"
    statenet_pop.save_pretrained(save_dir)
    config = json.loads((save_dir / CONFIG_FILE).read_text())
    assert config["_model_class"] == "StateNet"


def test_round_trip_outputs_match_bitwise(tmp_path, statenet_pop):
    """Save → load → forward must match the original output exactly."""
    save_dir = tmp_path / "ckpt"
    statenet_pop.save_pretrained(save_dir)
    loaded = StateNet.from_pretrained(save_dir)

    statenet_pop.eval()
    loaded.eval()
    x = torch.randn(2, 1, 34, 50)
    with torch.no_grad():
        y_orig = statenet_pop(x)
        y_loaded = loaded(x)
    assert y_orig.shape == y_loaded.shape
    assert torch.equal(y_orig, y_loaded), \
        f"max abs diff = {(y_orig - y_loaded).abs().max().item()}"


def test_metadata_round_trip(tmp_path, statenet_pop):
    save_dir = tmp_path / "ckpt"
    metadata = {"dataset": "NS1", "test_cc_norm": 0.77, "epochs": 80}
    statenet_pop.save_pretrained(save_dir, metadata=metadata)

    _, loaded_meta = load_pretrained_from_dir(StateNet, save_dir)
    assert loaded_meta == metadata


def test_no_metadata_round_trip(tmp_path, statenet_pop):
    """metadata.json is optional — load returns None when absent."""
    save_dir = tmp_path / "ckpt"
    statenet_pop.save_pretrained(save_dir)  # no metadata
    _, loaded_meta = load_pretrained_from_dir(StateNet, save_dir)
    assert loaded_meta is None


def test_extra_kwargs_overrides_config(tmp_path):
    """A non-JSON kwarg (prefiltering nn.Module) is dropped at save time;
    re-supplying it through extra_kwargs must produce a working model.
    """
    pre = make_prefiltering("adaptrans", 34, dt=5.0)
    m = Linear(n_frequency_bands=34, temporal_window_size=9, out_neurons=5,
               prefiltering=pre)
    save_dir = tmp_path / "ckpt"
    m.save_pretrained(save_dir)

    # Without extra_kwargs the config defaults to prefiltering=None → C_in=1,
    # so the saved (C_in=2) weights mismatch and load_state_dict raises.
    with pytest.raises(RuntimeError, match=r"size mismatch|shape"):
        Linear.from_pretrained(save_dir)

    # Re-supply the prefilter via extra_kwargs and the model loads cleanly.
    pre2 = make_prefiltering("adaptrans", 34, dt=5.0)
    m_loaded = Linear.from_pretrained(save_dir,
                                      extra_kwargs={"prefiltering": pre2})
    m.eval(); m_loaded.eval()
    x = torch.randn(1, 1, 34, 30)
    with torch.no_grad():
        assert torch.equal(m(x), m_loaded(x))


def test_class_mismatch_warns_but_loads_when_compatible(tmp_path, statenet_pop):
    """Saving as one class then loading as a renamed subclass: we warn but
    don't hard-fail. Here we simulate it by writing the checkpoint and
    flipping the _model_class field to a different name.
    """
    save_dir = tmp_path / "ckpt"
    statenet_pop.save_pretrained(save_dir)
    config = json.loads((save_dir / CONFIG_FILE).read_text())
    config["_model_class"] = "SomeOtherName"
    (save_dir / CONFIG_FILE).write_text(json.dumps(config))

    with pytest.warns(UserWarning, match="saved as 'SomeOtherName'"):
        StateNet.from_pretrained(save_dir)


def test_missing_checkpoint_files_raise(tmp_path, statenet_pop):
    save_dir = tmp_path / "ckpt"
    statenet_pop.save_pretrained(save_dir)

    (save_dir / WEIGHTS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=WEIGHTS_FILE):
        StateNet.from_pretrained(save_dir)


def test_save_pretrained_requires_init_kwargs(tmp_path):
    """A bare nn.Module (not a NeuralModel subclass) has no _init_kwargs;
    save_pretrained_to_dir must refuse it loudly."""
    bare = torch.nn.Linear(3, 3)
    with pytest.raises(AttributeError, match="_init_kwargs"):
        save_pretrained_to_dir(bare, tmp_path / "x")


def test_from_pretrained_returns_metadata_when_requested(tmp_path, statenet_pop):
    save_dir = tmp_path / "ckpt"
    statenet_pop.save_pretrained(save_dir, metadata={"k": "v"})
    model, meta = StateNet.from_pretrained(save_dir, return_metadata=True)
    assert isinstance(model, StateNet)
    assert meta == {"k": "v"}
