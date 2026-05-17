"""Contract tests for the `NeuralDataset` / `NeuralModel` base classes.

These classes define the interface every concrete dataset/model subclass must
respect. The tests here are intentionally narrow: they check the attributes
and methods that the contract promises, without exercising any concrete
implementation (which may need data or optional deps).
"""

import inspect

import pytest
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# NeuralDataset
# -----------------------------------------------------------------------------

def test_neural_dataset_init_signature():
    """NeuralDataset.__init__ must accept `path` and `dt_ms`."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    sig = inspect.signature(NeuralDataset.__init__)
    params = list(sig.parameters)
    assert params[0] == "self"
    assert "path" in params, f"NeuralDataset.__init__ must accept `path` (got {params})"
    assert "dt_ms" in params, f"NeuralDataset.__init__ must accept `dt_ms` (got {params})"


def test_neural_dataset_is_pytorch_dataset_subclass():
    from deepSTRF.datasets.neural_dataset import NeuralDataset
    from torch.utils.data import Dataset

    assert issubclass(NeuralDataset, Dataset)


def test_neural_dataset_is_abstract_base_class():
    """NeuralDataset should be a proper ABC."""
    from abc import ABC
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    assert issubclass(NeuralDataset, ABC)


def test_neural_dataset_public_methods_exist():
    """Methods subclasses and users rely on."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    for name in [
        "__init__",
        "__len__",
        "__getitem__",
        "select_neuron",
        "select_population",
        "select_pop_by_nrn_attr",
        "get_N",
        "get_S",
        "get_nrn_meta",
        "nrn_masks",   # @property, reachable via hasattr
        "validate",
    ]:
        assert hasattr(NeuralDataset, name), f"NeuralDataset is missing method {name!r}"


def test_neural_dataset_base_attributes_after_init():
    """A bare NeuralDataset has the scaffolding attributes ready for subclass population."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    ds = NeuralDataset("/tmp/nowhere", dt_ms=1.0)
    assert ds.path == "/tmp/nowhere"
    assert ds.dt == 1.0
    assert ds.responses == []
    assert ds.stims == []
    assert ds.stim_meta == []
    assert ds.nrn_meta == []
    assert ds.N_neurons == 0
    assert ds.I == []
    # nrn_masks is a derived @property: empty-responses state yields a (0, 0) bool tensor
    assert tuple(ds.nrn_masks.shape) == (0, 0)
    assert ds.nrn_masks.dtype == torch.bool


def test_neural_dataset_validate_fails_on_empty():
    """Calling validate() on an un-populated base must raise."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    ds = NeuralDataset("/tmp/nowhere", dt_ms=1.0)
    with pytest.raises(AssertionError):
        ds.validate()


def test_neural_dataset_validate_succeeds_on_populated_fixture():
    """A minimal correctly-populated subclass passes validate()."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    class _Fixture(NeuralDataset):
        def __init__(self):
            super().__init__("/tmp/nowhere", dt_ms=1.0)
            self.N_neurons = 2
            self.stim_meta = [("s0",), ("s1",)]
            self.stims = [torch.zeros(1, 4, 10), torch.zeros(1, 4, 10)]
            self.responses = [
                [torch.zeros(1, 10), torch.zeros(1, 10)],
                [torch.zeros(1, 10), torch.zeros(1, 10)],
            ]
            self.nrn_meta = [{"uid": "n0"}, {"uid": "n1"}]
            self.validate()

    fx = _Fixture()  # must not raise
    assert fx.N_neurons == 2
    assert tuple(fx.nrn_masks.shape) == (2, 2)
    assert fx.nrn_masks.dtype == torch.bool


def test_audio_neural_dataset_validate_requires_F():
    """AudioNeuralDataset.validate() adds self.F > 0 check."""
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    ds = AudioNeuralDataset("/tmp/nowhere", dt_ms=1.0)
    assert ds.F == -1  # sentinel set by base
    with pytest.raises(AssertionError):
        ds.validate()


# -----------------------------------------------------------------------------
# NeuralModel
# -----------------------------------------------------------------------------

def _make_concrete_neural_model(**kwargs):
    """Minimal NeuralModel subclass for testing — populates the required readout slot."""
    from deepSTRF.models.neural_model import NeuralModel

    class _Concrete(NeuralModel):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.readout = nn.Identity()

    return _Concrete(**kwargs)


def test_neural_model_is_nn_module_and_abc():
    from abc import ABC
    from deepSTRF.models.neural_model import NeuralModel

    assert issubclass(NeuralModel, nn.Module)
    assert issubclass(NeuralModel, ABC)


def test_neural_model_concrete_instantiation_and_defaults():
    m = _make_concrete_neural_model()
    assert m.O == 1
    # Four canonical slots default to nn.Module instances
    assert isinstance(m.wav2spec, nn.Module)
    assert isinstance(m.prefiltering, nn.Module)
    assert isinstance(m.core, nn.Module)
    assert isinstance(m.readout, nn.Module)

    m2 = _make_concrete_neural_model(out_neurons=7)
    assert m2.O == 7


def test_neural_model_count_trainable_params_zero_by_default():
    """A bare concrete model has no trainable params (Identity slots only)."""
    m = _make_concrete_neural_model()
    assert m.count_trainable_params() == 0


def test_neural_model_validate_passes_on_valid_instance():
    m = _make_concrete_neural_model()
    m.validate()  # must not raise


def test_neural_model_validate_fails_on_bad_state():
    m = _make_concrete_neural_model()
    m.O = 0
    with pytest.raises(AssertionError):
        m.validate()


def test_neural_model_validate_fails_without_readout():
    """validate() must reject a model whose readout slot was not populated."""
    from deepSTRF.models.neural_model import NeuralModel

    class _NoReadout(NeuralModel):
        pass

    m = _NoReadout()
    with pytest.raises(AssertionError):
        m.validate()


# ---------------------------------------------------------------------------
# standardize_stims
# ---------------------------------------------------------------------------


def _stim_dataset_fixture():
    """Tiny dataset with three (1, F=4, T=10) stims; per-band scales differ."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    class _Fixture(NeuralDataset):
        def __init__(self):
            super().__init__("/tmp/nowhere", dt_ms=1.0)
            self.N_neurons = 1
            self.stim_meta = [("s0",), ("s1",), ("s2",)]
            torch.manual_seed(0)
            # Per-band scales: band 0 ~ N(2, 1), band 1 ~ N(0, 0.5),
            # band 2 ~ N(-1, 2), band 3 ~ N(0, 0.1).  Different across stims.
            scales = torch.tensor([1.0, 0.5, 2.0, 0.1]).view(1, 4, 1)
            offsets = torch.tensor([2.0, 0.0, -1.0, 0.0]).view(1, 4, 1)
            self.stims = [
                torch.randn(1, 4, 10) * scales + offsets for _ in range(3)
            ]
            self.responses = [[torch.zeros(1, 10)] for _ in range(3)]
            self.nrn_meta = [{"uid": "n0"}]
            self.validate()

    return _Fixture()


def test_standardize_stims_per_band_makes_unit_std():
    """Per-band standardization on all stims yields per-band std == 1."""
    fx = _stim_dataset_fixture()
    fx.standardize_stims(per_band=True)
    cat = torch.cat(fx.stims, dim=-1)            # (1, F=4, T_total)
    assert torch.allclose(cat.mean(dim=(0, 2)), torch.zeros(4), atol=1e-6)
    assert torch.allclose(cat.std(dim=(0, 2)), torch.ones(4), atol=1e-6)


def test_standardize_stims_subset_applies_to_all():
    """Stats from a subset are applied to ALL stims, including unselected ones."""
    fx = _stim_dataset_fixture()
    # Compute stats from stims 0 and 1; stim 2 (held-out test) should be
    # transformed with the same stats but its post-standardized std need
    # not be 1.
    fx.standardize_stims(stim_indices=[0, 1], per_band=True)
    train_cat = torch.cat([fx.stims[0], fx.stims[1]], dim=-1)
    held_out = fx.stims[2]
    # stims 0, 1: per-band std = 1 by construction.
    assert torch.allclose(train_cat.std(dim=(0, 2)), torch.ones(4), atol=1e-6)
    # stim 2: per-band std generally != 1 (different sample).
    held_std = held_out.std(dim=(0, 2))
    assert not torch.allclose(held_std, torch.ones(4), atol=1e-2)


def test_standardize_stims_stores_normalization_dict():
    fx = _stim_dataset_fixture()
    out = fx.standardize_stims(stim_indices=[0, 1], per_band=True)
    assert out is fx.stim_normalization
    assert set(out) == {"mean", "std", "per_band", "stim_indices"}
    assert out["per_band"] is True
    assert out["stim_indices"] == [0, 1]
    assert out["mean"].shape == (1, 4, 1)
    assert out["std"].shape == (1, 4, 1)


def test_standardize_stims_global_scalar_path():
    fx = _stim_dataset_fixture()
    out = fx.standardize_stims(per_band=False)
    assert out["mean"].dim() == 0
    assert out["std"].dim() == 0
    cat = torch.cat(fx.stims, dim=-1)
    assert abs(cat.mean().item()) < 1e-6
    assert abs(cat.std().item() - 1.0) < 1e-6


def test_standardize_stims_empty_subset_raises():
    fx = _stim_dataset_fixture()
    with pytest.raises(ValueError, match="no stims"):
        fx.standardize_stims(stim_indices=[])


# ---------------------------------------------------------------------------
# normalize_responses
# ---------------------------------------------------------------------------


def _response_dataset_fixture():
    """Tiny dataset with 3 stims × 2 neurons; per-neuron scales differ."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    class _Fixture(NeuralDataset):
        def __init__(self):
            super().__init__("/tmp/nowhere", dt_ms=1.0)
            self.N_neurons = 2
            self.stim_meta = [("s0",), ("s1",), ("s2",)]
            self.stims = [torch.zeros(1, 1, 10) for _ in range(3)]
            torch.manual_seed(0)
            # neuron 0: positive, scale ~5 ; neuron 1: signed, scale ~2
            self.responses = [
                [(torch.rand(1, 10) * 5.0),
                 (torch.randn(1, 10) * 2.0)] for _ in range(3)
            ]
            self.nrn_meta = [{"uid": "n0"}, {"uid": "n1"}]
            self.validate()

    return _Fixture()


def test_normalize_responses_max_default():
    fx = _response_dataset_fixture()
    fx.normalize_responses(method="max")
    # all valid (s, r, t) lie in [-1, 1]; for non-neg neuron 0 also [0, 1]
    for s in range(3):
        for n in range(2):
            r = fx.responses[s][n]
            assert r.abs().max().item() <= 1.0 + 1e-6


def test_normalize_responses_zscore_unit_variance():
    fx = _response_dataset_fixture()
    fx.normalize_responses(method="zscore")
    for n in range(2):
        cat = torch.cat([fx.responses[s][n].flatten() for s in range(3)])
        assert abs(cat.mean().item()) < 1e-5
        assert abs(cat.std().item() - 1.0) < 1e-5


def test_normalize_responses_subset_applies_to_all():
    fx = _response_dataset_fixture()
    fx.normalize_responses(method="zscore", stim_indices=[0, 1])
    # train subset: zero-mean / unit-std per neuron
    for n in range(2):
        cat = torch.cat([fx.responses[s][n].flatten() for s in [0, 1]])
        assert abs(cat.mean().item()) < 1e-5
        assert abs(cat.std().item() - 1.0) < 1e-5
    # held-out stim 2 was transformed with the same stats — generally != 1 std
    for n in range(2):
        held = fx.responses[2][n].flatten()
        assert abs(held.std().item() - 1.0) > 1e-3


def test_normalize_responses_stores_dict():
    fx = _response_dataset_fixture()
    out = fx.normalize_responses(method="zscore", stim_indices=[0, 1])
    assert out is fx.response_normalization
    assert set(out) == {"method", "scale", "offset", "stim_indices"}
    assert out["method"] == "zscore"
    assert out["scale"].shape == (2,)
    assert out["offset"].shape == (2,)
    assert out["stim_indices"] == [0, 1]


def test_normalize_responses_preserves_nan_sentinel():
    """The structural (1, 1) NaN-sentinel must survive normalization untouched."""
    fx = _response_dataset_fixture()
    fx.responses[1][0] = torch.full((1, 1), float("nan"))   # uncorded combo
    fx.normalize_responses(method="zscore")
    survivor = fx.responses[1][0]
    assert survivor.shape == (1, 1)
    assert torch.isnan(survivor).all().item()


def test_normalize_responses_invalid_method():
    fx = _response_dataset_fixture()
    with pytest.raises(ValueError, match="method"):
        fx.normalize_responses(method="bogus")
