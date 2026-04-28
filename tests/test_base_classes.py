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
        "get_neuron_metadata",
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
    assert ds.neuron_metadata == []
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
            self.neuron_metadata = [{"uid": "n0"}, {"uid": "n1"}]
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
