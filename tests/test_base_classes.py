"""Contract tests for the `NeuralDataset` / `NeuralModel` base classes.

These classes define the interface every concrete dataset/model subclass must
respect. The tests here are intentionally narrow: they check the attributes
and methods that the contract promises, without exercising any concrete
implementation (which may need data or optional deps).
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# NeuralDataset
# -----------------------------------------------------------------------------

def test_neural_dataset_has_expected_init_signature():
    """The NeuralDataset(path, dt_ms) parent contract."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset
    ds = NeuralDataset(path="/nonexistent", dt_ms=5.0)

    # Core attributes promised by the parent.
    assert ds.path == "/nonexistent"
    assert ds.dt == 5.0
    assert ds.responses == []
    assert ds.stims == []
    assert ds.stim_meta == []
    assert ds.pop_metadata == []
    assert ds.I == []
    assert ds.N_neurons == 0
    assert ds.S == 0


def test_neural_dataset_is_pytorch_dataset_subclass():
    from deepSTRF.datasets.neural_dataset import NeuralDataset
    from torch.utils.data import Dataset

    assert issubclass(NeuralDataset, Dataset)


def test_neural_dataset_public_methods_exist():
    """Methods the subclasses (and users) rely on."""
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    for name in [
        "get_N",
        "get_S",
        "get_pop_metadata",
        "__len__",
        "__getitem__",
        "select_neuron",
        "select_population",
        "select_pop_by_nrn_attr",
        "compute_nrn_masks",
    ]:
        assert hasattr(NeuralDataset, name), f"NeuralDataset is missing method {name!r}"


# -----------------------------------------------------------------------------
# NeuralModel
# -----------------------------------------------------------------------------

def test_neural_model_is_nn_module():
    from deepSTRF.models.neural_model import NeuralModel
    assert issubclass(NeuralModel, nn.Module)


def test_neural_model_instantiation_and_defaults():
    from deepSTRF.models.neural_model import NeuralModel

    m = NeuralModel()
    assert m.O == 1
    assert isinstance(m.output_activation, nn.Identity)

    m2 = NeuralModel(out_neurons=7)
    assert m2.O == 7


def test_neural_model_forward_is_abstract():
    """The base class should raise NotImplementedError on forward()."""
    from deepSTRF.models.neural_model import NeuralModel

    m = NeuralModel()
    try:
        m.forward(torch.zeros(1))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("NeuralModel.forward should raise NotImplementedError")


def test_neural_model_count_trainable_params_zero_by_default():
    """The bare NeuralModel has no trainable params (only an Identity output activation)."""
    from deepSTRF.models.neural_model import NeuralModel
    m = NeuralModel()
    assert m.count_trainable_params() == 0
