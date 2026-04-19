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
    """NeuralDataset.__init__ must accept `path` as its first real argument.

    Signature-agnostic test: doesn't instantiate (the local WIP base also
    requires `dt_ms`, tracked base does not — both should satisfy this test).
    """
    import inspect
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    sig = inspect.signature(NeuralDataset.__init__)
    params = list(sig.parameters)
    assert params[0] == "self"
    assert "path" in params, f"NeuralDataset.__init__ must accept `path` (got {params})"


def test_neural_dataset_is_pytorch_dataset_subclass():
    from deepSTRF.datasets.neural_dataset import NeuralDataset
    from torch.utils.data import Dataset

    assert issubclass(NeuralDataset, Dataset)


def test_neural_dataset_public_methods_exist():
    """Methods the subclasses (and users) rely on, scoped to what's on the
    current base. Forward-looking methods (get_S, compute_nrn_masks, ...)
    are part of the WIP base-class modernization and will be added here
    once that lands.
    """
    from deepSTRF.datasets.neural_dataset import NeuralDataset

    for name in [
        "__init__",
        "__len__",
        "__getitem__",
        "select_neuron",
        "select_population",
        "get_N",
        "get_pop_metadata",
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
