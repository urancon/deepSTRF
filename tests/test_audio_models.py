"""
Forward-shape and causality contract tests for the audio model zoo.

Every concrete model in ``deepSTRF.models.audio`` must satisfy:

1. Construction with sensible defaults works.
2. ``forward((B, 1, F, T))`` returns ``(B, N, 1, T)`` — the canonical
   ``(B, N, R=1, T)`` rank documented in ``model_paradigm.md`` §3.
3. Bitwise causality in eval mode: changing future timesteps leaves past
   outputs unchanged. (Train-mode dropout is allowed to wobble; eval is
   the contract surface.)
4. ``validate()`` passes (so the four-slot template's invariants hold).
5. Plays nicely with ``AdapTrans`` (``C_in == 2``) and ``ICAdaptation``
   (``C_in == 1``) prefilters.
"""
import pytest
import torch

from deepSTRF.models.audio import (
    Linear, LinearNonlinear, NetworkReceptiveField, DNet,
    ConvNet2D, Transformer, StateNet,
)
from deepSTRF.models.prefiltering import make_prefiltering


# Reproducible everywhere — many models contain Parameter(torch.rand(...))
# or Conv2d which depend on global RNG.
def _seed():
    torch.manual_seed(0)


# Common shape parameters
B, F, T_in = 2, 34, 50
T_window = 9
N = 5


# ---------------------------------------------------------------------------
# Factories — one per model. Centralized so the parametrize lists below
# stay readable; each factory takes a fresh prefilter (or None) so smoke
# tests can swap prefilters cleanly.
# ---------------------------------------------------------------------------

def _make_linear(prefilter=None):
    return Linear(n_frequency_bands=F, temporal_window_size=T_window,
                  out_neurons=N, prefiltering=prefilter)


def _make_linear_nonlinear(prefilter=None):
    return LinearNonlinear(n_frequency_bands=F, temporal_window_size=T_window,
                           out_neurons=N, prefiltering=prefilter)


def _make_nrf(prefilter=None):
    return NetworkReceptiveField(n_frequency_bands=F, temporal_window_size=T_window,
                                 n_hidden=8, out_neurons=N, prefiltering=prefilter)


def _make_dnet(prefilter=None):
    return DNet(n_frequency_bands=F, temporal_window_size=T_window,
                n_hidden=8, out_neurons=N, prefiltering=prefilter)


def _make_convnet2d(prefilter=None):
    return ConvNet2D(n_frequency_bands=F, kernel_size=(3, 9),
                     c_hidden=8, n_hidden=16, out_neurons=N, prefiltering=prefilter)


def _make_transformer(prefilter=None):
    return Transformer(n_frequency_bands=F, temporal_window_size=T_window,
                       token_size=(F, 1), embedding_dim=32, n_heads=2, n_layers=1,
                       out_neurons=N, prefiltering=prefilter)


def _make_statenet(prefilter=None):
    return StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                    rnn_type='GRU', out_neurons=N, prefiltering=prefilter)


# All factories, parametrized for the standard tests below.
ALL_FACTORIES = [
    pytest.param(_make_linear,           id='Linear'),
    pytest.param(_make_linear_nonlinear, id='LinearNonlinear'),
    pytest.param(_make_nrf,              id='NetworkReceptiveField'),
    pytest.param(_make_dnet,             id='DNet'),
    pytest.param(_make_convnet2d,        id='ConvNet2D'),
    pytest.param(_make_transformer,      id='Transformer'),
    pytest.param(_make_statenet,         id='StateNet'),
]


# ---------------------------------------------------------------------------
# Forward output rank: (B, N, R=1, T)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_forward_output_rank_no_prefilter(factory):
    _seed()
    m = factory()
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in), \
        f"{type(m).__name__}: expected (B={B}, N={N}, R=1, T={T_in}); got {tuple(y.shape)}"


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_forward_output_rank_with_adaptrans(factory):
    """AdapTrans gives C_in=2 — the model must thread it through correctly."""
    _seed()
    pref = make_prefiltering('adaptrans', n_frequency_bands=F, dt=5.0)
    m = factory(prefilter=pref)
    assert m.C_in == 2, f"AdapTrans should give C_in=2; got C_in={m.C_in}"
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in), \
        f"{type(m).__name__}+AdapTrans: expected (B={B}, N={N}, R=1, T={T_in}); got {tuple(y.shape)}"


@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_forward_output_rank_with_icadaptation(factory):
    """ICAdaptation gives C_in=1 — single-channel paper-faithful prefilter."""
    _seed()
    pref = make_prefiltering('icadaptation', n_frequency_bands=F, dt=5.0)
    m = factory(prefilter=pref)
    assert m.C_in == 1, f"ICAdaptation should give C_in=1; got C_in={m.C_in}"
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, N, 1, T_in)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_bitwise_causality_in_eval_mode(factory):
    """Changing future input timesteps must not change past output timesteps."""
    _seed()
    m = factory()
    m.eval()
    cut = T_in // 2
    x = torch.randn(B, 1, F, T_in)
    x_perturbed = x.clone()
    x_perturbed[..., cut:] = torch.randn_like(x_perturbed[..., cut:])
    with torch.no_grad():
        y = m(x)
        y_perturbed = m(x_perturbed)
    diff = (y - y_perturbed)[..., :cut].abs().max().item()
    # Allow 1e-6 to absorb rounding in long FP chains (S4 hits ~1e-7).
    assert diff < 1e-5, \
        f"{type(m).__name__}: causality violation; max past diff = {diff:.2e}"


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_validate_passes(factory):
    _seed()
    m = factory()
    m.validate()  # must not raise


# ---------------------------------------------------------------------------
# Output rank invariant: variable-T inputs are honored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
@pytest.mark.parametrize("T_test", [25, 50, 100])
def test_output_T_matches_input_T(factory, T_test):
    """Models must accept arbitrary input lengths and preserve T (causal pad)."""
    _seed()
    m = factory()
    x = torch.randn(B, 1, F, T_test)
    y = m(x)
    assert y.shape[-1] == T_test, \
        f"{type(m).__name__}: input T={T_test} produced output T={y.shape[-1]}"


# ---------------------------------------------------------------------------
# Single-neuron edge case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_single_neuron(factory):
    """N=1 should work — degenerate but supported."""
    _seed()
    # Override out_neurons=1; factories default to N=5.
    if factory is _make_linear:
        m = Linear(n_frequency_bands=F, temporal_window_size=T_window, out_neurons=1)
    elif factory is _make_linear_nonlinear:
        m = LinearNonlinear(n_frequency_bands=F, temporal_window_size=T_window, out_neurons=1)
    elif factory is _make_nrf:
        m = NetworkReceptiveField(n_frequency_bands=F, temporal_window_size=T_window,
                                  n_hidden=8, out_neurons=1)
    elif factory is _make_dnet:
        m = DNet(n_frequency_bands=F, temporal_window_size=T_window, n_hidden=8, out_neurons=1)
    elif factory is _make_convnet2d:
        m = ConvNet2D(n_frequency_bands=F, kernel_size=(3, 9), c_hidden=8, n_hidden=16, out_neurons=1)
    elif factory is _make_transformer:
        m = Transformer(n_frequency_bands=F, temporal_window_size=T_window,
                        token_size=(F, 1), embedding_dim=32, n_heads=2, n_layers=1, out_neurons=1)
    elif factory is _make_statenet:
        m = StateNet(n_frequency_bands=F, kernel_size=7, hidden_channels=4,
                     rnn_type='GRU', out_neurons=1)
    else:
        pytest.skip(f"no single-neuron path for {factory}")
    x = torch.randn(B, 1, F, T_in)
    y = m(x)
    assert y.shape == (B, 1, 1, T_in)


# ---------------------------------------------------------------------------
# Slots — every concrete model populates the four canonical slots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_FACTORIES)
def test_canonical_slots_populated(factory):
    """All four template slots must hold nn.Module instances after __init__."""
    import torch.nn as nn
    _seed()
    m = factory()
    for slot in ('wav2spec', 'prefiltering', 'core', 'readout'):
        assert hasattr(m, slot), f"{type(m).__name__} missing slot {slot!r}"
        attr = getattr(m, slot)
        assert isinstance(attr, nn.Module), \
            f"{type(m).__name__}.{slot} is {type(attr).__name__}, expected nn.Module"
