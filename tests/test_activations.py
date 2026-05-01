"""Tests for the parametric output activations.

Covers:
- non_negative_output=True (default) guarantees output ≥ 0 across input range
- non_negative_output=False permits signed output
- forward shape contract
- state_dict round-trip preserves output (closure-pattern bug regression)
- bias=False path
"""

from __future__ import annotations

import torch

from deepSTRF.models.activations import (
    ParametricDoubleExponential,
    ParametricSigmoid,
)


# -------------------------------------------------------------------------
# ParametricSigmoid
# -------------------------------------------------------------------------


def _wide_input(N=8, n_samples=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_samples, N, generator=g) * 5.0     # span roughly (-25, 25)


def test_parametric_sigmoid_default_output_is_non_negative():
    torch.manual_seed(0)
    m = ParametricSigmoid(num_features=8)               # default non_negative_output=True
    out = m(_wide_input(N=8))
    assert (out >= 0).all(), f"output went below zero at min={out.min().item()}"


def test_parametric_sigmoid_signed_mode_can_go_negative():
    """With non_negative_output=False, the activation should be able to dip below 0
    for at least some parameter realisations."""
    saw_negative = False
    for seed in range(5):
        torch.manual_seed(seed)
        m = ParametricSigmoid(num_features=4, non_negative_output=False)
        # nudge baseline negative to ensure we do see negative output
        with torch.no_grad():
            m._raw_a.uniform_(-1.5, -0.5)
        out = m(_wide_input(N=4, seed=seed))
        if (out < 0).any():
            saw_negative = True
            break
    assert saw_negative, "non_negative_output=False should permit negative output"


def test_parametric_sigmoid_default_with_bias_false():
    torch.manual_seed(0)
    m = ParametricSigmoid(num_features=8, bias=False)
    out = m(_wide_input(N=8))
    assert (out >= 0).all()


def test_parametric_sigmoid_forward_shape():
    m = ParametricSigmoid(num_features=8)
    x = torch.randn(2, 8)
    assert m(x).shape == x.shape
    x2 = torch.randn(2, 5, 8)                           # (B, T, N)
    assert m(x2).shape == x2.shape


def test_parametric_sigmoid_state_dict_round_trip():
    """Save + load + run produces identical output to the original."""
    torch.manual_seed(0)
    m1 = ParametricSigmoid(num_features=8)
    m2 = ParametricSigmoid(num_features=8)
    m2.load_state_dict(m1.state_dict())
    x = _wide_input(N=8)
    assert torch.allclose(m1(x), m2(x), atol=1e-7)


def test_parametric_sigmoid_a_property_is_none_when_bias_false():
    m = ParametricSigmoid(num_features=4, bias=False)
    assert m.a is None
    assert m.b is not None


def test_parametric_sigmoid_property_softplus_only_in_non_negative_mode():
    """When non_negative_output=False, the public `b` attribute is the raw param."""
    m = ParametricSigmoid(num_features=4, non_negative_output=False)
    assert torch.equal(m.b, m._raw_b)
    if m.bias:
        assert torch.equal(m.a, m._raw_a)


def test_parametric_sigmoid_is_differentiable():
    m = ParametricSigmoid(num_features=4)
    x = torch.randn(3, 4, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None
    # Check that the raw parameters got gradients too.
    assert m._raw_b.grad is not None
    assert m.c.grad is not None
    assert m.d.grad is not None


# -------------------------------------------------------------------------
# ParametricDoubleExponential
# -------------------------------------------------------------------------


def test_parametric_double_exp_default_output_is_non_negative():
    torch.manual_seed(0)
    m = ParametricDoubleExponential(num_features=8)
    out = m(_wide_input(N=8))
    assert (out >= 0).all(), f"output went below zero at min={out.min().item()}"


def test_parametric_double_exp_signed_mode_can_go_negative():
    saw_negative = False
    for seed in range(5):
        torch.manual_seed(seed)
        m = ParametricDoubleExponential(num_features=4, non_negative_output=False)
        with torch.no_grad():
            m._raw_b.uniform_(-1.5, -0.5)
        out = m(_wide_input(N=4, seed=seed))
        if (out < 0).any():
            saw_negative = True
            break
    assert saw_negative


def test_parametric_double_exp_default_with_bias_false():
    torch.manual_seed(0)
    m = ParametricDoubleExponential(num_features=8, bias=False)
    out = m(_wide_input(N=8))
    assert (out >= 0).all()


def test_parametric_double_exp_forward_shape():
    m = ParametricDoubleExponential(num_features=8)
    x = torch.randn(2, 8)
    assert m(x).shape == x.shape


def test_parametric_double_exp_state_dict_round_trip():
    torch.manual_seed(0)
    m1 = ParametricDoubleExponential(num_features=8)
    m2 = ParametricDoubleExponential(num_features=8)
    m2.load_state_dict(m1.state_dict())
    x = _wide_input(N=8)
    assert torch.allclose(m1(x), m2(x), atol=1e-7)


def test_parametric_double_exp_b_property_is_none_when_bias_false():
    m = ParametricDoubleExponential(num_features=4, bias=False)
    assert m.b is None
    assert m.a is not None


def test_parametric_double_exp_is_differentiable():
    m = ParametricDoubleExponential(num_features=4)
    x = torch.randn(3, 4, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None
    assert m._raw_a.grad is not None
    assert m.k.grad is not None
    assert m.s.grad is not None


# -------------------------------------------------------------------------
# Cross-class: pairing with poisson_loss(log_input=False)
# -------------------------------------------------------------------------


def test_parametric_activations_pair_with_poisson_loss():
    """End-to-end: a default-mode parametric activation should never make
    poisson_loss(log_input=False) NaN, even with extreme inputs."""
    from deepSTRF.metrics import poisson_loss

    for cls in (ParametricSigmoid, ParametricDoubleExponential):
        torch.manual_seed(0)
        m = cls(num_features=4)
        x = torch.randn(2, 4, 1, 50) * 10.0             # (B, N, R=1, T)
        # Activation is applied per-position; we apply on the last dim's N axis.
        # Reshape to (B*R*T, N) → activate → reshape back.
        B, N, R, T = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, N)
        pred = m(flat).reshape(B, R, T, N).permute(0, 3, 1, 2).contiguous()
        gt = torch.full_like(pred, 1.0)
        loss = poisson_loss(pred, gt)
        assert torch.isfinite(loss).all(), f"{cls.__name__} produced non-finite poisson_loss"
