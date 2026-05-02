"""Tests for ``deepSTRF.models.layers`` — pluggable STRF kernels.

Covers SeparableSTRF and ParametricSTRF as drop-in ``kernel=`` arguments
for STRFReadout-using models. CausalSTRFConv default behaviour is
exercised by ``tests/test_audio_models.py``.
"""

from __future__ import annotations

import pytest
import torch

from deepSTRF.models.audio import Linear
from deepSTRF.models.layers import ParametricSTRF, SeparableSTRF


# -----------------------------------------------------------------------------
# SeparableSTRF — shape contract
# -----------------------------------------------------------------------------


def test_separable_strf_build_kernel_shape():
    m = SeparableSTRF(F=34, T=15, C_in=1, C_out=119)
    kernel = m.build_kernel()
    assert kernel.shape == (119, 1, 34, 15), (
        f"build_kernel() must emit (C_out, C_in, F, T); got {tuple(kernel.shape)}"
    )


def test_separable_strf_forward_shape():
    m = SeparableSTRF(F=34, T=15, C_in=1, C_out=8)
    x = torch.randn(2, 1, 34, 100)
    y = m(x)
    # conv2d with no padding: output T = T_in - T_kernel + 1 = 100 - 15 + 1 = 86
    # output F = F_in - F_kernel + 1 = 34 - 34 + 1 = 1
    assert y.shape == (2, 8, 1, 86), f"got {tuple(y.shape)}"


def test_separable_strf_param_count_is_rank1():
    """Separable STRF should use ~C_out·C_in·(F+T) params, not C_out·C_in·F·T."""
    F, T, C_in, C_out = 34, 15, 1, 50
    m = SeparableSTRF(F=F, T=T, C_in=C_in, C_out=C_out, bias=False)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == C_out * C_in * (F + T), f"got {n}"


def test_separable_strf_kernel_is_outer_product():
    """The (F, T) kernel slice for one (C_out, C_in) cell should be a rank-1
    outer product w_f ⊗ w_t."""
    m = SeparableSTRF(F=8, T=4, C_in=1, C_out=1)
    K = m.build_kernel()                      # (1, 1, 8, 4)
    K2d = K[0, 0]                             # (8, 4)
    # rank-1 ↔ all 2x2 minors are 0
    minor = K2d[0, 0] * K2d[1, 1] - K2d[0, 1] * K2d[1, 0]
    assert minor.abs().item() < 1e-6, f"non-rank-1 minor: {minor.item()}"


# -----------------------------------------------------------------------------
# Pluggable kernels work end-to-end on Linear
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_factory,name", [
    (lambda F, T, N: None,                                              "vanilla"),
    (lambda F, T, N: SeparableSTRF(F, T, C_in=1, C_out=N),              "separable"),
    (lambda F, T, N: ParametricSTRF(F, T, C_in=1, C_out=N, num_gaussians=4), "parametric"),
])
def test_linear_with_kernel_runs_end_to_end(kernel_factory, name):
    F, T, N = 16, 9, 5
    model = Linear(n_frequency_bands=F, temporal_window_size=T, out_neurons=N,
                   kernel=kernel_factory(F, T, N))
    x = torch.randn(2, 1, F, 50)
    y = model(x)
    assert y.shape == (2, N, 1, 50), (
        f"{name} kernel: expected (B, N, 1, T); got {tuple(y.shape)}"
    )
