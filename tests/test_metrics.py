"""Tests for the deepSTRF.metrics functional API.

Covers shape contracts, NaN handling (default + mask= override), reduction
semantics, and analytic correctness checks. Mirrors the contract in
``docs/_source/md/metrics_paradigm.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from deepSTRF.metrics import mse_loss, poisson_loss


# -------------------------------------------------------------------------
# mse_loss
# -------------------------------------------------------------------------


def _toy_psth(B=2, N=3, T=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(B, N, 1, T, generator=g)
    gt = torch.randn(B, N, 1, T, generator=g)
    return pred, gt


def test_mse_loss_zero_when_pred_equals_gt():
    pred, _ = _toy_psth()
    out = mse_loss(pred, pred.clone())
    assert torch.allclose(out, torch.tensor(0.0))


def test_mse_loss_reduction_none_returns_per_neuron():
    pred, gt = _toy_psth(B=2, N=4, T=10)
    per_neuron = mse_loss(pred, gt, reduction="none")
    assert per_neuron.shape == (4,)


def test_mse_loss_reduction_mean_is_nanmean_of_none():
    pred, gt = _toy_psth(B=2, N=4, T=10)
    per_neuron = mse_loss(pred, gt, reduction="none")
    scalar = mse_loss(pred, gt, reduction="mean")
    assert torch.allclose(scalar, torch.nanmean(per_neuron))


def test_mse_loss_reduction_sum_is_nansum_of_none():
    pred, gt = _toy_psth(B=2, N=4, T=10)
    per_neuron = mse_loss(pred, gt, reduction="none")
    scalar = mse_loss(pred, gt, reduction="sum")
    assert torch.allclose(scalar, torch.nansum(per_neuron))


def test_mse_loss_invalid_reduction_raises():
    pred, gt = _toy_psth()
    with pytest.raises(ValueError, match="reduction"):
        mse_loss(pred, gt, reduction="MEAN")


def test_mse_loss_shape_mismatch_raises():
    pred = torch.randn(2, 3, 1, 10)
    gt = torch.randn(2, 4, 1, 10)
    with pytest.raises(ValueError, match="shape"):
        mse_loss(pred, gt)


def test_mse_loss_drops_nan_positions_by_default():
    pred = torch.zeros(1, 2, 1, 4)
    gt = torch.tensor([[[[1.0, 1.0, math.nan, math.nan]],
                       [[2.0, 2.0, 2.0,        2.0]]]])
    per_neuron = mse_loss(pred, gt, reduction="none")
    # neuron 0: only first 2 positions valid; (0-1)^2 mean = 1
    # neuron 1: all 4 positions valid; (0-2)^2 mean = 4
    assert torch.allclose(per_neuron, torch.tensor([1.0, 4.0]))


def test_mse_loss_neuron_with_no_valid_positions_is_nan():
    pred = torch.zeros(1, 2, 1, 4)
    gt = torch.full((1, 2, 1, 4), math.nan)
    gt[0, 1, 0, :] = 2.0  # neuron 1 has data; neuron 0 is fully NaN
    per_neuron = mse_loss(pred, gt, reduction="none")
    assert torch.isnan(per_neuron[0])
    assert torch.allclose(per_neuron[1], torch.tensor(4.0))
    # mean reduction nan-skips
    assert torch.allclose(mse_loss(pred, gt, reduction="mean"), torch.tensor(4.0))


def test_mse_loss_mask_override_replaces_nan_mask():
    pred = torch.zeros(1, 1, 1, 4)
    gt = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    # restrict to first 2 positions only
    mask = torch.tensor([[[[True, True, False, False]]]])
    per_neuron = mse_loss(pred, gt, mask=mask, reduction="none")
    # all 4 gt are real & equal to 1; the user's mask drops the last 2.
    # mse over kept positions = (0-1)^2 = 1
    assert torch.allclose(per_neuron, torch.tensor([1.0]))


def test_mse_loss_mask_override_does_not_protect_against_nan():
    """If user passes a mask that includes NaN positions, NaN propagates."""
    pred = torch.zeros(1, 1, 1, 4)
    gt = torch.tensor([[[[1.0, 1.0, math.nan, 1.0]]]])
    # mask says ALL positions are valid (incl. the NaN one)
    mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    per_neuron = mse_loss(pred, gt, mask=mask, reduction="none")
    assert torch.isnan(per_neuron[0])


def test_mse_loss_is_differentiable():
    pred = torch.randn(2, 3, 1, 10, requires_grad=True)
    gt = torch.randn(2, 3, 1, 10)
    loss = mse_loss(pred, gt)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


def test_mse_loss_grad_zero_at_pred_equal_gt():
    gt = torch.randn(1, 2, 1, 5)
    pred = gt.clone().detach().requires_grad_(True)
    mse_loss(pred, gt).backward()
    assert torch.allclose(pred.grad, torch.zeros_like(pred))


def test_mse_loss_bad_mask_dtype_raises():
    pred, gt = _toy_psth(B=1, N=1, T=4)
    bad_mask = torch.ones_like(gt)  # float, not bool
    with pytest.raises(TypeError, match="bool"):
        mse_loss(pred, gt, mask=bad_mask)


# -------------------------------------------------------------------------
# poisson_loss
# -------------------------------------------------------------------------


def test_poisson_loss_shape_contract():
    pred = torch.full((2, 3, 1, 10), 0.5)
    gt = torch.full((2, 3, 1, 10), 1.0)
    per_neuron = poisson_loss(pred, gt, reduction="none")
    assert per_neuron.shape == (3,)


def test_poisson_loss_negative_pred_raises():
    pred = torch.tensor([[[[-0.1, 0.5, 1.0]]]])
    gt = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    with pytest.raises(ValueError, match="non-negative"):
        poisson_loss(pred, gt)


def test_poisson_loss_negative_pred_outside_mask_is_ok():
    """A negative pred at a masked-out position should NOT raise."""
    pred = torch.tensor([[[[-0.1, 0.5, 1.0, 0.5]]]])
    gt = torch.tensor([[[[math.nan, 1.0, 1.0, 1.0]]]])
    per_neuron = poisson_loss(pred, gt, reduction="none")
    assert per_neuron.shape == (1,)
    assert not torch.isnan(per_neuron).item()


def test_poisson_loss_drops_nan_positions_by_default():
    pred = torch.tensor([[[[0.5, 0.5, 0.5, 0.5]]]])
    gt = torch.tensor([[[[1.0, 1.0, math.nan, math.nan]]]])
    per_neuron = poisson_loss(pred, gt, reduction="none")
    # NLL = pred - gt * log(pred + eps) = 0.5 - 1 * log(0.5 + 1e-8) ≈ 0.5 + 0.6931
    expected = 0.5 - 1.0 * math.log(0.5 + 1e-8)
    assert torch.allclose(per_neuron, torch.tensor([expected]), atol=1e-5)


def test_poisson_loss_is_differentiable():
    pred = torch.full((1, 2, 1, 5), 0.5, requires_grad=True)
    gt = torch.full((1, 2, 1, 5), 1.0)
    poisson_loss(pred, gt).backward()
    assert pred.grad is not None


def test_poisson_loss_invalid_reduction_raises():
    pred = torch.full((1, 1, 1, 4), 0.5)
    gt = torch.full((1, 1, 1, 4), 1.0)
    with pytest.raises(ValueError, match="reduction"):
        poisson_loss(pred, gt, reduction="None")  # capitalized typo from old API
