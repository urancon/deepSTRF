"""Tests for the deepSTRF.metrics functional API.

Covers shape contracts, NaN handling (default + mask= override), reduction
semantics, and analytic correctness checks. Mirrors the contract in
``docs/_source/md/metrics_paradigm.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from deepSTRF.metrics import (
    coherence,
    corrcoef,
    fve,
    mse_loss,
    noise_power,
    normalized_corrcoef,
    poisson_loss,
    signal_power,
    snr,
)
from deepSTRF.metrics.performance import compute_CCmax, compute_TTRC


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


# -------------------------------------------------------------------------
# corrcoef
# -------------------------------------------------------------------------


def test_corrcoef_self_is_one():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 3, 1, 50, generator=g)
    cc = corrcoef(pred, pred.clone(), reduction="none")
    assert torch.allclose(cc, torch.ones(3), atol=1e-5)


def test_corrcoef_anti_correlation_is_minus_one():
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(1, 1, 1, 100, generator=g)
    cc = corrcoef(-gt, gt, reduction="none")
    assert torch.allclose(cc, torch.tensor([-1.0]), atol=1e-5)


def test_corrcoef_constant_input_returns_nan():
    pred = torch.zeros(1, 1, 1, 10)               # constant -> zero variance
    gt = torch.randn(1, 1, 1, 10)
    cc = corrcoef(pred, gt, reduction="none")
    assert torch.isnan(cc[0])


def test_corrcoef_drops_nan_positions():
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(1, 1, 1, 100, generator=g)
    gt_with_nan = gt.clone()
    gt_with_nan[..., 50:] = float("nan")
    pred = gt.clone()
    cc = corrcoef(pred, gt_with_nan, reduction="none")
    # corrcoef only over the 50 valid positions, where pred==gt → 1
    assert torch.allclose(cc, torch.ones(1), atol=1e-5)


def test_corrcoef_reduction_shape():
    pred = torch.randn(1, 5, 1, 30)
    gt = torch.randn(1, 5, 1, 30)
    assert corrcoef(pred, gt, reduction="none").shape == (5,)
    assert corrcoef(pred, gt, reduction="mean").dim() == 0
    assert corrcoef(pred, gt, reduction="sum").dim() == 0


# -------------------------------------------------------------------------
# fve
# -------------------------------------------------------------------------


def test_fve_self_is_one():
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(1, 3, 1, 50, generator=g)
    out = fve(gt.clone(), gt, reduction="none")
    assert torch.allclose(out, torch.ones(3), atol=1e-5)


def test_fve_zero_pred_against_zero_mean_gt_is_zero():
    """If pred is the mean of gt (i.e. 0 for zero-mean gt), FVE = 0."""
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(1, 1, 1, 200, generator=g)
    gt = gt - gt.mean()
    pred = torch.zeros_like(gt)
    out = fve(pred, gt, reduction="none")
    assert torch.allclose(out, torch.zeros(1), atol=1e-5)


def test_fve_can_be_negative_for_bad_pred():
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(1, 1, 1, 200, generator=g)
    pred = -gt * 5.0                         # wildly worse than predicting the mean
    out = fve(pred, gt, reduction="none")
    assert (out < 0).all()


# -------------------------------------------------------------------------
# signal_power / noise_power / snr (Sahani–Linden)
# -------------------------------------------------------------------------


def _two_repeat_signal_plus_noise(B=1, N=1, T=200, seed=0, sig_scale=1.0, noise_scale=0.0):
    g = torch.Generator().manual_seed(seed)
    signal = torch.randn(B, N, 1, T, generator=g) * sig_scale
    noise = torch.randn(B, N, 2, T, generator=g) * noise_scale
    return signal.expand(B, N, 2, T) + noise


def test_signal_power_noiseless_is_signal_variance():
    """With zero noise, SP should equal the signal variance."""
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(1, 1, 1, 500, generator=g)
    responses = signal.expand(1, 1, 4, 500)        # 4 identical repeats: noise = 0
    sp = signal_power(responses, reduction="none")
    expected = signal.var(unbiased=True)
    assert torch.allclose(sp[0], expected, atol=1e-4)


def test_noise_power_zero_for_identical_repeats():
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(1, 1, 1, 500, generator=g)
    responses = signal.expand(1, 1, 4, 500)
    np_ = noise_power(responses, reduction="none")
    assert torch.allclose(np_[0], torch.tensor(0.0), atol=1e-5)


def test_signal_power_single_trial_returns_nan():
    responses = torch.randn(2, 3, 1, 100)          # R = 1: undefined
    sp = signal_power(responses, reduction="none")
    assert torch.isnan(sp).all()


def test_signal_power_handles_nan_padded_repeats():
    """A cell with NaN in some repeats should still get a sensible SP."""
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(1, 1, 1, 200, generator=g)
    responses = signal.expand(1, 1, 4, 200).clone()
    responses[0, 0, 2:, :] = float("nan")          # only 2 valid repeats
    sp = signal_power(responses, reduction="none")
    assert not torch.isnan(sp).any()


def test_signal_power_long_stim_dominates_short_stim():
    """Length-weighted SP: a long high-variance stim should dominate a short low-variance stim."""
    g = torch.Generator().manual_seed(0)
    # Stim 0: long (T=400), high signal variance ≈ 4
    sig0 = torch.randn(400, generator=g) * 2.0
    # Stim 1: short (T=20), low signal variance ≈ 0.01, NaN-padded to T=400
    sig1 = torch.randn(20, generator=g) * 0.1
    sig1_padded = torch.cat([sig1, torch.full((380,), float("nan"))])

    R = 4
    responses = torch.stack(
        [sig0.unsqueeze(0).expand(R, 400),                            # (R, 400)
         sig1_padded.unsqueeze(0).expand(R, 400)],                    # (R, 400)
        dim=0,
    ).unsqueeze(1)                                                    # (B=2, N=1, R, T=400)

    sp = signal_power(responses, reduction="none")[0].item()

    # Length-weighted: w0=400, w1=20. SP_n ≈ (400 * Var(sig0) + 20 * Var(sig1)) / 420
    sp_long = sig0.var(unbiased=True).item()
    sp_short = sig1.var(unbiased=True).item()
    expected_lw = (400 * sp_long + 20 * sp_short) / 420.0
    expected_eqavg = 0.5 * (sp_long + sp_short)

    # The length-weighted answer is much closer to sp_long; the equal-mean version
    # would give the average of the two stims (factor of ~2 different here).
    assert abs(sp - expected_lw) < 0.1
    assert abs(sp - expected_eqavg) > 1.0


def test_snr_high_when_clean():
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(1, 1, 1, 500, generator=g)
    noise = torch.randn(1, 1, 4, 500, generator=g) * 0.05
    responses = signal.expand(1, 1, 4, 500) + noise
    out = snr(responses, reduction="none")
    assert out[0].item() > 50.0


# -------------------------------------------------------------------------
# normalized_corrcoef
# -------------------------------------------------------------------------


def test_normalized_corrcoef_single_trial_falls_back_to_raw():
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(1, 2, 1, 100, generator=g)        # R = 1
    pred = gt.clone()                                   # perfect prediction
    out = normalized_corrcoef(pred, gt, method="schoppe", reduction="none")
    assert torch.allclose(out, torch.ones(2), atol=1e-5)


def test_normalized_corrcoef_invalid_method_raises():
    pred = torch.randn(1, 1, 1, 50)
    resp = torch.randn(1, 1, 4, 50)
    with pytest.raises(ValueError, match="schoppe"):
        normalized_corrcoef(pred, resp, method="bogus")


def test_normalized_corrcoef_pred_R_must_be_one():
    pred = torch.randn(1, 1, 2, 50)                    # bad: R=2 on pred
    resp = torch.randn(1, 1, 4, 50)
    with pytest.raises(ValueError, match="R-axis"):
        normalized_corrcoef(pred, resp)


def test_normalized_corrcoef_schoppe_perfect_pred():
    """Noiseless responses + perfect prediction → CCnorm ≈ 1."""
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(1, 1, 1, 500, generator=g)
    responses = signal.expand(1, 1, 4, 500).clone()
    pred = signal.clone()                               # (1, 1, 1, 500)
    out = normalized_corrcoef(pred, responses, method="schoppe", reduction="none")
    assert torch.isclose(out[0], torch.tensor(1.0), atol=1e-3)


def test_normalized_corrcoef_hsu_perfect_pred():
    g = torch.Generator().manual_seed(0)
    signal = torch.randn(1, 1, 1, 500, generator=g)
    responses = signal.expand(1, 1, 4, 500).clone()
    pred = signal.clone()
    out = normalized_corrcoef(pred, responses, method="hsu", reduction="none")
    assert torch.isclose(out[0], torch.tensor(1.0), atol=1e-3)


# -------------------------------------------------------------------------
# coherence
# -------------------------------------------------------------------------


def test_coherence_rejects_nan_input():
    pred = torch.randn(1, 1, 1, 256)
    gt = torch.randn(1, 1, 1, 256)
    gt[..., 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        coherence(pred, gt, dt_ms=5.0)


def test_coherence_returns_per_neuron_scalar():
    pred = torch.randn(1, 4, 1, 256)
    gt = torch.randn(1, 4, 1, 256)
    out = coherence(pred, gt, dt_ms=5.0, reduction="none")
    assert out.shape == (4,)


def test_coherence_self_is_high():
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(1, 1, 1, 1024, generator=g)
    out = coherence(gt.clone(), gt, dt_ms=5.0, reduction="none")
    # mean MSC should be near 1 for identical signals
    assert out[0].item() > 0.95


# -------------------------------------------------------------------------
# compute_CCmax / compute_TTRC (internal Wehr helpers)
# -------------------------------------------------------------------------


def test_compute_CCmax_R_one_returns_one():
    responses = torch.randn(2, 1, 100)
    out = compute_CCmax(responses)
    assert torch.allclose(out, torch.ones(2))


def test_compute_CCmax_shape_contract():
    responses = torch.randn(3, 4, 100)
    out = compute_CCmax(responses)
    assert out.shape == (3,)


def test_compute_CCmax_rejects_wrong_rank():
    with pytest.raises(ValueError, match="\\(B, R, T\\)"):
        compute_CCmax(torch.randn(1, 1, 1, 100))


def test_compute_TTRC_R_one_returns_one():
    out = compute_TTRC(torch.randn(2, 1, 50))
    assert torch.allclose(out, torch.ones(2))


def test_compute_TTRC_perfect_repeats():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(1, 1, 200, generator=g)
    responses = base.expand(1, 4, 200)              # 4 identical repeats
    out = compute_TTRC(responses)
    assert torch.allclose(out, torch.ones(1), atol=1e-5)
