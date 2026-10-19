"""Noise-corrected metrics on silent and nearly silent responses: no NaN poisoning,
no blow-up, NaN-free reductions."""
import torch

from deepSTRF.metrics import normalized_corrcoef
from deepSTRF.metrics.performance import _split_half_rho, compute_CCmax

B, R, T = 4, 10, 200


def _driven(seed=0):
    g = torch.Generator().manual_seed(seed)
    rate = torch.rand(B, 1, 1, T, generator=g) * 0.3
    resp = torch.poisson(rate.expand(B, 1, R, T).clone(), generator=g)
    pred = rate + 0.01 * torch.randn(B, 1, 1, T, generator=g)
    return pred, resp


def test_hsu_drops_a_silent_stimulus_instead_of_going_nan():
    pred, resp = _driven()
    resp[1] = 0                                   # the neuron never fires to stim 1
    full = normalized_corrcoef(pred, resp, method="hsu", reduction="none")
    assert torch.isfinite(full).all()
    # CCmax ignores the silent stim, so it equals the CCmax of the other stims
    from deepSTRF.metrics.performance import _ccmax_per_neuron
    keep = [0, 2, 3]
    ccmax_full = _ccmax_per_neuron(resp, ~resp.isnan(), 126)
    ccmax_kept = _ccmax_per_neuron(resp[keep], ~resp[keep].isnan(), 126)
    assert torch.allclose(ccmax_full, ccmax_kept)


def test_split_half_rho_silent_block_and_one_sided_split():
    assert _split_half_rho(torch.zeros(4, 20), 126) is None          # silent: undefined
    block = torch.zeros(2, 8)
    block[0, ::2] = 1                                                # one half fired, one never
    assert float(_split_half_rho(block, 126)) == 0.0                 # counts as rho = 0


def test_compute_ccmax_silent_cell_is_nan_not_error():
    x = torch.zeros(2, 6, 50)
    x[1] = torch.poisson(torch.full((6, 50), 0.5))
    out = compute_CCmax(x)
    assert torch.isnan(out[0]) and not torch.isinf(out).any()


def test_schoppe_rounding_level_signal_power_is_nan():
    pred, _ = _driven()
    resp = torch.zeros(B, 1, R, T)
    resp[0, 0, 0, 50] = 1                         # one spike in the whole set
    assert torch.isnan(normalized_corrcoef(pred, resp, method="schoppe", reduction="none")).all()
    # the plain SP > 0 test (threshold 0) is what used to explode
    raw = normalized_corrcoef(pred, resp, method="schoppe", reduction="none", sp_rel_threshold=0.0)
    assert torch.isnan(raw).all() or raw.abs().max() > 10


def test_reductions_stay_finite_with_silent_neurons():
    pred, resp = _driven()
    pred = pred.expand(B, 3, 1, T).clone()
    resp = resp.expand(B, 3, R, T).clone()
    resp[:, 1] = 0                                # one neuron silent everywhere
    resp[1, 2] = 0                                # one neuron silent on one stim
    for method in ("schoppe", "hsu"):
        per = normalized_corrcoef(pred, resp, method=method, reduction="none")
        assert torch.isnan(per[1]) and torch.isfinite(per[[0, 2]]).all()
        assert torch.isfinite(normalized_corrcoef(pred, resp, method=method, reduction="mean"))
