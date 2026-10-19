"""hanning_smooth / ResponseSmoothingTransform: sum-normalised Hanning kernel."""
import pytest
import torch

from deepSTRF.utils import ResponseSmoothingTransform, hanning_smooth


def _impulse(T=21, at=10):
    x = torch.zeros(1, T, dtype=torch.float64)
    x[0, at] = 1.0
    return x


@pytest.mark.parametrize("dt_ms", [1.0, 2.0, 5.0, 10.0])
def test_preserves_spike_count(dt_ms):
    y = hanning_smooth(_impulse(T=61, at=30), window_ms=21.0, dt_ms=dt_ms)
    assert torch.isclose(y.sum(), torch.tensor(1.0, dtype=y.dtype))


def test_kernel_at_5ms():
    y = hanning_smooth(_impulse(), window_ms=21.0, dt_ms=5.0)
    # MATLAB hanning(5) = [.25, .75, 1, .75, .25] (no zero end points), sum 3
    k = torch.tensor([0.25, 0.75, 1.0, 0.75, 0.25], dtype=y.dtype) / 3
    assert torch.allclose(y[0, 8:13], k)
    assert y[0, :8].abs().sum() == 0 and y[0, 13:].abs().sum() == 0


def test_kernel_at_10ms_smooths():
    y = hanning_smooth(_impulse(), window_ms=21.0, dt_ms=10.0)
    assert torch.allclose(y[0, 9:12], torch.tensor([0.25, 0.5, 0.25], dtype=y.dtype))


def test_matches_matlab_hanning():
    import numpy as np
    for K in (3, 5, 21):
        n = np.arange(1, K + 1)
        matlab = 0.5 * (1 - np.cos(2 * np.pi * n / (K + 1)))
        assert np.allclose(np.hanning(K + 2)[1:-1], matlab)


def test_constant_interior_unchanged():
    x = torch.full((2, 3, 40), 0.7, dtype=torch.float64)
    y = hanning_smooth(x, window_ms=21.0, dt_ms=1.0)
    assert torch.allclose(y[..., 10:-10], x[..., 10:-10])


@pytest.mark.parametrize("shape", [(5, 40), (4, 3, 40), (2, 4, 3, 40)])
def test_transform_matches_function(shape):
    x = torch.rand(*shape, dtype=torch.float64)
    t = ResponseSmoothingTransform(dt_ms=1.0, window_size_ms=21.0)
    assert torch.allclose(t(x), hanning_smooth(x, window_ms=21.0, dt_ms=1.0))
