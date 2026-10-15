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
    # np.hanning(5) = [0, .5, 1, .5, 0] -> normalised [0, .25, .5, .25, 0]
    assert torch.allclose(y[0, 9:12], torch.tensor([0.25, 0.5, 0.25], dtype=y.dtype))
    assert y[0, :9].abs().sum() == 0 and y[0, 12:].abs().sum() == 0


def test_no_op_at_10ms():
    x = torch.rand(3, 50, dtype=torch.float64)
    assert torch.allclose(hanning_smooth(x, window_ms=21.0, dt_ms=10.0), x)


def test_constant_interior_unchanged():
    x = torch.full((2, 3, 40), 0.7, dtype=torch.float64)
    y = hanning_smooth(x, window_ms=21.0, dt_ms=1.0)
    assert torch.allclose(y[..., 10:-10], x[..., 10:-10])


@pytest.mark.parametrize("shape", [(5, 40), (4, 3, 40), (2, 4, 3, 40)])
def test_transform_matches_function(shape):
    x = torch.rand(*shape, dtype=torch.float64)
    t = ResponseSmoothingTransform(dt_ms=1.0, window_size_ms=21.0)
    assert torch.allclose(t(x), hanning_smooth(x, window_ms=21.0, dt_ms=1.0))
