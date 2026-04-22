"""Contract tests for NeuralDataset concatenation.

Covers both the module-level ``concat_neural_datasets`` entry point and
the ``__add__`` sugar. Uses an inline minimal ``AudioNeuralDataset``
subclass fixture so tests don't depend on any real dataset on disk.
"""

import pytest
import torch


def _fake_audio_dataset(N: int, S: int, dt: float = 1.0, F: int = 4, T: int = 8):
    """Minimal concrete AudioNeuralDataset with ``N`` neurons and ``S`` stims."""
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass

    ds = _FakeAudio(path="/tmp/nowhere", dt_ms=dt)
    ds.F = F
    ds.N_neurons = N
    ds.stim_meta = [{"name": f"s{i}", "type": "synthetic"} for i in range(S)]
    ds.stims = [torch.zeros(1, F, T) for _ in range(S)]
    ds.responses = [[torch.ones(3, T) * (s + 1) for _ in range(N)] for s in range(S)]
    ds.neuron_metadata = [{"uid": f"n{i}", "area": "X"} for i in range(N)]
    ds.validate()
    return ds


def test_concat_two_audio_datasets():
    """AA-style concat: block-diagonal responses, (S1+S2, N1+N2) mask."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    c = concat_neural_datasets([a, b])

    assert c.N_neurons == 5
    assert c.get_S() == 5
    assert len(c.stims) == 5
    assert len(c.stim_meta) == 5
    assert len(c.neuron_metadata) == 5

    # block-diagonal: first 3 stims x first 2 neurons = real; rest NaN
    for s in range(3):
        for n in range(2):
            assert not c.responses[s][n].isnan().any(), f"ds1 real block ({s},{n})"
        for n in range(2, 5):
            assert c.responses[s][n].isnan().all(), f"cross-block ({s},{n})"
    for s in range(3, 5):
        for n in range(0, 2):
            assert c.responses[s][n].isnan().all(), f"cross-block ({s},{n})"
        for n in range(2, 5):
            assert not c.responses[s][n].isnan().any(), f"ds2 real block ({s},{n})"

    # nrn_masks (derived) reflects the block-diagonal structure
    assert tuple(c.nrn_masks.shape) == (5, 5)
    assert c.nrn_masks.dtype == torch.bool
    assert c.nrn_masks[:3, :2].all()
    assert c.nrn_masks[3:, 2:].all()
    assert not c.nrn_masks[:3, 2:].any()
    assert not c.nrn_masks[3:, :2].any()


def test_concat_n_ary():
    """N-ary concat: 3 datasets, check final shape."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    c = _fake_audio_dataset(N=1, S=4)
    out = concat_neural_datasets([a, b, c])

    assert out.N_neurons == 2 + 3 + 1
    assert out.get_S() == 3 + 2 + 4


def test_concat_add_sugar_equivalent():
    """ds1 + ds2 must produce the same N / S as concat_neural_datasets([ds1, ds2])."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    c1 = a + b
    c2 = concat_neural_datasets([a, b])
    assert c1.N_neurons == c2.N_neurons
    assert c1.get_S() == c2.get_S()


def test_concat_dt_mismatch_rejects():
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3, dt=1.0)
    b = _fake_audio_dataset(N=3, S=2, dt=2.0)
    with pytest.raises(AssertionError, match="dt mismatch"):
        concat_neural_datasets([a, b])


def test_concat_F_mismatch_rejects():
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3, F=16)
    b = _fake_audio_dataset(N=3, S=2, F=32)
    with pytest.raises(AssertionError, match="F mismatch"):
        concat_neural_datasets([a, b])


def test_concat_add_returns_notimplemented_for_non_dataset():
    """ds + non_dataset must delegate to Python's default (and eventually TypeError)."""
    a = _fake_audio_dataset(N=2, S=3)
    with pytest.raises(TypeError):
        _ = a + 42
