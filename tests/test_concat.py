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


def test_concat_getitem_only_yields_iterable_stims_under_selection():
    """Selecting only source A's neurons must hide source B's stims from __len__/__getitem__.

    This is the property that makes concatenated datasets safe with DataLoader:
    iterating range(len(ds)) must never produce a stim whose responses are
    fully NaN under the current neuron selection.
    """
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    c = concat_neural_datasets([a, b])

    # all 5 stims iterable when all neurons selected (default after validate)
    assert len(c) == 5

    # select only A's neurons -> only A's stims iterable
    c.select_population([0, 1])
    assert len(c) == 3
    seen_metas = [c[i][3] for i in range(len(c))]
    assert seen_metas == c.stim_meta[:3]

    # every yielded item must have at least one non-NaN response
    for i in range(len(c)):
        _, resps, _, _ = c[i]
        assert any(not r.isnan().any() for r in resps), \
            f"item {i} has fully-NaN responses under selection {c.I}"

    # accessing beyond the iterable range must raise, not silently return a NaN slab
    with pytest.raises(IndexError):
        _ = c[len(c)]

    # select only B's neurons -> only B's stims iterable
    c.select_population([2, 3, 4])
    assert len(c) == 2
    seen_metas = [c[i][3] for i in range(len(c))]
    assert seen_metas == c.stim_meta[3:]
    for i in range(len(c)):
        _, resps, _, _ = c[i]
        assert any(not r.isnan().any() for r in resps)


def test_dataloader_over_concat_skips_cross_block_stims():
    """Sanity check that a real DataLoader over a concat dataset skips cross-block stims."""
    from torch.utils.data import DataLoader
    from deepSTRF.utils.data import concat_neural_datasets, neural_collate

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    c = concat_neural_datasets([a, b])
    c.select_population([0, 1])  # only A's neurons

    loader = DataLoader(c, batch_size=2, shuffle=False, collate_fn=neural_collate)
    batches = list(loader)

    # 3 iterable stims, batch_size 2 -> 2 batches (sizes 2 and 1)
    assert len(batches) == 2
    total_items = sum(b_[0].shape[0] for b_ in batches)
    assert total_items == 3

    # every batch must contain only A's stims (i.e. valid_mask has at least one True per item)
    for stims, responses, valid_mask, metas in batches:
        per_item_has_data = valid_mask.any(dim=(1, 2, 3))  # (B,) bool
        assert per_item_has_data.all(), \
            "DataLoader yielded a batch item with no valid (s,n,r,t) data anywhere"


def test_concat_tags_metadata_with_default_class_name():
    """Default provenance: every output meta dict gets ``dataset = type(source).__name__``.

    Tag is written into both ``stim_meta`` and ``neuron_metadata`` so
    post-hoc filtering works on either axis.
    """
    from deepSTRF.utils.data import concat_neural_datasets
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    # build two subclasses with distinct class names so the default tag distinguishes them
    class _A(AudioNeuralDataset):
        pass

    class _B(AudioNeuralDataset):
        pass

    def make(cls, N, S, F=4, T=8, dt=1.0):
        ds = cls(path="/tmp/nowhere", dt_ms=dt)
        ds.F = F
        ds.N_neurons = N
        ds.stim_meta = [{"name": f"s{i}", "type": "synthetic"} for i in range(S)]
        ds.stims = [torch.zeros(1, F, T) for _ in range(S)]
        ds.responses = [[torch.ones(3, T) * (s + 1) for _ in range(N)] for s in range(S)]
        ds.neuron_metadata = [{"uid": f"n{i}", "area": "X"} for i in range(N)]
        ds.validate()
        return ds

    a = make(_A, N=2, S=3)
    b = make(_B, N=3, S=2)
    c = concat_neural_datasets([a, b])

    # every output stim_meta tagged with its source class name
    assert [m["dataset"] for m in c.stim_meta] == ["_A"] * 3 + ["_B"] * 2
    # every output neuron_metadata tagged the same way
    assert [m["dataset"] for m in c.neuron_metadata] == ["_A"] * 2 + ["_B"] * 3


def test_concat_tags_metadata_with_explicit_names():
    """Explicit names override the default class-name tag."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    c = concat_neural_datasets([a, b], names=["src_a", "src_b"])

    assert [m["dataset"] for m in c.stim_meta] == ["src_a"] * 3 + ["src_b"] * 2
    assert [m["dataset"] for m in c.neuron_metadata] == ["src_a"] * 2 + ["src_b"] * 3


def test_concat_does_not_mutate_input_metadata():
    """Tagging the output must not write the ``dataset`` key into the source dicts."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    # snapshot input dicts (by identity) before concat
    a_stim_dicts = list(a.stim_meta)
    a_nrn_dicts = list(a.neuron_metadata)
    b_stim_dicts = list(b.stim_meta)
    b_nrn_dicts = list(b.neuron_metadata)

    _ = concat_neural_datasets([a, b], names=["A", "B"])

    for m in a_stim_dicts + a_nrn_dicts + b_stim_dicts + b_nrn_dicts:
        assert "dataset" not in m, \
            f"concat mutated an input metadata dict in place: {m!r}"


def test_concat_names_length_must_match():
    """names= must have one entry per source dataset."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    with pytest.raises(AssertionError, match="names must have one entry per dataset"):
        concat_neural_datasets([a, b], names=["only_one"])


def test_concat_dataset_tag_enables_source_filtering():
    """The ``dataset`` tag must be selectable via the existing per-attr filters."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=3)
    b = _fake_audio_dataset(N=3, S=2)
    c = concat_neural_datasets([a, b], names=["src_a", "src_b"])

    # neuron-axis: pick only src_b neurons
    sel = c.select_pop_by_nrn_attr("dataset", "src_b")
    assert sel == [2, 3, 4], f"expected b's neurons (indices 2..4), got {sel}"

    # stim-axis: pick only src_a stims
    c.reset_pop_selection()
    s_sel = c.select_stims_by_attr("dataset", "src_a")
    assert s_sel == [0, 1, 2], f"expected a's stims (indices 0..2), got {s_sel}"


def test_concat_dataset_tag_overrides_existing_key():
    """If a source already has a ``dataset`` key, concat overwrites it."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=2, S=2)
    b = _fake_audio_dataset(N=2, S=2)
    # pre-tag a's metadata as if it had been concat'd before
    for m in a.stim_meta:
        m["dataset"] = "inner_a"
    for m in a.neuron_metadata:
        m["dataset"] = "inner_a"

    c = concat_neural_datasets([a, b], names=["outer_a", "outer_b"])
    # outer tag wins on the output
    assert [m["dataset"] for m in c.stim_meta] == ["outer_a"] * 2 + ["outer_b"] * 2
    assert [m["dataset"] for m in c.neuron_metadata] == ["outer_a"] * 2 + ["outer_b"] * 2
    # source dicts still carry the inner tag, untouched
    assert all(m["dataset"] == "inner_a" for m in a.stim_meta)
    assert all(m["dataset"] == "inner_a" for m in a.neuron_metadata)


def test_select_pop_by_nrn_attr_tolerates_mixed_schemas():
    """Concatenating sources with different ``neuron_metadata`` keys must not break filtering.

    Real-world case: AA1 has ``area`` but AA4 doesn't. ``select_pop_by_nrn_attr("area",
    "Field_L")`` on the concatenation should pick only AA1's matching neurons and
    silently skip AA4's neurons (which lack the key) — not raise ``KeyError``.
    """
    from deepSTRF.utils.data import concat_neural_datasets

    a = _fake_audio_dataset(N=4, S=2)
    b = _fake_audio_dataset(N=3, S=2)
    # tag a's neurons with an extra attribute that b doesn't have
    for n, md in enumerate(a.neuron_metadata):
        md["region"] = "Field_L" if n < 2 else "MLd"
    # b has no 'region' key

    c = concat_neural_datasets([a, b])
    selected = c.select_pop_by_nrn_attr("region", "Field_L")
    assert selected == [0, 1], \
        f"only a's first two neurons match region=Field_L, got {selected}"

    # filter by a key that NO neuron has -> empty result, no KeyError
    selected_none = c.select_pop_by_nrn_attr("nonexistent_key", "anything")
    assert selected_none == []
