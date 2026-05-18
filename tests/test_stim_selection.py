"""Contract tests for the stim-selection API + bidirectional filtering rule.

Covers:
 - select_stim / select_stims / select_stims_by_attr
 - reset_stim_selection
 - select_pop_by_stim_attr (formerly NotImplementedError)
 - bidirectional rule: when S_sel is set, _selected() also auto-hides neurons
   whose only valid responses lie outside the selected stim subset
 - __len__ / __getitem__ honour the intersection of S_sel and the neuron filter

Uses an inline minimal AudioNeuralDataset subclass so tests don't depend on any
real dataset on disk.
"""

import pytest
import torch


def _audio_with_subset_meta(N: int, S_est: int, S_val: int, dt: float = 1.0,
                            F: int = 4, T: int = 8,
                            est_only_neurons: int = 0,
                            val_only_neurons: int = 0):
    """Minimal AudioNeuralDataset where stims have a {'subset': 'est'|'val'} tag.

    Parameters
    ----------
    est_only_neurons, val_only_neurons : int
        Number of neurons (counted from the end) that have NaN-sentinel
        responses for stims of the *other* subset. Lets tests exercise the
        bidirectional rule.
    """
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass

    ds = _FakeAudio(path="/tmp/nowhere", dt_ms=dt)
    ds.F = F
    ds.N_neurons = N

    stim_meta = []
    stims = []
    for i in range(S_est):
        stim_meta.append({"name": f"est{i}", "subset": "est"})
        stims.append(torch.zeros(1, F, T))
    for i in range(S_val):
        stim_meta.append({"name": f"val{i}", "subset": "val"})
        stims.append(torch.zeros(1, F, T))
    ds.stim_meta = stim_meta
    ds.stims = stims

    # responses: real (R, T) tensor by default; NaN sentinel for the (subset, neuron)
    # combinations specified by est_only_neurons / val_only_neurons.
    responses = []
    for s, sm in enumerate(stim_meta):
        row = []
        for n in range(N):
            is_val_only_cell = n >= N - val_only_neurons
            is_est_only_cell = (n >= N - val_only_neurons - est_only_neurons
                                and n < N - val_only_neurons)
            if sm["subset"] == "est" and is_val_only_cell:
                row.append(torch.full((1, 1), float("nan")))
            elif sm["subset"] == "val" and is_est_only_cell:
                row.append(torch.full((1, 1), float("nan")))
            else:
                row.append(torch.ones(2, T) * (s + 1))
        responses.append(row)
    ds.responses = responses

    ds.nrn_meta = [{"uid": f"n{i}"} for i in range(N)]
    ds.validate()
    return ds


def test_default_state_no_stim_filter():
    """Fresh dataset has S_sel == None (no restriction), all stims iterable."""
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    assert ds.S_sel is None
    assert len(ds) == 6
    assert ds._selected_stims() == [0, 1, 2, 3, 4, 5]


def test_select_stim_single():
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    ds.select_stim(2)
    assert ds.S_sel == [2]
    assert len(ds) == 1
    _, _, _, meta = ds[0]
    assert meta["name"] == "est2"


def test_select_stim_rejects_out_of_range():
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    with pytest.raises(AssertionError):
        ds.select_stim(99)
    with pytest.raises(AssertionError):
        ds.select_stim(-1)


def test_select_stims_list():
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    ds.select_stims([1, 3, 5])
    assert ds.S_sel == [1, 3, 5]
    assert len(ds) == 3
    metas = [ds[i][3]["name"] for i in range(len(ds))]
    assert metas == ["est1", "est3", "val1"]


def test_select_stims_by_attr():
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    val_idxs = ds.select_stims_by_attr("subset", "val")
    assert val_idxs == [4, 5]
    assert ds.S_sel == [4, 5]
    assert len(ds) == 2
    for i in range(len(ds)):
        _, _, _, meta = ds[i]
        assert meta["subset"] == "val"


def test_select_stims_by_missing_attr_returns_empty():
    """Explicit zero-stim selection must yield zero-length, not silently disable the filter."""
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    assert ds.select_stims_by_attr("does_not_exist", "anything") == []
    assert ds.S_sel == []
    assert len(ds) == 0


def test_reset_stim_selection():
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    ds.select_stim(1)
    assert len(ds) == 1
    ds.reset_stim_selection()
    assert ds.S_sel is None
    assert len(ds) == 6


def test_bidirectional_hides_neurons_outside_stim_subset():
    """Selecting val stims must auto-hide neurons that have only est data.

    Setup: 5 neurons, last 2 have only est responses (NaN sentinel for val
    stims). Selecting val should keep only the first 3.
    """
    ds = _audio_with_subset_meta(N=5, S_est=4, S_val=2, est_only_neurons=2)
    ds.select_stims_by_attr("subset", "val")
    # _selected returns only neurons with at least one valid val response
    assert ds._selected() == [0, 1, 2]
    # __getitem__ batches contain only those neurons
    _, resps, _, _ = ds[0]
    assert len(resps) == 3
    # none of those responses are NaN sentinels
    for r in resps:
        assert not r.isnan().any()


def test_bidirectional_combines_with_explicit_neuron_selection():
    """Explicit ``self.I`` is intersected with the bidirectional filter."""
    ds = _audio_with_subset_meta(N=5, S_est=4, S_val=2, est_only_neurons=2)
    ds.select_population([2, 3, 4])    # picks 1 val-having + 2 est-only
    ds.select_stims_by_attr("subset", "val")
    # only neuron 2 has both: in user's selection AND val data
    assert ds._selected() == [2]


def test_select_pop_by_stim_attr_now_works():
    """Implements what was raising NotImplementedError before."""
    ds = _audio_with_subset_meta(N=5, S_est=4, S_val=2, val_only_neurons=2)
    selected = ds.select_pop_by_stim_attr("subset", "est")
    # neurons 3 and 4 are val-only, so they must be excluded
    assert selected == [0, 1, 2]
    assert ds.I == [0, 1, 2]


def test_select_pop_by_stim_attr_no_match_returns_empty():
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    assert ds.select_pop_by_stim_attr("subset", "nonexistent") == []
    assert ds.I == []


def test_iter_idx_intersects_S_sel_with_neuron_filter():
    """Both filters compose: S_sel limits stims; neuron mask further trims them."""
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    ds.select_stims([0, 5])             # one est + one val
    # all 3 neurons see both stims, so both stims iterable
    assert len(ds) == 2
    # now hide neuron-0 (still has 2 neurons covering both stims)
    ds.select_population([1, 2])
    assert len(ds) == 2


def test_concat_S_sel_initialized_unrestricted():
    """concat_neural_datasets returns a fresh instance with S_sel == None (no restriction)."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _audio_with_subset_meta(N=2, S_est=2, S_val=1)
    b = _audio_with_subset_meta(N=3, S_est=1, S_val=1)
    c = concat_neural_datasets([a, b])
    assert c.S_sel is None
    # full iteration covers both blocks (5 stims total)
    assert len(c) == 5


def test_concat_select_stims_by_attr_propagates():
    """A single select_stims_by_attr call works across mixed sources."""
    from deepSTRF.utils.data import concat_neural_datasets

    a = _audio_with_subset_meta(N=2, S_est=2, S_val=1)
    b = _audio_with_subset_meta(N=3, S_est=1, S_val=1)
    c = concat_neural_datasets([a, b])
    val_idxs = c.select_stims_by_attr("subset", "val")
    # a has 1 val (idx 2), b has 1 val (idx 4 in the merged list)
    assert val_idxs == [2, 4]
    assert len(c) == 2


# ---------------------------------------------------------------------------
# Predicate variants of the filter API (threshold / range / compound queries)
# ---------------------------------------------------------------------------


def _audio_with_numeric_nrn_meta(snrs, depths=None, areas=None,
                                 S_est: int = 2, S_val: int = 1,
                                 dt: float = 1.0, F: int = 4, T: int = 8):
    """Audio dataset whose ``nrn_meta`` carries numeric + categorical fields.

    Lets tests exercise threshold / range / compound predicates. The N axis
    is implied by ``len(snrs)``; per-neuron metadata is heterogeneous when
    ``depths`` / ``areas`` are partially ``None`` (simulates concatenated
    datasets with missing keys).
    """
    from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset

    class _FakeAudio(AudioNeuralDataset):
        pass

    N = len(snrs)
    ds = _FakeAudio(path="/tmp/nowhere", dt_ms=dt)
    ds.F = F
    ds.N_neurons = N

    stim_meta, stims = [], []
    for i in range(S_est):
        stim_meta.append({"name": f"est{i}", "subset": "est", "duration_s": 1.0 + i})
        stims.append(torch.zeros(1, F, T))
    for i in range(S_val):
        stim_meta.append({"name": f"val{i}", "subset": "val", "duration_s": 2.5 + i})
        stims.append(torch.zeros(1, F, T))
    ds.stim_meta = stim_meta
    ds.stims = stims

    responses = [[torch.ones(2, T) for _ in range(N)] for _ in stim_meta]
    ds.responses = responses

    nrn_meta = []
    for i, snr in enumerate(snrs):
        m = {"uid": f"n{i}", "snr": snr}
        if depths is not None:
            m["depth_um"] = depths[i]
        if areas is not None:
            m["area"] = areas[i]
        nrn_meta.append(m)
    ds.nrn_meta = nrn_meta
    ds.validate()
    return ds


def test_select_pop_by_nrn_predicate_threshold():
    ds = _audio_with_numeric_nrn_meta(snrs=[0.1, 0.4, 0.6, 0.9])
    sel = ds.select_pop_by_nrn_predicate(lambda n: n["snr"] > 0.5)
    assert sel == [2, 3]
    assert ds.I == [2, 3]


def test_select_pop_by_nrn_predicate_range():
    ds = _audio_with_numeric_nrn_meta(
        snrs=[0.5, 0.5, 0.5, 0.5],
        depths=[150, 250, 750, 900],
    )
    sel = ds.select_pop_by_nrn_predicate(
        lambda n: 200 <= n["depth_um"] <= 800
    )
    assert sel == [1, 2]


def test_select_pop_by_nrn_predicate_compound():
    ds = _audio_with_numeric_nrn_meta(
        snrs=[0.2, 0.7, 0.7, 0.9],
        areas=["MLd", "Field_L", "CM", "MLd"],
    )
    sel = ds.select_pop_by_nrn_predicate(
        lambda n: n["snr"] > 0.5 and n["area"] in {"Field_L", "MLd"}
    )
    assert sel == [1, 3]


def test_select_pop_by_nrn_predicate_missing_key_is_skipped():
    """Predicates that raise KeyError on a neuron must skip it, not crash."""
    ds = _audio_with_numeric_nrn_meta(snrs=[0.6, 0.7, 0.8])
    # only neuron 1 has 'area'
    ds.nrn_meta[1]["area"] = "MLd"
    sel = ds.select_pop_by_nrn_predicate(lambda n: n["area"] == "MLd")
    assert sel == [1]


def test_select_pop_by_nrn_predicate_typeerror_is_skipped():
    """Comparison against missing-as-None must skip silently."""
    ds = _audio_with_numeric_nrn_meta(snrs=[0.6, 0.7, 0.8])
    # poison neuron 1's snr with a non-numeric value
    ds.nrn_meta[1]["snr"] = None
    sel = ds.select_pop_by_nrn_predicate(lambda n: n["snr"] > 0.5)
    assert sel == [0, 2]


def test_select_pop_by_nrn_predicate_empty_when_no_match():
    ds = _audio_with_numeric_nrn_meta(snrs=[0.1, 0.2, 0.3])
    sel = ds.select_pop_by_nrn_predicate(lambda n: n["snr"] > 1.0)
    assert sel == []
    assert ds.I == []


def test_select_stims_by_predicate_threshold():
    ds = _audio_with_numeric_nrn_meta(snrs=[0.5, 0.5], S_est=3, S_val=2)
    # est durations: 1.0, 2.0, 3.0; val durations: 2.5, 3.5
    sel = ds.select_stims_by_predicate(lambda s: s["duration_s"] >= 2.5)
    assert sel == [2, 3, 4]
    assert ds.S_sel == [2, 3, 4]


def test_select_stims_by_predicate_compound():
    ds = _audio_with_numeric_nrn_meta(snrs=[0.5, 0.5], S_est=2, S_val=2)
    sel = ds.select_stims_by_predicate(
        lambda s: s["subset"] == "val" and s["duration_s"] < 3.0
    )
    # val stims have durations 2.5, 3.5 -> only the first one (global idx 2)
    assert sel == [2]


def test_select_stims_by_predicate_missing_key_is_skipped():
    ds = _audio_with_numeric_nrn_meta(snrs=[0.5, 0.5])
    sel = ds.select_stims_by_predicate(lambda s: s["does_not_exist"] > 0)
    assert sel == []
    assert ds.S_sel == []


def test_select_pop_by_stim_predicate_keeps_only_covering_neurons():
    """Mirrors test_select_pop_by_stim_attr_now_works using a predicate."""
    ds = _audio_with_subset_meta(N=5, S_est=4, S_val=2, val_only_neurons=2)
    # est stims have duration not set on the subset helper, so use 'subset' attr
    sel = ds.select_pop_by_stim_predicate(lambda s: s["subset"] == "est")
    # neurons 3 and 4 are val-only -> excluded
    assert sel == [0, 1, 2]
    assert ds.I == [0, 1, 2]


def test_select_pop_by_stim_predicate_empty_s_idxs_returns_empty():
    ds = _audio_with_subset_meta(N=3, S_est=4, S_val=2)
    assert ds.select_pop_by_stim_predicate(lambda s: s["subset"] == "nope") == []
    assert ds.I == []
