"""Tests for ``deepSTRF.datasets.audio.wingert2026``.

The structural / data-paradigm checks need the actual Zenodo archive
(`10.5281/zenodo.18331549`) on disk — they're skipped automatically
when the local data dir is missing (typical in CI). Pure-function
checks (cell-id parsing, package re-export, download idempotency) run
unconditionally.

Override the local path with ``$WINGERT2026_DATA`` if needed.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import torch


WINGERT_LOCAL = os.environ.get(
    "WINGERT2026_DATA",
    "/home/ulysse/Documents/NRFdatasets/Audio/Wingert2026/18331549",
)
HAS_DATA = (
    os.path.isdir(os.path.join(WINGERT_LOCAL, "recordings"))
    and os.path.isfile(os.path.join(WINGERT_LOCAL, "cell_list.csv"))
)


# ============================================================
# Pure functions — always run
# ============================================================

def test_parse_wingert_cell_id_3_segment():
    from deepSTRF.datasets.audio._wingert_native import parse_wingert_cell_id
    out = parse_wingert_cell_id("CLT027c-009-1")
    assert out == {"animal": "CLT", "electrode": 9, "unit_in_electrode": 1}


def test_parse_wingert_cell_id_4_segment_slj_a():
    from deepSTRF.datasets.audio._wingert_native import parse_wingert_cell_id
    # SLJ032a's A-probe uses the 4-segment format too.
    out = parse_wingert_cell_id("SLJ032a-A-003-1")
    assert out == {"animal": "SLJ", "electrode": 3, "unit_in_electrode": 1}


def test_parse_wingert_cell_id_4_segment_slj_b():
    from deepSTRF.datasets.audio._wingert_native import parse_wingert_cell_id
    out = parse_wingert_cell_id("SLJ032a-B-154-1")
    assert out == {"animal": "SLJ", "electrode": 154, "unit_in_electrode": 1}


def test_parse_wingert_cell_id_unparseable():
    from deepSTRF.datasets.audio._wingert_native import parse_wingert_cell_id
    bad = parse_wingert_cell_id("not-a-real-cell")
    assert bad == {"animal": None, "electrode": None, "unit_in_electrode": None}
    assert parse_wingert_cell_id(None) == {
        "animal": None, "electrode": None, "unit_in_electrode": None,
    }


def test_rasterize_spike_times_floor_convention():
    """Spike at t=0.005 with fs=100 lands in bin 0 (floor of 0.5)."""
    from deepSTRF.datasets.audio._wingert_native import rasterize_spike_times
    spikes_s = np.array([0.005, 0.015, 0.005, 0.999, 1.000, 1.500])
    counts = rasterize_spike_times(spikes_s, T=100, fs=100)
    # 0.005 -> floor(0.5) = bin 0 (2 spikes in same bin)
    # 0.015 -> floor(1.5) = bin 1
    # 0.999 -> floor(99.9) = bin 99
    # 1.000 -> floor(100) = bin 100 → out of range (dropped)
    # 1.500 -> bin 150 → out of range (dropped)
    assert counts[0] == 2.0
    assert counts[1] == 1.0
    assert counts[99] == 1.0
    assert counts.sum() == 4.0  # 2 spikes dropped past T


def test_rasterize_empty_input():
    from deepSTRF.datasets.audio._wingert_native import rasterize_spike_times
    counts = rasterize_spike_times(np.array([]), T=10, fs=100)
    assert counts.shape == (10,)
    assert counts.sum() == 0


def test_re_exported_from_package():
    from deepSTRF.datasets.audio import (
        Wingert2026Dataset, download_wingert2026,
    )
    assert Wingert2026Dataset is not None
    assert callable(download_wingert2026)


def test_download_skips_when_already_present(tmp_path):
    """download_wingert2026 is a no-op when recordings/ has ≥60 .tgz."""
    from deepSTRF.datasets.audio import download_wingert2026
    (tmp_path / "recordings").mkdir()
    for i in range(60):
        (tmp_path / "recordings" / f"site{i:02d}_fake.tgz").write_text("")
    (tmp_path / "cell_list.csv").write_text("cellid,siteid,area\n")
    # Should return the path without trying to fetch anything.
    out = download_wingert2026(dest=str(tmp_path))
    assert out == str(tmp_path)


# ============================================================
# Data-dependent integration tests
# ============================================================

skip_if_no_data = pytest.mark.skipif(
    not HAS_DATA,
    reason=(
        f"Wingert2026 data not at {WINGERT_LOCAL!r}; set $WINGERT2026_DATA or "
        f"call Wingert2026Dataset(download=True)."
    ),
)


# ---- enumerate-only (cheap; opens only cell_list.csv) ----

@skip_if_no_data
def test_enumerate_only_area_A1():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, area="A1", _enumerate_only=True)
    assert ds.N_neurons == 2128


@skip_if_no_data
def test_enumerate_only_area_PEG():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, area="PEG", _enumerate_only=True)
    assert ds.N_neurons == 746


@skip_if_no_data
def test_enumerate_only_area_AC():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, area="AC", _enumerate_only=True)
    assert ds.N_neurons == 217


@skip_if_no_data
def test_enumerate_only_default_excludes_unlabeled():
    """Default area=None drops the 131 area=NaN cells; N=3128, not 3259."""
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, area=None, _enumerate_only=True)
    assert ds.N_neurons == 3128


@skip_if_no_data
def test_enumerate_only_include_unlabeled():
    """include_unlabeled=True adds the 131 NaN-area cells from PRN0{10,11,20}b."""
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(
        path=WINGERT_LOCAL, area=None, _enumerate_only=True, include_unlabeled=True,
    )
    assert ds.N_neurons == 3259
    # The unlabeled cells have area=None.
    unlabeled = [n for n in ds.nrn_meta if n["area"] is None]
    assert len(unlabeled) == 131
    assert all(n["session"] in {"PRN010b", "PRN011b", "PRN020b"} for n in unlabeled)


@skip_if_no_data
def test_enumerate_only_area_list_intersection():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(
        path=WINGERT_LOCAL, area=["A1", "PEG"], _enumerate_only=True,
    )
    assert ds.N_neurons == 2128 + 746


@skip_if_no_data
def test_enumerate_only_slj032a_two_probes():
    """SLJ032a probe-A and probe-B are addressable as separate sites."""
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds_a = Wingert2026Dataset(path=WINGERT_LOCAL, site="SLJ032a", _enumerate_only=True)
    ds_b = Wingert2026Dataset(path=WINGERT_LOCAL, site="SLJ032a-B", _enumerate_only=True)
    assert ds_a.N_neurons == 76
    assert ds_b.N_neurons == 47
    # All A-probe cells have the 4-segment cell id pattern.
    assert all(meta["cell_id"].startswith("SLJ032a-A-") for meta in ds_a.nrn_meta)
    assert all(meta["cell_id"].startswith("SLJ032a-B-") for meta in ds_b.nrn_meta)


@skip_if_no_data
def test_enumerate_only_typo_site_raises():
    """Filter typo surfaces an AssertionError with the offending name."""
    from deepSTRF.datasets.audio import Wingert2026Dataset
    with pytest.raises(AssertionError, match="DOES_NOT_EXIST"):
        Wingert2026Dataset(
            path=WINGERT_LOCAL, site="DOES_NOT_EXIST", _enumerate_only=True,
        )


@skip_if_no_data
def test_enumerate_only_bad_area_raises():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    with pytest.raises(AssertionError, match="V1"):
        Wingert2026Dataset(path=WINGERT_LOCAL, area="V1", _enumerate_only=True)


# ---- nrn_meta shape ----

@skip_if_no_data
def test_nrn_meta_keys_for_labeled_cell():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, site="CLT027c", _enumerate_only=True)
    meta = ds.nrn_meta[0]
    expected_keys = {
        "cell_id", "site", "session", "area", "layer", "depth",
        "narrow", "celltype", "sw", "goodpred",
        "animal", "electrode", "unit_in_electrode",
    }
    assert set(meta.keys()) == expected_keys
    assert meta["session"] == "CLT027c"
    assert meta["site"] == "CLT027c"
    assert meta["area"] == "A1"
    assert isinstance(meta["goodpred"], bool)


# ---- full single-site load ----

@pytest.fixture(scope="module")
def clt027c_dataset():
    """Cached single-site load; CLT027c is the smallest (~20 cells, 11 stims, 7s)."""
    if not HAS_DATA:
        pytest.skip("Wingert2026 data not available")
    from deepSTRF.datasets.audio import Wingert2026Dataset
    return Wingert2026Dataset(path=WINGERT_LOCAL, site="CLT027c")


@skip_if_no_data
def test_single_site_basic_shape(clt027c_dataset):
    ds = clt027c_dataset
    assert ds.F == 32
    assert ds.N_neurons == 20
    assert len(ds.stims) == len(ds.stim_meta) == len(ds.responses) == 11
    # 22s cohort: T=2200
    assert all(tuple(s.shape) == (1, 32, 2200) for s in ds.stims)


@skip_if_no_data
def test_single_site_stim_meta_subsets(clt027c_dataset):
    """val = STIM_00*, est = STIM_seq* — file-name prefix, not R count."""
    ds = clt027c_dataset
    for m in ds.stim_meta:
        expected = "val" if m["name"].startswith("STIM_00") else "est"
        assert m["subset"] == expected
    # CLT027c happens to have 2 test stims (R=1, R=2) + 9 est stims.
    assert sum(1 for m in ds.stim_meta if m["subset"] == "val") == 2
    assert sum(1 for m in ds.stim_meta if m["subset"] == "est") == 9


@skip_if_no_data
def test_single_site_no_sentinels(clt027c_dataset):
    """Single-site load: every (stim, cell) pair has real data."""
    ds = clt027c_dataset
    sentinels = [t for row in ds.responses for t in row if t.numel() == 1]
    assert len(sentinels) == 0


@skip_if_no_data
def test_normalization_yields_unit_range(clt027c_dataset):
    ds = clt027c_dataset
    s_min = min(s.min().item() for s in ds.stims)
    s_max = max(s.max().item() for s in ds.stims)
    assert s_min == pytest.approx(0.0, abs=1e-6)
    assert s_max == pytest.approx(1.0, abs=1e-6)
    real = [t for row in ds.responses for t in row if t.numel() > 1]
    r_min = min(t.min().item() for t in real)
    r_max = max(t.max().item() for t in real)
    assert r_min == pytest.approx(0.0, abs=1e-6)
    assert r_max == pytest.approx(1.0, abs=1e-6)


@skip_if_no_data
def test_subset_est_filter():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, site="CLT027c", subset="est")
    assert all(m["subset"] == "est" for m in ds.stim_meta)
    assert all(not m["name"].startswith("STIM_00") for m in ds.stim_meta)


@skip_if_no_data
def test_subset_val_filter():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, site="CLT027c", subset="val")
    assert all(m["subset"] == "val" for m in ds.stim_meta)
    assert all(m["name"].startswith("STIM_00") for m in ds.stim_meta)


@skip_if_no_data
def test_smooth_does_not_change_shape():
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(path=WINGERT_LOCAL, site="CLT027c", smooth=True)
    assert tuple(ds.responses[0][0].shape) == (1, 2200)


# ---- multi-site block-diagonal load ----

@pytest.fixture(scope="module")
def two_site_dataset():
    """CLT027c (A1) + CLT028c (A1) — exercises the cross-session sentinel path."""
    if not HAS_DATA:
        pytest.skip("Wingert2026 data not available")
    from deepSTRF.datasets.audio import Wingert2026Dataset
    return Wingert2026Dataset(path=WINGERT_LOCAL, site=["CLT027c", "CLT028c"])


@skip_if_no_data
def test_two_site_block_diagonal(two_site_dataset):
    ds = two_site_dataset
    by_session = {}
    for n_idx, meta in enumerate(ds.nrn_meta):
        by_session.setdefault(meta["session"], []).append(n_idx)
    # For each session's stim entries, only same-session cells should be real;
    # all other-session cells should be NaN sentinels.
    for s_idx, smeta in enumerate(ds.stim_meta):
        same = by_session[smeta["session"]]
        other = [n for n in range(ds.N_neurons) if n not in same]
        for n in same:
            assert ds.responses[s_idx][n].numel() > 1, (
                f"expected real data at s={s_idx},n={n} (same session)"
            )
        for n in other:
            assert ds.responses[s_idx][n].numel() == 1, (
                f"expected NaN sentinel at s={s_idx},n={n} (other session)"
            )


@skip_if_no_data
def test_sentinels_share_one_reference(two_site_dataset):
    """Memory-regression guard: all (1,1) NaN sentinels are the SAME object."""
    ds = two_site_dataset
    sentinel_ids = {
        id(t) for row in ds.responses for t in row if t.numel() == 1
    }
    assert len(sentinel_ids) == 1


@skip_if_no_data
def test_slj032a_shared_session_stims():
    """Both probes load from the same .tgz; no stim duplication."""
    from deepSTRF.datasets.audio import Wingert2026Dataset
    ds = Wingert2026Dataset(
        path=WINGERT_LOCAL, site=["SLJ032a", "SLJ032a-B"],
    )
    assert ds.N_neurons == 76 + 47
    # One session => one stim entry per stim name, not two.
    sessions = {m["session"] for m in ds.stim_meta}
    assert sessions == {"SLJ032a"}


# ---- DataLoader / collate roundtrip ----

@skip_if_no_data
def test_collate_roundtrip(clt027c_dataset):
    from deepSTRF.utils import neural_collate
    from torch.utils.data import DataLoader
    ds = clt027c_dataset
    ld = DataLoader(ds, batch_size=4, collate_fn=neural_collate)
    stims, responses, valid_mask, metas = next(iter(ld))
    B = 4
    assert stims.shape == (B, 1, 32, 2200)
    assert responses.shape[0] == B
    assert responses.shape[1] == ds.N_neurons
    assert responses.shape[3] == 2200
    assert valid_mask.shape == responses.shape
    assert len(metas) == B
