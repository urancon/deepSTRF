"""Tests for ``deepSTRF.datasets.audio.downer2025``.

The structural / data-paradigm checks need the actual Zenodo archive
(`10.5281/zenodo.16175377`) on disk — they're skipped automatically when
the local data dir is missing (typical in CI). The filename parser and
the high-rep stim-ID discovery helpers are exercised in isolation.

Override the local path with ``$DOWNER2025_DATA`` if needed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


DOWNER_LOCAL = os.environ.get(
    "DOWNER2025_DATA",
    "/home/ulysse/Documents/NRFdatasets/Audio/Downer2025/auditory_cortex_data",
)
HAS_DATA = (
    os.path.isdir(os.path.join(DOWNER_LOCAL, "sessions"))
    and os.path.isdir(os.path.join(DOWNER_LOCAL, "stimuli"))
    and os.path.isfile(os.path.join(DOWNER_LOCAL, "sessions_metadata.yml"))
)


# ============================================================
# Filename parsing (network-free, pure function)
# ============================================================

def test_parse_channel_filename_plain():
    from deepSTRF.datasets.audio.downer2025 import _parse_channel_filename
    assert _parse_channel_filename("b_180413_Ch49_MUspk.mat") == ("b", "180413", 49, "")


def test_parse_channel_filename_p_suffix():
    from deepSTRF.datasets.audio.downer2025 import _parse_channel_filename
    assert _parse_channel_filename("b_180501_Ch5p_MUspk.mat") == ("b", "180501", 5, "p")


def test_parse_channel_filename_s2_suffix():
    from deepSTRF.datasets.audio.downer2025 import _parse_channel_filename
    assert _parse_channel_filename("c_190604_Ch10s2_MUspk.mat") == ("c", "190604", 10, "s2")


def test_parse_channel_filename_ps2_suffix():
    from deepSTRF.datasets.audio.downer2025 import _parse_channel_filename
    assert _parse_channel_filename("c_180627_Ch1ps2_MUspk.mat") == ("c", "180627", 1, "ps2")


def test_parse_channel_filename_skips_trialinfo():
    from deepSTRF.datasets.audio.downer2025 import _parse_channel_filename
    assert _parse_channel_filename("b_180413_TRIALINFO.mat") is None


def test_parse_channel_filename_unrecognised_returns_none():
    from deepSTRF.datasets.audio.downer2025 import _parse_channel_filename
    assert _parse_channel_filename("readme.txt") is None
    assert _parse_channel_filename("b_180413_NotAChan_MUspk.mat") is None


# ============================================================
# YAML metadata parsing (needs the YAML file)
# ============================================================

@pytest.fixture(scope="module")
def sess_meta():
    if not HAS_DATA:
        pytest.skip("Downer2025 data dir missing — skip integration test")
    from deepSTRF.datasets.audio.downer2025 import _parse_sessions_metadata
    return _parse_sessions_metadata(DOWNER_LOCAL)


def test_sessions_metadata_basic_fields(sess_meta):
    assert sess_meta["n_reps_canonical"] == {"timit": 11, "mVocs": 15}
    # All sessions should map to one of {b, c, f}
    assert set(sess_meta["animal"].values()) <= {"b", "c", "f"}
    # Hemisphere values are {RH, LH}
    assert set(sess_meta["hemisphere"].values()) <= {"RH", "LH"}
    # Area group values are {core, non-primary}
    assert set(sess_meta["area_group"].values()) <= {"core", "non-primary"}


def test_sessions_metadata_bad_sessions_absent_from_disk(sess_meta):
    # Per the README, bad sessions are excluded from the Zenodo release.
    disk = set(os.listdir(os.path.join(DOWNER_LOCAL, "sessions")))
    assert disk.isdisjoint(sess_meta["bad"]), \
        f"bad sessions on disk: {sorted(disk & sess_meta['bad'])}"


# ============================================================
# Enumerate-only mode (~1 s)
# ============================================================

@pytest.fixture(scope="module")
def ds_enum():
    if not HAS_DATA:
        pytest.skip("Downer2025 data dir missing — skip integration test")
    from deepSTRF.datasets.audio import Downer2025Dataset
    return Downer2025Dataset(path=DOWNER_LOCAL, _enumerate_only=True)


def test_enumerate_total_neurons_matches_paper(ds_enum):
    # Ahmed 2025 p5: "a total of 1718" multi-units.
    assert ds_enum.N_neurons == 1718
    assert len(ds_enum.nrn_meta) == 1718


def test_enumerate_cell_ids_unique(ds_enum):
    # Suffix variants ('p', 's2', 'ps2') must keep cell_ids distinct.
    cell_ids = [n["cell_id"] for n in ds_enum.nrn_meta]
    assert len(set(cell_ids)) == 1718


def test_enumerate_area_split_matches_yaml(ds_enum):
    n_core = sum(1 for n in ds_enum.nrn_meta if n["area_group"] == "core")
    n_np = sum(1 for n in ds_enum.nrn_meta if n["area_group"] == "non-primary")
    assert n_core + n_np == 1718
    # Spot-check the values found during the Phase-1 audit.
    assert n_core == 909
    assert n_np == 809


def test_enumerate_metadata_keys(ds_enum):
    required = {"cell_id", "session_id", "animal_id", "hemisphere",
                 "area_group", "area", "channel", "channel_suffix",
                 "n_channels_in_session", "coord_x", "coord_y", "recording_type"}
    for n in ds_enum.nrn_meta:
        assert required <= set(n.keys()), \
            f"missing keys: {required - set(n.keys())}"
        assert n["recording_type"] == "multi-unit"


# ============================================================
# Filter API
# ============================================================

def test_areas_primary_is_alias_for_core():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    a = Downer2025Dataset(path=DOWNER_LOCAL, areas=("core",), _enumerate_only=True)
    b = Downer2025Dataset(path=DOWNER_LOCAL, areas=("primary",), _enumerate_only=True)
    assert a.N_neurons == b.N_neurons == 909


def test_animals_filter():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    ds = Downer2025Dataset(path=DOWNER_LOCAL, animals=("b",), _enumerate_only=True)
    assert all(n["animal_id"] == "b" for n in ds.nrn_meta)
    assert ds.N_neurons == 212


def test_fine_area_filter():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    ds = Downer2025Dataset(path=DOWNER_LOCAL, areas=("A1",), _enumerate_only=True)
    assert all(n["area"] == "A1" for n in ds.nrn_meta)
    assert ds.N_neurons == 813


# ============================================================
# TIMIT end-to-end on one session (~15 s)
# ============================================================

@pytest.fixture(scope="module")
def ds_timit_one_session():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    return Downer2025Dataset(path=DOWNER_LOCAL, stimuli="timit",
                              sessions=["180413"], dt_ms=5.0, smooth=False)


def test_timit_shape_invariants(ds_timit_one_session):
    ds = ds_timit_one_session
    assert ds.F == 32
    assert ds.audio_fs == 16000 and ds.fmax == 8000
    assert ds.species == "squirrel monkey"
    assert ds.N_neurons == 16  # session 180413 has 16 channels
    assert ds.N_neurons == len(ds.nrn_meta)
    S = len(ds.stim_meta)
    assert S == 499  # full TIMIT bank
    assert len(ds.stims) == S
    assert len(ds.responses) == S
    for s in range(3):
        stim = ds.stims[s]
        assert stim.ndim == 3 and stim.shape[0] == 1 and stim.shape[1] == ds.F
        assert not stim.isnan().any(), "stim tensors must never contain NaN"
        for n in range(ds.N_neurons):
            r = ds.responses[s][n]
            assert r.ndim == 2
            if r.isnan().any():
                assert tuple(r.shape) == (1, 1), "NaN-sentinel must be (1, 1)"


def test_timit_per_stim_T_alignment(ds_timit_one_session):
    """For non-sentinel cells, response T must match the stim's T."""
    ds = ds_timit_one_session
    for s in range(min(20, len(ds.stims))):
        T_stim = ds.stims[s].shape[-1]
        for n in range(ds.N_neurons):
            r = ds.responses[s][n]
            if tuple(r.shape) == (1, 1):
                continue
            assert r.shape[-1] == T_stim, \
                f"T mismatch at (s={s},n={n}): stim T={T_stim}, resp T={r.shape[-1]}"


def test_timit_canonical_split_matches_paper(ds_timit_one_session):
    """The 10 11-rep TIMIT IDs are the canonical test set."""
    ds = ds_timit_one_session
    test_ids = sorted(m["stim_id"] for m in ds.stim_meta if m["split"] == "test")
    assert test_ids == [12, 13, 32, 43, 56, 163, 212, 218, 287, 308]
    n_est = sum(1 for m in ds.stim_meta if m["split"] == "estimation")
    assert n_est == 489


def test_timit_subset_filter():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    ds_test = Downer2025Dataset(path=DOWNER_LOCAL, stimuli="timit",
                                  sessions=["180413"], subset="test", smooth=False)
    assert len(ds_test.stim_meta) == 10
    assert all(m["split"] == "test" for m in ds_test.stim_meta)


# ============================================================
# mVocs end-to-end on one session (~10 s)
# ============================================================

@pytest.fixture(scope="module")
def ds_mvocs_one_session():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    return Downer2025Dataset(path=DOWNER_LOCAL, stimuli="mvocs",
                              sessions=["180413"], dt_ms=5.0, smooth=False)


def test_mvocs_shape_invariants(ds_mvocs_one_session):
    ds = ds_mvocs_one_session
    assert ds.F == 32 and ds.audio_fs == 16000 and ds.fmax == 8000
    assert ds.N_neurons == 16
    assert len(ds.stim_meta) == 303
    # NaN-sentinel discipline
    for s in range(min(5, len(ds.stims))):
        assert not ds.stims[s].isnan().any()


def test_mvocs_canonical_test_set_matches_paper(ds_mvocs_one_session):
    """The 11 IDs at exactly 15 reps in the WAV are the canonical test set."""
    ds = ds_mvocs_one_session
    test_ids = sorted(m["stim_id"] for m in ds.stim_meta if m["split"] == "test")
    assert test_ids == [7, 9, 12, 15, 24, 29, 30, 33, 44, 45, 48]


def test_mvocs_n_reps_in_wav_present(ds_mvocs_one_session):
    """mVoc stim_meta surfaces the per-voc canonical rep count from the WAV."""
    ds = ds_mvocs_one_session
    for m in ds.stim_meta:
        assert "n_reps_in_wav" in m
        assert m["n_reps_in_wav"] >= 1
    # Voc with the most reps in the WAV (id with 30 reps).
    max_reps = max(m["n_reps_in_wav"] for m in ds.stim_meta)
    assert max_reps == 30


# ============================================================
# Collate produces well-formed batches
# ============================================================

def test_timit_collate_produces_correct_shapes(ds_timit_one_session):
    from torch.utils.data import DataLoader
    from deepSTRF.utils.data import neural_collate
    loader = DataLoader(ds_timit_one_session, batch_size=2, shuffle=False,
                        collate_fn=neural_collate)
    stims, resps, mask, metas = next(iter(loader))
    assert stims.shape[:3] == (2, 1, ds_timit_one_session.F)
    assert resps.shape[:2] == (2, ds_timit_one_session.N_neurons)
    assert mask.shape == resps.shape
    assert not stims.isnan().any(), "stims must be NaN-free"
    assert len(metas) == 2


# ============================================================
# Cross-mode concatenation (same F, dt, audio_fs, fmax)
# ============================================================

def test_timit_mvocs_concat_works():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    from deepSTRF.utils.data import concat_neural_datasets
    timit = Downer2025Dataset(path=DOWNER_LOCAL, stimuli="timit",
                                sessions=["180413"], smooth=False)
    mvocs = Downer2025Dataset(path=DOWNER_LOCAL, stimuli="mvocs",
                                sessions=["180413"], smooth=False)
    both = concat_neural_datasets([timit, mvocs])
    assert len(both.stim_meta) == 499 + 303
    # Same recording channels in each, so concat doubles N along the stim axis,
    # not the neuron axis -- shared neurons keep their identities.
    # (concat semantics: stim+neuron axes both grow; cross blocks are NaN.)


# ============================================================
# Paper-tuning smoke (n_resamples small for speed)
# ============================================================

def test_compute_paper_tuning_runs(ds_timit_one_session):
    ds = ds_timit_one_session
    summary = ds.compute_paper_tuning(n_resamples=200, verbose=False)
    assert summary["stimuli"] == "timit"
    assert summary["n_with_data"] == ds.N_neurons
    # On a real session we expect some neurons to be flagged as tuned
    # (even with only 200 resamples the strongly-tuned ones still cross
    # p<0.05). Just sanity-check the booleans were written.
    for n in ds.nrn_meta:
        assert isinstance(n["ahmed2025_timit_tuned"], bool)
        assert isinstance(n["ahmed2025_timit_well_tuned"], bool)
        assert "ahmed2025_timit_p_wilcoxon" in n
        assert "ahmed2025_timit_delta_normalized" in n


def test_compute_paper_tuning_rejects_non_multiple_dt(ds_timit_one_session):
    """dt_ms_analysis must be an integer multiple of self.dt."""
    ds = ds_timit_one_session
    with pytest.raises(ValueError, match="integer multiple"):
        ds.compute_paper_tuning(dt_ms_analysis=7.5, n_resamples=10, verbose=False)


def test_compute_paper_tuning_errors_when_no_test_stims():
    if not HAS_DATA:
        pytest.skip("data missing")
    from deepSTRF.datasets.audio import Downer2025Dataset
    ds_est = Downer2025Dataset(path=DOWNER_LOCAL, stimuli="timit",
                                 sessions=["180413"], subset="estimation",
                                 smooth=False)
    with pytest.raises(RuntimeError, match="No test-split stims"):
        ds_est.compute_paper_tuning(n_resamples=10, verbose=False)


# ============================================================
# Zenodo download helper (network-free unit tests)
# ============================================================

def test_download_downer2025_skips_when_extracted_exists(tmp_path):
    """download_downer2025 is idempotent: if the unzipped layout already
    exists (sessions/ subdir present), it returns the path without
    re-downloading or re-unzipping."""
    from deepSTRF.datasets.audio import download_downer2025

    fake_extracted = tmp_path / "auditory_cortex_data"
    (fake_extracted / "sessions").mkdir(parents=True)
    out = download_downer2025(dest=str(tmp_path))
    assert out == str(fake_extracted)


def test_download_downer2025_reexported_from_package():
    """Re-export check: the helper is accessible via deepSTRF.datasets.audio."""
    from deepSTRF.datasets.audio import download_downer2025  # noqa: F401


def test_missing_path_error_message_mentions_download():
    """The FileNotFoundError when path doesn't exist tells the user how
    to obtain the data."""
    from deepSTRF.datasets.audio import Downer2025Dataset
    with pytest.raises(FileNotFoundError, match="download=True"):
        Downer2025Dataset(path="/definitely/does/not/exist",
                            _enumerate_only=True)
