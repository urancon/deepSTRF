"""CRCNS-AA4 loader keeps silent trials and silent (unit, stimulus) responses as zeros.

Needs the unpacked AA4 archive: ``$AA4_DATA`` (the folder holding the per-bird
folders) or the deepSTRF cache; skips otherwise.
"""
from __future__ import annotations

import glob
import os

import h5py
import pytest
import torch

BIRD = "GreBlu9508M"


def _aa4_path():
    p = os.environ.get("AA4_DATA")
    if p is None:
        from deepSTRF.utils.data_download import default_cache_dir
        p = str(default_cache_dir("AA4"))
    if not os.path.isdir(os.path.join(p, BIRD, "wavfiles")):
        pytest.skip("CRCNS-AA4 data missing (set $AA4_DATA)")
    return p


@pytest.fixture(scope="module")
def ds():
    from deepSTRF.datasets.audio.crcns_aa4 import CRCNSAA4Dataset
    return CRCNSAA4Dataset(path=_aa4_path(), animals=(BIRD,), dt_ms=10.0, smooth=False)


def _filled(ds):
    return [r for row in ds.responses for r in row if not (r.shape == (1, 1) and torch.isnan(r).all())]


def test_every_recorded_trial_is_kept(ds):
    # regression: trials without a post-onset spike were dropped (down to R = 1);
    # every unit x stimulus in this bird has 9-12 trials in the h5 files
    assert min(r.shape[0] for r in _filled(ds)) >= 9


def test_silent_responses_are_zeros_not_missing(ds):
    # regression: a unit that heard a sound and never fired became a NaN "not heard" slot
    assert any(float(r.abs().sum()) == 0 for r in _filled(ds))


def test_sentinel_trials_exist_in_raw_data():
    # the h5 files mark a trial without spikes by a lone -999 spike time
    f = sorted(glob.glob(os.path.join(_aa4_path(), BIRD, "*.h5")))[0]
    with h5py.File(f, "r") as h:
        found = any(
            g[tk]["spike_times"][()].ravel().tolist() == [-999]
            for cls in h if cls not in ("class_info", "extra_info")
            for sk in h[cls] for g in [h[cls][sk]] for tk in g)
    assert found
