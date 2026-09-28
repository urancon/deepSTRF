"""Tests for ``CRCNSAA5Dataset``.

Synthetic tests reuse the fake release site from ``test_crcns_aa5_prepare``
(runs in CI). The real-data test runs when ``$AA5_DATA`` points to a slim
cache written by ``prepare_aa5``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from deepSTRF.datasets.audio import CRCNSAA5Dataset, prepare_aa5
from deepSTRF.datasets.audio.crcns_aa5 import STIM_CLASSES, _stim_info
from tests.test_crcns_aa5_prepare import PLAYBACKS, SITE, UNITS, _spikes, raw_site  # noqa: F401

CALL = "BlaBla0506_MAF_Te_8-1-9_fs25k.wav"
SONG = "stim3.wav"


@pytest.fixture(scope="module")
def cache(raw_site, tmp_path_factory) -> Path:  # noqa: F811
    dest = tmp_path_factory.mktemp("aa5_ds_cache")
    prepare_aa5(raw_site, dest, progress=False)
    return dest


def _pair(ds, stim_name, cell_suffix):
    s = next(i for i, m in enumerate(ds.stim_meta) if m["name"] == stim_name)
    n = next(i for i, m in enumerate(ds.nrn_meta) if m["cell_id"].endswith(cell_suffix))
    return s, n


def test_shapes_and_validate(cache):
    ds = CRCNSAA5Dataset(cache, dt_ms=5.0)
    assert ds.T == 1000 and ds.onset_bin == 100
    assert ds.N_neurons == 2                               # empty unit dropped
    assert [m["name"] for m in ds.stim_meta] == [CALL, SONG]
    for s, stim in enumerate(ds.stims):
        assert stim.shape == (1, 32, 1000)
        for n in range(ds.N_neurons):
            r = ds.responses[s][n]
            assert tuple(r.shape) == (1, 1) or r.shape[1] == 1000
    b = ds[0]
    assert set(b) >= {"stims", "responses", "valid_mask", "stim_meta"}


def test_default_drops_misaligned_and_overlapping_playbacks(cache):
    ds = CRCNSAA5Dataset(cache)
    pm = {p["trial"]: p for p in ds.playback_meta[SITE]}
    assert pm[9]["misaligned"] and pm[9]["dropped"]         # mic: sound 1 s late
    assert pm[3]["next_playback_overlap"] and pm[3]["dropped"]   # next onset at +3 s
    assert not pm[7]["dropped"] and not pm[12]["dropped"]
    s, n = _pair(ds, CALL, "e10-c1")                         # kept trials 3, 9 -> both dropped
    assert tuple(ds.responses[s][n].shape) == (1, 1) and not ds.nrn_masks[s, n]
    s, n = _pair(ds, SONG, "e10-c1")
    assert ds.trial_ids[s][n] == [7, 12]


def test_opt_out_of_trial_dropping(cache):
    ds = CRCNSAA5Dataset(cache, drop_misaligned=False, drop_next_playback_overlap=False)
    s, n = _pair(ds, CALL, "e10-c1")
    assert ds.trial_ids[s][n] == [3, 9] and ds.responses[s][n].shape[0] == 2
    assert not any(p["dropped"] for p in ds.playback_meta[SITE])


def test_spike_binning_is_exact(cache):
    ds = CRCNSAA5Dataset(cache, dt_ms=5.0, smooth=False, drop_misaligned=False,
                         drop_next_playback_overlap=False)
    for unit, trials in UNITS.items():
        for t in trials:
            stim = PLAYBACKS[t][0]
            s, n = _pair(ds, stim, unit[len("goodPlayback-"):-len(".pkl")])
            row = ds.trial_ids[s][n].index(t)
            assert ds.responses[s][n][row].sum().item() == len(_spikes(unit, t))


def test_no_nan_inside_real_responses(cache):
    ds = CRCNSAA5Dataset(cache)
    for row in ds.responses:
        for r in row:
            assert tuple(r.shape) == (1, 1) or not torch.isnan(r).any()


def test_min_trials(cache):
    ds = CRCNSAA5Dataset(cache, min_trials=2, drop_misaligned=False, drop_next_playback_overlap=False)
    s, n = _pair(ds, SONG, "e11-c4")                         # only trial 12
    assert not ds.nrn_masks[s, n]


def test_causal_spectrogram_starts_at_sound_onset(cache):
    ds = CRCNSAA5Dataset(cache, dt_ms=5.0)
    spec = ds.stims[0][0]
    assert spec[:, : ds.onset_bin].abs().max().item() == 0.0   # silence before t=0
    assert spec[:, ds.onset_bin].sum().item() > 0.0


def test_waveform_mode(cache):
    ds = CRCNSAA5Dataset(cache, dt_ms=5.0, return_waveform=True, audio_fs=25000)
    assert ds.hop == 125
    for w in ds.stims:
        assert w.shape == (1, 1000 * 125)
    # int16 full-scale units: the synthetic call peaks at 3000 / 32768
    assert ds.stims[0].abs().max().item() == pytest.approx(3000 / 32768, rel=1e-3)


def test_metadata(cache):
    ds = CRCNSAA5Dataset(cache)
    call, song = ds.stim_meta
    assert (call["stim_class"], call["call_type"], call["vocalizer"], call["vocalizer_code"],
            call["vocalizer_sex"], call["vocalizer_age"], call["rendition"]) == \
        ("call", "Te", "BlaBla0506", "MAF", "M", "adult", "8-1-9")
    assert call["native_fs"] == 25000 and song["native_fs"] == 32000
    assert call["sound_onset_s"] == pytest.approx(0.0, abs=5e-3)   # Hann-tapered onset
    assert call["duration_s"] == pytest.approx(0.4, abs=0.01)
    assert (song["stim_class"], song["call_type"], song["song_id"]) == ("song", "So", 3)
    n = ds.nrn_meta[0]
    assert n["site"] == SITE and n["animal_id"] == "ZF4F" and n["sex"] == "F"
    assert n["depth_turns"] == 3.0 and n["depth_um"] == 750.0
    assert n["spike_snr"] == 6.5 and np.isfinite(n["auditory_z"])
    assert "is_auditory" not in n and "auditory_p" not in n


def test_stim_info_parsing():
    assert _stim_info("stim10_sfilt.wav")["stim_class"] == "song_sfilt"
    assert _stim_info("stim1_tfilt.wav")["song_id"] == 1
    assert _stim_info("randripple7.wav")["stim_class"] == "ripple"
    assert _stim_info("LblGre0000_UCF_LT_12-3-4_fs25k.wav")["vocalizer_sex"] is None
    assert _stim_info("LblGre0000_UCF_LT_12-3-4_fs25k.wav")["vocalizer_age"] == "chick"
    assert {_stim_info(f)["stim_class"] for f in ("stim1.wav", "stim1_sfilt.wav", "stim1_tfilt.wav",
                                                   "randripple1.wav", CALL)} == set(STIM_CLASSES)


def test_stimulus_class_filter(cache):
    ds = CRCNSAA5Dataset(cache, stimuli=("song",))
    assert [m["name"] for m in ds.stim_meta] == [SONG]
    assert ds.select_stim_class("song") == [0]
    with pytest.raises(ValueError, match="stimulus class"):
        CRCNSAA5Dataset(cache, stimuli=("speech",))


def test_bad_arguments_and_missing_cache(cache, tmp_path):
    with pytest.raises(ValueError, match="must divide"):
        CRCNSAA5Dataset(cache, dt_ms=3.0)
    with pytest.raises(ValueError, match="must be an integer"):
        CRCNSAA5Dataset(cache, dt_ms=1.0, audio_fs=22050)
    with pytest.raises(ValueError, match="Unknown AA5 bird"):
        CRCNSAA5Dataset(cache, animals=["ZF9X"])
    with pytest.raises(FileNotFoundError, match="prepare_aa5"):
        CRCNSAA5Dataset(tmp_path)


def test_raw_path_prepares_on_the_fly(raw_site, tmp_path):  # noqa: F811
    ds = CRCNSAA5Dataset(tmp_path / "cache", raw_path=raw_site)
    assert ds.N_neurons == 2 and ds.sites == [SITE]


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

AA5_DATA = os.environ.get("AA5_DATA")


@pytest.mark.skipif(not AA5_DATA, reason="set $AA5_DATA to a prepare_aa5 cache to run")
def test_real_cache_smoke():
    ds = CRCNSAA5Dataset(AA5_DATA)
    assert ds.N_neurons > 0 and len(ds.stims) > 0
    calls = [m for m in ds.stim_meta if m["stim_class"] == "call"]
    assert {m["call_type"] for m in calls} <= set(("Ag", "Be", "DC", "Di", "LT", "Ne", "So", "Te", "Th", "Wh"))
    for row in ds.responses:
        for r in row:
            assert tuple(r.shape) == (1, 1) or (r.shape[1] == ds.T and not torch.isnan(r).any())
