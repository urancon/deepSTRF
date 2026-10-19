"""Tests for the Le 2025 song-level split (``Le2025Dataset.song_split``) and the
``song_id`` / ``familiarity_group`` stim metadata it relies on.

The split logic only reads ``stim_meta``, so most tests run on a hand-built
``stim_meta`` (no data needed). One data-gated test checks the hard-coded song
table against the figshare metadata, reading only ``song-birds.csv`` and the
stimulus file names (no responses, so no memory gate).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from deepSTRF.datasets.audio.le_2025 import (
    NATURAL_SONG_GROUPS,
    SYNTH_MOTIFS,
    Le2025Dataset,
    _NAT8B_TO_SONG,
    _parse_stim_stem,
)

_NAT8A_VARIANTS = [("C", None)] + [(v, ci) for ci in (1, 2) for v in ("G", "N", "GB", "CB")]
_EP_VARIANTS = ([("C", None), ("CM", None)]
                + [(v, ci) for ci in (1, 2) for v in ("G", "N", "GB", "CB", "GM")])


def _meta(experiment):
    """stim_meta as the loader builds it (only the keys the split reads)."""
    out = []
    if experiment == "nat8a":
        for song in NATURAL_SONG_GROUPS:
            for v, ci in _NAT8A_VARIANTS:
                out.append({"motif": song, "variant": v, "critical_interval": ci,
                            "experiment": experiment, "song_id": song,
                            "familiarity_group": NATURAL_SONG_GROUPS[song]})
    elif experiment == "nat8b":
        for mk, song in _NAT8B_TO_SONG.items():
            for v, ci in _EP_VARIANTS:
                out.append({"motif": mk, "variant": v, "critical_interval": ci,
                            "experiment": experiment, "song_id": song,
                            "familiarity_group": NATURAL_SONG_GROUPS[song]})
    else:
        for m in SYNTH_MOTIFS:
            for v, ci in _EP_VARIANTS:
                out.append({"motif": m, "variant": v, "critical_interval": ci,
                            "experiment": experiment, "song_id": m,
                            "familiarity_group": None})
    return out


def _ds(*experiments):
    """A bare Le2025Dataset carrying only stim_meta (concat order = argument order)."""
    ds = Le2025Dataset.__new__(Le2025Dataset)
    ds.stim_meta = [m for e in experiments for m in _meta(e)]
    return ds


def _songs_by_set(ds, split, exclude_n=True):
    out = {}
    for k, idx in split.items():
        out[k] = {ds.stim_meta[i]["song_id"] for i in idx
                  if not (exclude_n and ds.stim_meta[i]["variant"] == "N")}
    return out


@pytest.mark.parametrize("experiments", [("nat8a",), ("nat8b",), ("synth8b",),
                                         ("nat8a", "nat8b", "synth8b")])
def test_partition_disjoint_and_complete(experiments):
    ds = _ds(*experiments)
    split = ds.song_split(seed=0)
    assert set(split) == {"train", "val", "test"}
    allidx = split["train"] + split["val"] + split["test"]
    assert sorted(allidx) == list(range(len(ds.stim_meta)))


def test_every_song_in_one_set_except_noise_bursts():
    ds = _ds("nat8a", "nat8b", "synth8b")
    by = _songs_by_set(ds, ds.song_split(seed=3))
    assert not (by["train"] & by["val"]) and not (by["train"] & by["test"])
    assert not (by["val"] & by["test"])


def test_noise_bursts_always_train():
    ds = _ds("nat8a", "nat8b", "synth8b")
    for seed in range(5):
        split = ds.song_split(seed=seed)
        for k in ("val", "test"):
            assert all(ds.stim_meta[i]["variant"] != "N" for i in split[k])


def test_set_sizes_and_familiarity_stratification():
    ds = _ds("nat8a", "nat8b", "synth8b")
    test_groups = set()
    for seed in range(20):
        by = _songs_by_set(ds, ds.song_split(seed=seed))
        nat = {}
        for k, n in (("train", 6), ("val", 1), ("test", 1)):
            nat[k] = [s for s in by[k] if s in NATURAL_SONG_GROUPS]
            syn = [s for s in by[k] if s in SYNTH_MOTIFS]
            assert len(nat[k]) == n and len(syn) == n, (seed, k, by[k])
        val_group, test_group = (NATURAL_SONG_GROUPS[nat[k][0]] for k in ("val", "test"))
        assert val_group != test_group, seed
        test_groups.add(test_group)
    assert test_groups == {"178B", "180B"}   # either group can give the test song


def test_seed0_draw_is_stable():
    """Published numbers refer to song_split(seed=0); changing the draw breaks them."""
    ds = _ds("nat8a", "nat8b", "synth8b")
    by = _songs_by_set(ds, ds.song_split(seed=0))
    assert by["test"] == {"O129", "synth8mk0"}
    assert by["val"] == {"B2", "synth8mk6"}


def test_same_seed_same_songs_across_experiments_and_concat():
    pooled = _ds("nat8a", "nat8b", "synth8b")
    ref = _songs_by_set(pooled, pooled.song_split(seed=7))
    for e in ("nat8a", "nat8b", "synth8b"):
        ds = _ds(e)
        by = _songs_by_set(ds, ds.song_split(seed=7))
        for k in by:
            assert by[k] <= ref[k], (e, k)


def test_seed_none_is_random_and_int_is_reproducible():
    ds = _ds("nat8b")
    assert ds.song_split(seed=1) == ds.song_split(seed=1)
    draws = {tuple(sorted(_songs_by_set(ds, ds.song_split())["test"])) for _ in range(30)}
    assert len(draws) > 1


def test_nat8b_motif_names_resolve_to_songs():
    assert set(_NAT8B_TO_SONG.values()) == set(NATURAL_SONG_GROUPS)
    assert _parse_stim_stem("ep_nat8mk1_C", "nat8b")["motif"] == "nat8mk1"
    assert _NAT8B_TO_SONG["nat8mk1"] == "R56"


# ---------------------------------------------------------------------------
# Data-gated: the hard-coded song table matches the figshare metadata
# ---------------------------------------------------------------------------

_ROOT = Path(os.environ.get(
    "LE2025_DATA",
    str(Path.home() / "Library/Caches/deepSTRF/Le_2025/zebf-auditory-restoration-1"),
))
_HAS_META = (_ROOT / "metadata" / "song-birds.csv").exists()


@pytest.mark.skipif(not _HAS_META, reason=f"Le2025 data not at {_ROOT}; set $LE2025_DATA.")
def test_song_table_matches_figshare_metadata():
    rows = list(csv.DictReader(open(_ROOT / "metadata" / "song-birds.csv")))
    assert {r["bird"]: r["group"] for r in rows} == NATURAL_SONG_GROUPS
    assert {r["nat8b_encoding"]: r["bird"] for r in rows} == _NAT8B_TO_SONG
    motifs = {}
    for e in ("nat8a", "nat8b", "synth8b"):
        motifs[e] = {_parse_stim_stem(p.stem, e)["motif"]
                     for p in (_ROOT / f"{e}-stimuli").glob("*.wav")}
    assert motifs["nat8a"] == set(NATURAL_SONG_GROUPS)
    assert motifs["nat8b"] == set(_NAT8B_TO_SONG)
    assert motifs["synth8b"] == set(SYNTH_MOTIFS)
