"""Tests for the CRCNS-AA4 stimulus identity table (duplicate merging, ``overlap_group``)
and ``CRCNSAA4Dataset.group_split``.

The split only reads ``stim_meta``, so most tests run on a ``stim_meta`` built from
the shipped table (no data needed). One data-gated test checks the table against the
wav files: every md5 is found, and the copies merged into one stimulus have the same
length and the same log-mel spectrogram (on its louder half of bins).
"""
from __future__ import annotations

import collections
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from deepSTRF.datasets.audio.crcns_aa4 import AA4_ANIMAL_IDS, CRCNSAA4Dataset, aa4_stim_groups


def _dataset(merge=True):
    """A CRCNSAA4Dataset carrying only the stim_meta the loader would build for the
    whole corpus (only the keys the split reads)."""
    meta, seen = [], set()
    for md5, (canon, group, stim_type) in sorted(aa4_stim_groups().items()):
        uid = canon if merge else md5
        if uid in seen:
            continue
        seen.add(uid)
        meta.append({"name": uid, "type": stim_type, "overlap_group": group})
    ds = CRCNSAA4Dataset.__new__(CRCNSAA4Dataset)
    ds.stim_meta = meta
    return ds


def test_table_is_consistent():
    t = aa4_stim_groups()
    assert len(t) == 594
    canon = {c for c, _, _ in t.values()}
    assert len(canon) == 539
    assert len({g for _, g, _ in t.values()}) == 369
    for md5, (c, g, typ) in t.items():
        assert t[c][0] == c                       # a canonical copy is its own canonical
        assert t[c][1] == g and t[c][2] == typ    # duplicates share group and type
    types = collections.defaultdict(set)
    for _, g, typ in t.values():
        types[g].add(typ)
    assert all(len(v) == 1 for v in types.values())


@pytest.mark.parametrize("merge", [True, False])
def test_split_never_splits_a_group(merge):
    ds = _dataset(merge)
    split = ds.group_split(seed=0)
    assert sorted(i for v in split.values() for i in v) == list(range(len(ds.stim_meta)))
    groups = {k: {ds.stim_meta[i]["overlap_group"] for i in v} for k, v in split.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_split_fractions_per_type():
    ds = _dataset()
    split = ds.group_split(seed=0)
    for stim_type in ("song", "call", "mlnoise"):
        n = {k: sum(ds.stim_meta[i]["type"] == stim_type for i in v) for k, v in split.items()}
        total = sum(n.values())
        tol = 2 / total if stim_type != "call" else 0.02
        assert abs(n["train"] / total - 0.7) <= tol + 0.05
        assert abs(n["test"] / total - 0.2) <= tol + 0.05
        assert n["val"] > 0 and n["test"] > 0


def test_seed0_draw_is_stable():
    split = _dataset().group_split(seed=0)
    assert {k: len(v) for k, v in split.items()} == {"train": 377, "val": 54, "test": 108}


def test_split_is_deterministic_and_subset_invariant():
    ds = _dataset()
    a, b = ds.group_split(seed=3), ds.group_split(seed=3)
    assert a == b
    assert ds.group_split(seed=4) != a
    # a subset of the stimuli (e.g. fewer birds loaded) keeps each sound's set
    full = {ds.stim_meta[i]["name"]: k for k, v in a.items() for i in v}
    sub = CRCNSAA4Dataset.__new__(CRCNSAA4Dataset)
    sub.stim_meta = ds.stim_meta[::3]
    for k, v in sub.group_split(seed=3).items():
        assert all(full[sub.stim_meta[i]["name"]] == k for i in v)


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        _dataset().group_split(seed=0, fractions=(0.8, 0.1, 0.2))


def _aa4_root():
    env = os.environ.get("AA4_DATA")
    if env:
        return Path(env)
    from deepSTRF.utils.data_download import default_cache_dir
    return Path(default_cache_dir("AA4"))


@pytest.mark.slow
def test_table_matches_the_wavs():
    root = _aa4_root()
    if not all((root / a / "wavfiles").is_dir() for a in AA4_ANIMAL_IDS):
        pytest.skip("CRCNS-AA4 data missing (set AA4_DATA to the folder holding the 6 birds)")
    import soundfile as sf
    import torch
    import torchaudio

    t = aa4_stim_groups()
    path = {}
    for a in AA4_ANIMAL_IDS:
        for f in sorted((root / a / "wavfiles").glob("*.wav")):
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            if md5 in t:
                path.setdefault(md5, f)
    assert set(path) == set(t), "table md5s missing from the wav files"

    mel = torchaudio.transforms.MelSpectrogram(24414, n_fft=1024, hop_length=244, n_mels=32,
                                               f_min=250, f_max=8000)
    spec = lambda w: torch.log(mel(torch.from_numpy(w)) + 1e-6).numpy()
    copies = collections.defaultdict(list)
    for md5, (c, _, _) in t.items():
        copies[c].append(md5)
    n_sets = 0
    for c, ms in copies.items():
        if len(ms) == 1:
            continue
        n_sets += 1
        ref = sf.read(path[c], dtype="float32")[0]
        r = spec(ref)
        loud = r > np.median(r)   # the copies differ by quantisation noise, visible in silence only
        for m in ms:
            w = sf.read(path[m], dtype="float32")[0]
            assert len(w) == len(ref)
            assert np.corrcoef(spec(w)[loud], r[loud])[0, 1] > 0.999
    assert n_sets == 35
