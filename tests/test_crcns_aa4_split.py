"""Tests for the CRCNS-AA4 stimulus identity table (duplicate merging, ``overlap_group``,
``caller`` / ``call_type`` / ``identity``) and ``CRCNSAA4Dataset.group_split``.

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
    for md5, (canon, group, stim_type, caller, call_type, ident) in sorted(aa4_stim_groups().items()):
        uid = canon if merge else md5
        if uid in seen:
            continue
        seen.add(uid)
        meta.append({"name": uid, "type": stim_type, "overlap_group": group, "caller": caller,
                     "call_type": call_type, "identity": ident})
    ds = CRCNSAA4Dataset.__new__(CRCNSAA4Dataset)
    ds.stim_meta = meta
    return ds


def test_table_is_consistent():
    t = aa4_stim_groups()
    assert len(t) == 594
    canon = {v[0] for v in t.values()}
    assert len(canon) == 539
    assert len({v[1] for v in t.values()}) == 224
    assert len({v[5] for v in t.values()}) == 136
    for md5, (c, g, typ, caller, ctype, ident) in t.items():
        assert t[c][0] == c                       # a canonical copy is its own canonical
        assert t[c][1:] == (g, typ, caller, ctype, ident)   # duplicates share everything
        assert ctype in {"call": {"Ag", "Be", "DC", "Di", "LT", "Ne", "Te", "Th", "Wh"},
                         "song": {"So"}, "mlnoise": {"Ri"}}[typ]
        assert (caller is None) == (typ == "mlnoise" or (typ == "song" and caller is None))
    types, ident_of_group, members = collections.defaultdict(set), {}, collections.defaultdict(set)
    for _, g, typ, caller, ctype, ident in t.values():
        types[g].add(typ)
        assert ident_of_group.setdefault(g, ident) == ident   # an overlap group is in one identity
        members[ident].add((caller, ctype))
    assert all(len(v) == 1 for v in types.values())
    # an identity is one caller x call type, except the learned-song trio and the two
    # cross-caller near-identical call pairs
    multi = sorted(sorted(map(str, v)) for v in members.values() if len(v) > 1)
    assert multi == [["('LblBlu1630', 'LT')", "('LblBlu2028', 'LT')"],
                     ["('WhiGra0114', 'Th')", "('WhiWhi1415', 'Th')"],
                     ["('wblu5396', 'So')", "('wblu5499', 'So')", "('wr5832', 'So')"]]
    # a caller x call type is in one identity
    ident_of_ct = collections.defaultdict(set)
    for _, _, _, caller, ctype, ident in t.values():
        if caller is not None:
            ident_of_ct[(caller, ctype)].add(ident)
    assert all(len(v) == 1 for v in ident_of_ct.values())


HOLDOUT_KEY = {"audio": "overlap_group", "caller_type": "identity", "caller": "caller",
               "call_type": "call_type"}
# what each holdout keeps in one set (shared audio always; then its own key)
KEPT = {"audio": ("overlap_group",), "caller_type": ("overlap_group", "identity"),
        "caller": ("overlap_group", "identity", "caller"),
        "call_type": ("overlap_group", "identity", "call_type")}


@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize("holdout", list(HOLDOUT_KEY))
def test_split_never_splits_a_unit(merge, holdout):
    ds = _dataset(merge)
    split = ds.group_split(seed=0, holdout=holdout)
    assert sorted(i for v in split.values() for i in v) == list(range(len(ds.stim_meta)))
    for key in KEPT[holdout]:
        sets = collections.defaultdict(set)
        for k, v in split.items():
            for i in v:
                m = ds.stim_meta[i]
                if m[key] is None or (key == "call_type" and m["type"] != "call"):
                    continue
                sets[m[key]].add(k)
        assert all(len(v) == 1 for v in sets.values()), (holdout, key)


def test_caller_type_split_has_every_call_type_in_train_and_test():
    ds = _dataset()
    split = ds.group_split(seed=0)
    for k in ("train", "test"):
        assert {ds.stim_meta[i]["call_type"] for i in split[k]} == {
            "Ag", "Be", "DC", "Di", "LT", "Ne", "Te", "Th", "Wh", "So", "Ri"}
    assert {ds.stim_meta[i]["call_type"] for i in split["val"]} == {"Ag", "Be", "DC", "Ne", "Te", "So", "Ri"}


def test_split_rejects_bad_holdout():
    with pytest.raises(ValueError):
        _dataset().group_split(seed=0, holdout="bird")


def test_audio_split_fractions_per_type():
    ds = _dataset()
    split = ds.group_split(seed=0, holdout="audio")
    for stim_type in ("song", "call", "mlnoise"):
        n = {k: sum(ds.stim_meta[i]["type"] == stim_type for i in v) for k, v in split.items()}
        total = sum(n.values())
        tol = 2 / total if stim_type != "call" else 0.02
        assert abs(n["train"] / total - 0.7) <= tol + 0.05
        assert abs(n["test"] / total - 0.2) <= tol + 0.05
        assert n["val"] > 0 and n["test"] > 0


@pytest.mark.parametrize("holdout,counts", [
    ("caller_type", (376, 50, 113)), ("caller", (385, 49, 105)),
    ("call_type", (354, 15, 170)), ("audio", (383, 51, 105))])
def test_seed0_draw_is_stable(holdout, counts):
    split = _dataset().group_split(seed=0, holdout=holdout)
    assert tuple(len(split[k]) for k in ("train", "val", "test")) == counts


@pytest.mark.parametrize("holdout", list(HOLDOUT_KEY))
def test_split_is_deterministic_and_subset_invariant(holdout):
    ds = _dataset()
    a, b = ds.group_split(seed=3, holdout=holdout), ds.group_split(seed=3, holdout=holdout)
    assert a == b
    assert ds.group_split(seed=4, holdout=holdout) != a
    # a subset of the stimuli (e.g. fewer birds loaded) keeps each sound's set
    full = {ds.stim_meta[i]["name"]: k for k, v in a.items() for i in v}
    sub = CRCNSAA4Dataset.__new__(CRCNSAA4Dataset)
    sub.stim_meta = ds.stim_meta[::3]
    for k, v in sub.group_split(seed=3, holdout=holdout).items():
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
    for md5, (c, *_) in t.items():
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
