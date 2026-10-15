"""Tests for the Wehr stimulus groups (``WEHR_SHARED_AUDIO``, ``overlap_group``) and
``CRCNSAC1Dataset.group_split``.

The split only reads ``stim_meta``, so most tests run on a ``stim_meta`` built for the
51 Wehr files (no data needed). One data-gated test checks every listed overlap on the
stimulus waveforms.
"""
from __future__ import annotations

import collections
import os

import numpy as np
import pytest
import torch

from deepSTRF.datasets.audio.crcns_ac1 import (
    WEHR_SHARED_AUDIO, CRCNSAC1Dataset, _WEHR_CORPUS, wehr_overlap_group)


def _dataset(stride=1):
    meta = [{"name": f"wehr/{c}/{i}", "experimenter": "wehr", "category": c, "idx": i,
             "overlap_group": wehr_overlap_group(c, i)}
            for c, n in _WEHR_CORPUS for i in range(1, n + 1)]
    ds = CRCNSAC1Dataset.__new__(CRCNSAC1Dataset)
    ds.stim_meta = meta[::stride]
    return ds


def test_groups():
    files = [f"{c}/{i}" for c, n in _WEHR_CORPUS for i in range(1, n + 1)]
    listed = [f for g in WEHR_SHARED_AUDIO for f in g]
    assert len(files) == 51 and len(listed) == len(set(listed)) == 23
    assert set(listed) <= set(files)
    assert len({wehr_overlap_group(*f.rsplit("/", 1)[:1], int(f.rsplit("/", 1)[1])) for f in files}) == 37
    # every group holds a fragment file: streams share audio only through fragments
    assert all(any(f.startswith("fragments/") for f in g) for g in WEHR_SHARED_AUDIO)


def test_split_keeps_groups_together_and_covers_everything():
    ds = _dataset()
    split = ds.group_split(seed=0)
    assert sorted(i for v in split.values() for i in v) == list(range(len(ds.stim_meta)))
    where = collections.defaultdict(set)
    for k, v in split.items():
        for i in v:
            where[ds.stim_meta[i]["overlap_group"]].add(k)
    assert all(len(v) == 1 for v in where.values())


@pytest.mark.parametrize("seed", range(5))
def test_every_category_reaches_test(seed):
    ds = _dataset()
    split = ds.group_split(seed=seed)
    cats = {ds.stim_meta[i]["category"] for i in split["test"]}
    assert cats == {"fragments", "category1", "category2", "category3"}


def test_seed0_draw_is_stable():
    split = _dataset().group_split(seed=0)
    assert tuple(len(split[k]) for k in ("train", "val", "test")) == (33, 4, 14)


def test_split_is_deterministic_and_subset_invariant():
    ds = _dataset()
    a = ds.group_split(seed=3)
    assert a == ds.group_split(seed=3) and a != ds.group_split(seed=4)
    full = {ds.stim_meta[i]["name"]: k for k, v in a.items() for i in v}
    sub = _dataset(stride=3)
    for k, v in sub.group_split(seed=3).items():
        assert all(full[sub.stim_meta[i]["name"]] == k for i in v)


def test_split_rejects_asari_sequences_and_bad_fractions():
    ds = _dataset()
    with pytest.raises(ValueError):
        ds.group_split(seed=0, fractions=(0.8, 0.1, 0.2))
    ds.stim_meta.append({"name": "asari/x", "experimenter": "asari"})   # sequence level
    with pytest.raises(ValueError, match="asari_level='segment'"):
        ds.group_split(seed=0)


# (fragment file, start sample, file that copies it, start sample): aligned starts of
# each shared stretch, from the spectrogram match refined on the waveforms
_COPIES = [
    ("fragments/1", 974, "category1/10", 391601),
    ("fragments/12", -719, "category2/10", 345702),
    ("fragments/12", 1027365, "category2/5", 585936),
    ("fragments/13", 1072662, "category2/15", 393554),
    ("fragments/13", -956, "category2/5", 1022458),
    ("fragments/14", 143946, "category2/3", 1466793),
    ("fragments/2", 2925, "category2/2", 979490),
    ("fragments/3", -14, "category1/11", 1562496),
    ("fragments/4", -7, "category1/1", 781248),
    ("fragments/5", -4, "category2/7", 488280),
    ("fragments/6", 977, "category2/17", 977),
    ("fragments/7", 977, "category2/18", 977),
    ("fragments/8", 977, "category2/10", 977),
    ("fragments/8", 346421, "fragments/12", 0),
    ("fragments/9", -8, "category2/18", 1269528),
]


def _wehr_stimuli():
    for env in ("CRCNS_AC1_DATA",):
        root = os.environ.get(env)
        if root and os.path.isdir(os.path.join(root, "crcns-ac1", "wehr", "Stimuli")):
            return os.path.join(root, "crcns-ac1", "wehr", "Stimuli")
    pytest.skip("CRCNS-AC1 data missing (set CRCNS_AC1_DATA to the folder holding crcns-ac1/)")


@pytest.mark.slow
def test_listed_copies_match_on_the_waveforms():
    from deepSTRF.datasets.audio._crcns_ac1_native import _load_mat, _unstruct
    root = _wehr_stimuli()
    load = lambda f: np.asarray(_unstruct(_load_mat(os.path.join(root, f + ".mat"))["stimulus"]).samples,
                                dtype=np.float64).ravel()
    group_of = {f: g for g in WEHR_SHARED_AUDIO for f in g}
    n = 97656 * 2                       # 2 s, starting 0.2 s into the shared stretch
    for frag, i, other, j in _COPIES:
        assert group_of[frag] is group_of[other]
        x, y = load(frag), load(other)
        a, b = i + 19531, j + 19531
        assert np.corrcoef(x[a:a + n], y[b:b + n])[0, 1] > 0.99, (frag, other)


# ============================================================
# Asari fragment table and segment-level split
# ============================================================

from deepSTRF.datasets.audio.crcns_ac1 import _asari_table, asari_fragment_sources  # noqa: E402


def _asari_dataset(stride=1):
    meta = [{"name": f"asari/{f}", "experimenter": "asari", "segment_file": f, "source": src,
             "overlap_group": f"asari/{src}"}
            for f, (src, _) in sorted(asari_fragment_sources().items())]
    ds = CRCNSAC1Dataset.__new__(CRCNSAC1Dataset)
    ds.stim_meta = meta[::stride]
    return ds


def test_asari_table():
    src = asari_fragment_sources()
    assert len(src) == 49 and len({v[0] for v in src.values()}) == 33
    # a short / long cut has the source of its base file, when the base is listed
    for f, (s, _) in src.items():
        base = f[:-4].rstrip("sl") + ".mat"
        if base != f and base in src:
            assert src[base][0] == s, f
    ens = _asari_table()["ensembles"]
    assert sum(c for c, _ in ens) == 53                      # 39 A1 + 14 MGB cells
    assert {x for _, e in ens for x in e} == {v[0] for v in src.values()}


def test_asari_split_keeps_sources_together_and_is_stable():
    ds = _asari_dataset()
    split = ds.group_split(seed=0)
    assert sorted(i for v in split.values() for i in v) == list(range(len(ds.stim_meta)))
    where = collections.defaultdict(set)
    for k, v in split.items():
        for i in v:
            where[ds.stim_meta[i]["source"]].add(k)
    assert all(len(v) == 1 for v in where.values())
    assert split == ds.group_split(seed=0) and split != ds.group_split(seed=1)
    full = {ds.stim_meta[i]["name"]: k for k, v in split.items() for i in v}
    sub = _asari_dataset(stride=3)
    for k, v in sub.group_split(seed=0).items():
        assert all(full[sub.stim_meta[i]["name"]] == k for i in v)


def test_asari_largest_fragment_set_gets_val_and_test():
    ds = _asari_dataset()
    split = ds.group_split(seed=0)
    set_of = {ds.stim_meta[i]["source"]: k for k, v in split.items() for i in v}
    _, largest = _asari_table()["ensembles"][0]
    assert {set_of[s] for s in largest} == {"train", "val", "test"}


@pytest.mark.slow
def test_asari_segments_are_slices_of_the_sequences():
    """MGB: every trial of a sequence, cut at a fragment's onset, is a trial of that
    fragment at the segment level. (The sequence-level stimulus is cut to the shortest
    recording across cells, so the segment level can hold more occurrences per cell than
    can be checked here.) 1-ms bins: the segment level cuts at the exact onset sample,
    the sequence level bins on a fixed grid, so 10-ms bins would differ by a sub-bin shift."""
    root = os.environ.get("CRCNS_AC1_DATA")
    if not root or not os.path.isdir(os.path.join(root, "crcns-ac1")):
        pytest.skip("CRCNS-AC1 data missing (set CRCNS_AC1_DATA)")
    kw = dict(path=root, experimenter="asari", sites="MGB", dt_ms=1.0)
    seq = CRCNSAC1Dataset(**kw)
    seg = CRCNSAC1Dataset(asari_level="segment", **kw)
    assert [m["session"] for m in seq.nrn_meta] == [m["session"] for m in seg.nrn_meta]
    by_name = {m["name"]: s for s, m in enumerate(seq.stim_meta)}
    n_checked = 0
    for s, m in enumerate(seg.stim_meta):
        assert m["occurrences"] and m["source"] == asari_fragment_sources()[m["segment_file"]][0]
        for n in range(seg.N_neurons):
            r = seg.responses[s][n]
            if torch.isnan(r).all():
                continue
            T, expected = r.shape[1], []
            for name, _pos, _prev, onset in m["occurrences"]:
                q = seq.responses[by_name[name]][n]
                b0 = int(round(onset / 0.001))
                if not torch.isnan(q).all() and b0 + T <= q.shape[1]:
                    expected.append(q[:, b0:b0 + T])
            if not expected:
                continue
            e = torch.cat(expected).numpy()
            assert r.shape[0] >= len(e), (m["name"], n)
            keep = e.std(1) > 0                         # silent trials have no r
            C = np.corrcoef(np.concatenate([e[keep], r.numpy()]))[:keep.sum(), keep.sum():]
            assert np.nanmax(C, axis=1).min() > 0.999, (m["name"], n)
            n_checked += 1
    assert n_checked > 50
