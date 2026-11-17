"""Fast, synthetic tests for the Yang & Long 2025 prepare step (episodes + splits).

Real-data checks live in ``tests/test_yang2025_real.py`` (slow, need $YANG2025_RAW).
"""
import numpy as np
import pytest

from deepSTRF.datasets.audio._yang2025_prepare import assign_splits, build_episodes


def _sy(rows):
    return np.asarray(rows, dtype=float)


def test_build_episodes_types_and_ids():
    sy = _sy([[1.0, 1.1], [1.2, 1.3], [5.0, 5.2], [8.0, 8.1], [8.2, 8.3], [12.0, 12.2], [15.0, 15.15]])
    flag = np.array([1, 1, 0, 1, 1, 0, 0])
    eps = build_episodes(sy, flag, [(1.0, 1.3), (8.0, 8.3)], bird="Ti81")
    assert [e["episode_type"] for e in eps] == ["warble", "call", "warble", "call", "call"]
    assert [e["bout_id"] for e in eps] == ["Ti81_w000", "Ti81_c000", "Ti81_w001", "Ti81_c001", "Ti81_c002"]
    assert eps[0]["onset_s"] == 1.0 and eps[0]["offset_s"] == 1.3
    assert all(e["bird"] == "Ti81" for e in eps)


def test_min_duration_drop():
    sy = _sy([[1.0, 1.001], [2.0, 2.2]])
    eps = build_episodes(sy, np.array([0, 0]), [], bird="Li145")
    assert [e["bout_id"] for e in eps] == ["Li145_c001"]


def test_episode_never_crosses_splice():
    sy = _sy([[1700.0, 1700.1], [1709.9, 1710.0]])
    eps = build_episodes(sy, np.array([1, 1]), [(1700.0, 1710.0)], bird="Or61", splices_s=[1703.0])
    assert [(e["onset_s"], e["offset_s"]) for e in eps] == [(1700.0, 1703.0), (1703.0, 1710.0)]
    assert eps[0]["bout_id"] == eps[1]["bout_id"] == "Or61_w000"


def _random_episodes(rng, n_bouts=40, bird="B"):
    eps = []
    for k in range(n_bouts):
        typ = "warble" if k % 3 else "call"
        for piece in range(rng.integers(1, 4)):
            on = 100.0 * k + 10 * piece
            eps.append(dict(bird=bird, episode_type=typ, bout_id=f"{bird}_{typ[0]}{k:03d}",
                            onset_s=on, offset_s=on + rng.uniform(0.1, 5.0)))
    return eps


def test_splits_keep_bouts_whole():
    eps = _random_episodes(np.random.default_rng(0))
    assign_splits(eps)
    split_of = {}
    for e in eps:
        assert e["split"] in ("train", "val", "test")
        assert split_of.setdefault(e["bout_id"], e["split"]) == e["split"]


def test_outsized_bout_goes_to_train():
    eps = [dict(bird="B", episode_type="warble", bout_id="B_w000", onset_s=0.0, offset_s=50.0)]
    eps += [dict(bird="B", episode_type="warble", bout_id=f"B_w{k:03d}",
                 onset_s=100.0 * k, offset_s=100.0 * k + 1.0) for k in range(1, 51)]
    for seed in range(5):
        assign_splits(eps, seed=seed)
        assert eps[0]["split"] == "train"
        for split in ("val", "test"):
            dur = sum(e["offset_s"] - e["onset_s"] for e in eps if e["split"] == split)
            assert 0.5 * 15.0 <= dur <= 1.25 * 15.0


def test_every_split_nonempty_per_stratum():
    eps = [dict(bird="B", episode_type="call", bout_id=f"B_c{k:03d}",
                onset_s=10.0 * k, offset_s=10.0 * k + d) for k, d in enumerate([0.2, 0.15, 0.25])]
    assign_splits(eps)
    assert sorted(e["split"] for e in eps) == ["test", "train", "val"]


def test_splits_deterministic():
    a = _random_episodes(np.random.default_rng(1))
    b = [dict(e) for e in a]
    c = [dict(e) for e in a]
    assign_splits(a, seed=0)
    assign_splits(b, seed=0)
    assign_splits(c, seed=1)
    assert [e["split"] for e in a] == [e["split"] for e in b]
    assert [e["split"] for e in a] != [e["split"] for e in c]
