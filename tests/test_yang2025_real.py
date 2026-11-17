"""Real-data checks for Yang2025 (slow; set $YANG2025_RAW to the raw release folder)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

RAW = os.environ.get("YANG2025_RAW")
pytestmark = pytest.mark.skipif(not RAW, reason="set $YANG2025_RAW to run")

EXPECTED_UNITS = {"Ti81": 49, "Li145": 81, "Bl122": 43, "Or61": 47}
EXPECTED_EPISODES = {"Ti81": 85, "Li145": 114, "Bl122": 49, "Or61": 90}


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    from deepSTRF.datasets.audio._yang2025_prepare import prepare_yang2025
    return prepare_yang2025(RAW, tmp_path_factory.mktemp("yang2025"))


def test_prepare_real(cache):
    index = json.loads((cache / "index.json").read_text())
    assert {b: len(u) for b, u in index["units"].items()} == EXPECTED_UNITS
    for bird, n in EXPECTED_EPISODES.items():
        eps = [e for e in index["episodes"] if e["bird"] == bird]
        # min-duration drop removes at most 2; the Or61 splice may add one piece
        assert n - 2 <= len(eps) <= n + 1, (bird, len(eps))
    for bird in EXPECTED_UNITS:      # no pre-roll reaches into another split
        eps = sorted((e for e in index["episodes"] if e["bird"] == bird), key=lambda e: e["onset_s"])
        for a, b in zip(eps[:-1], eps[1:]):
            if b["onset_s"] - a["offset_s"] < 1.0:
                assert a["split"] == b["split"], (a["bout_id"], b["bout_id"])
    for e in index["episodes"]:
        info = sf.info(str(cache / e["audio_file"]))
        assert abs(info.frames / info.samplerate - (e["audio_t1_s"] - e["audio_t0_s"])) < 1e-3
        assert e["audio_t0_s"] <= e["onset_s"] < e["offset_s"] <= e["audio_t1_s"]
    for bird in EXPECTED_UNITS:
        win = np.array([(e["audio_t0_s"], e["audio_t1_s"]) for e in index["episodes"]
                        if e["bird"] == bird])
        with np.load(cache / bird / "spikes.npz") as z:
            st = z["spike_times"]
            assert z["unit"].max() < EXPECTED_UNITS[bird]
        inside = ((st[:, None] >= win[:, 0]) & (st[:, None] < win[:, 1])).any(1)
        assert inside.all()


def test_loader_real(cache):
    import torch
    from deepSTRF.datasets.audio import Yang2025Dataset

    ds = Yang2025Dataset(path=cache)
    assert ds.N_neurons == sum(EXPECTED_UNITS.values()) == 220
    assert {m["split"] for m in ds.stim_meta} <= {"train", "val", "test"}
    split_of = {}
    for m in ds.stim_meta:
        assert split_of.setdefault(m["bout_id"], m["split"]) == m["split"]
    assert all(m["duration_s"] <= 5.0 + 0.3 + 0.005 for m in ds.stim_meta)
    assert all(torch.isfinite(s).all() for s in ds.stims)
    for bird, n in EXPECTED_UNITS.items():
        d = Yang2025Dataset(path=cache, birds=bird)
        assert d.N_neurons == n
        assert {m["bird"] for m in d.stim_meta} == {bird}
