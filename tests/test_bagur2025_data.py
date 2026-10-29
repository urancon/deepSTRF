"""Real-data tests for Bagur2025Dataset. Need BAGUR2025_DATA (auto-marked slow)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

ROOT = os.environ.get("BAGUR2025_DATA")
pytestmark = pytest.mark.skipif(not ROOT, reason="set BAGUR2025_DATA to the release folder")

from deepSTRF.datasets.audio import Bagur2025Dataset                   # noqa: E402
from deepSTRF.datasets.audio import _bagur2025_prepare as bp           # noqa: E402
from deepSTRF.datasets.audio import _bagur2025_stims as bs             # noqa: E402


@pytest.mark.parametrize("area, method, N, R, T", [
    ("IC", "ephys", 442, 14, 300), ("TH", "ephys", 484, 15, 300),
    ("TH", "2p", 3969, 15, 47), ("IC", "2p", 5936, 13, 47), ("AC", "2p", 19414, 14, 47)])
def test_full_load(area, method, N, R, T):
    ds = Bagur2025Dataset(ROOT, areas=area, method=method)
    assert ds.N_neurons == N and len(ds.stim_meta) == 140
    assert ds.responses[5][0].shape == (R, T) and ds.stims[0].shape[-1] == T
    if method == "ephys":
        r = ds.responses[5][0]
        assert torch.equal(r, r.round()) and float(r.min()) >= 0.0          # raw counts


def test_duplicate_json_matches_audio():
    names = [m["name"] for m in bs.build_stim_table(bp.code_dir(ROOT), bp.wav_dir(ROOT))]
    assert bs.find_duplicates(bp.wav_dir(ROOT), names) == bs.load_duplicate_groups()


def test_stim_table_matches_author_groups():
    st = bs.build_stim_table(bp.code_dir(ROOT), bp.wav_dir(ROOT))
    first = {"blank": 1, "pure_tones": 2, "up_ramps": 35, "down_ramps": 56, "am": 77,
             "chirp_durations": 89, "chirps": 101}                         # SoundNomenclature.m
    last = {"blank": 1, "pure_tones": 34, "up_ramps": 55, "down_ramps": 76, "am": 88,
            "chirp_durations": 100, "chirps": 140}
    for i, m in enumerate(st, start=1):
        assert first[m["category"]] <= i <= last[m["category"]], m["name"]
    ramp = {m["name"]: m["frequency_khz"] for m in st}
    assert ramp["Up4+6+9+24kHz"] == pytest.approx([4.0, 6.26, 9.798, 24.0], abs=0.01)


def test_cochlear_response_starts_at_bin_100():
    """t = 0 is the sound onset: the model (latency < 5 ms) leaves its baseline in bin 100."""
    st = bs.build_stim_table(bp.code_dir(ROOT), bp.wav_dir(ROOT))
    ds = Bagur2025Dataset(ROOT, areas="IC", method="ephys", spec_backend="cochlear_model")
    tones = [s for s, m in enumerate(st) if m["category"] in ("pure_tones", "chirps")]
    x = np.mean([ds.stims[s][0, :580].abs().mean(0).numpy() for s in tones], axis=0)
    base = x[:95].mean()
    assert x[99] < 1.1 * base and x[100] > 1.5 * base


def test_wav_levels():
    w = bp.wav_dir(ROOT)
    db = lambda n: 20 * np.log10(np.sqrt(np.mean(bs.load_wav_int16(w, n) ** 2)))   # noqa: E731
    for f in (5, 9, 24):
        assert db(f"tono70dB_{f}kHz") - db(f"tono60dB_{f}kHz") == pytest.approx(10.0, abs=0.1)
        assert db(f"tono80dB_{f}kHz") - db(f"tono70dB_{f}kHz") == pytest.approx(10.0, abs=0.1)
    assert db("tono70dB_4kHz") - db("tono60dB_4kHz") == pytest.approx(21.1, abs=0.2)  # documented
