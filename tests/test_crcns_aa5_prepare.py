"""Tests for the CRCNS-AA5 slimming step (``_crcns_aa5_prepare``).

Everything runs on a small synthetic site that mimics the release layout
(``<site>/PlaybackPkl/goodPlayback-e<E>-c<C>.pkl``, three pickled objects per
unit), so no real data is needed.
"""
from __future__ import annotations

import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from deepSTRF.datasets.audio import _crcns_aa5_prepare as P

SITE = "ZF4F_3t_190613_150928"
RNG = np.random.default_rng(0)


def _stim(fs, dur_s=0.4, amp=3000.0):
    """Integer-valued call-like wav in the -0.5 .. 4.5 s window (onset at t=0)."""
    t_axis = np.arange(int(5.0 * fs)) / fs - 0.5
    w = np.zeros_like(t_axis)
    on = (t_axis >= 0) & (t_axis < dur_s)
    tt = t_axis[on]
    w[on] = np.round(amp * np.sin(2 * np.pi * (2000 + 3000 * tt / dur_s) * tt) * np.hanning(on.sum()))
    return t_axis, w


def _mic(stim, fs_stim, shift_s=0.0):
    """Cage-mic stand-in at 25 kHz: the stimulus (delayed by shift_s) plus hum + noise."""
    s = P._to_mic_rate(stim, fs_stim)
    n = len(s)
    k = int(round(shift_s * 25000))
    m = np.zeros(n)
    if k >= 0:
        m[k:] = s[: n - k]
    else:
        m[:k] = s[-k:]
    t = np.arange(n) / 25000
    return 0.01 * m + 1e-3 * np.sin(2 * np.pi * 1200 * t) + 2e-4 * RNG.standard_normal(n)


STIMS = {
    # file: (fs, call_type, relation_short)
    "BlaBla0506_MAF_Te_8-1-9_fs25k.wav": (25000, "Te", "BlaB"),
    "stim3.wav": (32000, "So", "stim"),
}
# trialInd -> (file, start_time, mic shift in s)
PLAYBACKS = {
    3: ("BlaBla0506_MAF_Te_8-1-9_fs25k.wav", 100.0, 0.0),
    7: ("stim3.wav", 103.0, 0.0),                           # starts 3 s after trial 3
    9: ("BlaBla0506_MAF_Te_8-1-9_fs25k.wav", 118.0, 1.0),   # sound played 1 s late
    12: ("stim3.wav", 124.0, 0.0),
}
# unit name -> list of trialInd it kept (motion-artifact exclusion is per unit)
UNITS = {
    "goodPlayback-e10-c1.pkl": [3, 7, 9, 12],
    "goodPlayback-e11-c4.pkl": [3, 9, 12],
    "goodPlayback-e12-c2.pkl": [],                           # empty unit (nStim = 0)
}


def _spikes(unit, trial):
    if unit.endswith("c4.pkl") and trial == 12:
        return np.zeros(0)                                   # a kept trial with no spikes
    return np.sort(RNG.uniform(-0.5, 4.5, size=5 + trial))


def _write_unit(path: Path, unit: str, trials):
    info = {"Bird": "ZF4F", "Site": SITE, "Electrode": unit.split("-e")[1].split("-")[0],
            "Sort": np.int32(1), "RateThreshold": 1.0, "KDE_BW": 10.0,
            "nStim": len({PLAYBACKS[t][0] for t in trials}),
            "SpikeShape": np.linspace(-1, 1, 40), "SpikeStd": np.ones(40), "SpikeSNR": 6.5}
    abs_rows, rel_rows = {}, []
    for t in trials:
        f, start, _ = PLAYBACKS[t]
        abs_rows[t] = dict(file=f, relation_short=STIMS[f][2], call_type=STIMS[f][1],
                           start_time=start, duration=2.5, stop_time=start + 2.5,
                           file_path="x", spikes=None, On=True)
    dfabs = pd.DataFrame.from_dict(abs_rows, orient="index")
    for f, (fs, ct, rel) in STIMS.items():
        ts = [t for t in trials if PLAYBACKS[t][0] == f]
        if not ts:
            continue
        t_axis, w = _stim(fs)
        rel_rows.append(dict(
            file=f, relation_short=rel, call_type=ct, nTrials=len(ts), trialInd=ts,
            tStim=t_axis, stimWav=w, micWav=[_mic(w, fs, PLAYBACKS[t][2]) for t in ts],
            spikeTimes=[_spikes(unit, t) for t in ts],
            tKDE=np.arange(-0.5, 4.5, 1e-3), spikeKDE=np.zeros(5000),
            tMic=np.arange(125000) / 25000 - 0.5))
    dfrel = pd.DataFrame(rel_rows, columns=["file", "relation_short", "call_type", "nTrials",
                                            "trialInd", "tStim", "stimWav", "micWav",
                                            "spikeTimes", "tKDE", "spikeKDE", "tMic"])
    with open(path, "wb") as fh:
        for obj in (info, dfabs, dfrel):
            pickle.dump(obj, fh)


@pytest.fixture(scope="module")
def raw_site(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("aa5_raw")
    d = root / "ZF4F" / SITE / "PlaybackPkl"
    d.mkdir(parents=True)
    for unit, trials in UNITS.items():
        _write_unit(d / unit, unit, trials)
    (d / "._goodPlayback-e10-c1.pkl").write_bytes(b"\x00\x05\x16\x07")   # macOS resource fork
    return root


@pytest.fixture(scope="module")
def cache(raw_site, tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("aa5_cache")
    P.prepare_aa5(raw_site, dest, progress=False)
    return dest


# ---------------------------------------------------------------------------

def test_parse_site():
    d = P.parse_site("ZF4F_6_5_190618_110932")               # no trailing "t"
    assert d == dict(bird="ZF4F", depth_turns=6.5, depth_um=1625.0, date="190618", time="110932")
    assert P.parse_site("ZF6M_10_5t_190808_102150")["depth_turns"] == 10.5
    with pytest.raises(ValueError):
        P.parse_site("not_a_site")


def test_manifest_covers_all_release_sites():
    assert len(P.AA5_SITES) == 50
    assert {P.parse_site(s)["bird"] for s in P.AA5_SITES} == set(P.AA5_BIRDS)


def test_unpickler_maps_removed_pandas_index_module():
    if P._NEED_INDEX_SHIM:
        u = P._AA5Unpickler(open(__file__, "rb"))
        assert u.find_class("pandas.core.indexes.numeric", "Int64Index") is pd.Index


def test_site_layout(cache):
    site_dir = cache / "ZF4F" / SITE
    assert {p.name for p in site_dir.iterdir()} == {"units.json", "playbacks.json",
                                                   "spikes.npz", "manifest.json"}
    m = json.loads((site_dir / "manifest.json").read_text())
    assert m["cache_version"] == P.CACHE_VERSION
    assert (m["n_units"], m["n_playbacks"]) == (3, 4)
    assert m["n_trial_records"] == sum(len(t) for t in UNITS.values())
    assert not list((cache / "ZF4F").glob(".*.tmp"))


def test_units_metadata(cache):
    units = json.loads((cache / "ZF4F" / SITE / "units.json").read_text())
    by_file = {u["file"]: u for u in units}
    assert set(by_file) == set(UNITS)                        # resource fork skipped
    u = by_file["goodPlayback-e11-c4.pkl"]
    assert u["cell_id"] == f"{SITE}_e11-c4"
    assert (u["electrode"], u["cluster"], u["sort"]) == (11, 4, 1)
    assert u["spike_snr"] == 6.5 and len(u["spike_shape"]) == 40
    assert by_file["goodPlayback-e12-c2.pkl"]["n_stim"] == 0


def test_spikes_roundtrip(cache):
    units = json.loads((cache / "ZF4F" / SITE / "units.json").read_text())
    z = np.load(cache / "ZF4F" / SITE / "spikes.npz")
    got = {}
    for u, t, a, b in zip(z["unit"], z["trial"], z["start"], z["stop"]):
        got[(units[u]["file"], int(t))] = z["spike_times"][a:b]
    for unit, trials in UNITS.items():
        for t in trials:
            assert (unit, t) in got
    assert len(got[("goodPlayback-e11-c4.pkl", 12)]) == 0  # kept trial without spikes
    assert np.all(np.diff(got[("goodPlayback-e10-c1.pkl", 7)]) >= 0)


def test_playbacks_and_mic_alignment(cache):
    pbs = {p["trial"]: p for p in json.loads((cache / "ZF4F" / SITE / "playbacks.json").read_text())}
    assert set(pbs) == set(PLAYBACKS)
    for t, (f, start, shift) in PLAYBACKS.items():
        p = pbs[t]
        assert p["file"] == f and p["start_time"] == start
        assert abs(p["mic_offset_ms"] - 1e3 * shift) < 5.0, (t, p["mic_offset_ms"])
        assert p["mic_peak"] > 0.5
    assert pbs[3]["next_onset_s"] == pytest.approx(3.0)
    assert "next_onset_s" not in pbs[12]                      # last playback


def test_stimuli_lossless(cache):
    index = json.loads((cache / "stimuli" / "index.json").read_text())
    assert set(index) == set(STIMS)
    for f, (fs, _, _) in STIMS.items():
        w, sr = sf.read(cache / "stimuli" / f, dtype="int16")
        assert sr == fs == index[f]["fs"] and index[f]["t_start_s"] == -0.5
        np.testing.assert_array_equal(w, _stim(fs)[1].astype(np.int16))


def test_idempotent_and_overwrite(raw_site, cache):
    mf = cache / "ZF4F" / SITE / "manifest.json"
    before = mf.stat().st_mtime_ns
    assert P.prepare_aa5(raw_site, cache, progress=False) == [SITE]
    assert mf.stat().st_mtime_ns == before
    P.prepare_aa5(raw_site, cache, overwrite=True, mic_alignment_check=False, progress=False)
    assert mf.stat().st_mtime_ns != before
    pb = json.loads((cache / "ZF4F" / SITE / "playbacks.json").read_text())[0]
    assert "mic_offset_ms" not in pb


def test_tar_source_matches_folder(raw_site, cache, tmp_path):
    archive = tmp_path / f"{SITE}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(raw_site / "ZF4F" / SITE, arcname=SITE)
    dest = tmp_path / "cache"
    P.prepare_site(archive, dest, mic_alignment_check=False, progress=False)
    a = np.load(dest / "ZF4F" / SITE / "spikes.npz")
    b = np.load(cache / "ZF4F" / SITE / "spikes.npz")
    order_a = np.lexsort((a["trial"], a["unit"]))
    units_a = json.loads((dest / "ZF4F" / SITE / "units.json").read_text())
    units_b = json.loads((cache / "ZF4F" / SITE / "units.json").read_text())
    assert sorted(u["file"] for u in units_a) == sorted(u["file"] for u in units_b)
    assert len(a["spike_times"]) == len(b["spike_times"]) and len(order_a) == len(b["trial"])


def test_truncated_archive_is_skipped(tmp_path):
    src = tmp_path / "raw" / "ZF4F"
    src.mkdir(parents=True)
    (src / f"{SITE}.tar.gz").write_bytes(b"partial download")
    with pytest.warns(UserWarning, match="incomplete download"):
        assert P.prepare_aa5(tmp_path / "raw", tmp_path / "cache", progress=False) == []


def test_unknown_site_rejected(raw_site, tmp_path):
    with pytest.raises(ValueError, match="Unknown CRCNS-AA5 site"):
        P.prepare_aa5(raw_site, tmp_path, sites=["ZF9X_1t_000000_000000"], progress=False)


def test_download_aa5_resolves_paths_from_file_list(raw_site, tmp_path, monkeypatch):
    from deepSTRF.utils import data_download as dd
    archive = tmp_path / "remote" / f"{SITE}.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(raw_site / "ZF4F" / SITE, arcname=SITE)
    fetched = []
    monkeypatch.setattr(dd, "crcns_file_list", lambda ds, **kw: {
        f"some/remote/dir/{SITE}.tar.gz": archive.stat().st_size, "docs/readme.pdf": 10})

    def fake_download(fp, dest, **kw):
        fetched.append(fp)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(archive.read_bytes())
        return Path(dest)
    monkeypatch.setattr(dd, "crcns_download", fake_download)

    cache = tmp_path / "cache"
    with pytest.warns(UserWarning, match="expected"):          # synthetic size != release size
        done = P.download_aa5(cache, sites=[SITE], mic_alignment_check=False, progress=False)
    assert done == [SITE] and fetched == [f"aa-5/some/remote/dir/{SITE}.tar.gz"]
    assert not list((cache / "_archives").rglob("*.tar.gz"))     # archive deleted after slimming
    # second call: nothing left to do, no network
    monkeypatch.setattr(dd, "crcns_file_list", lambda *a, **k: pytest.fail("network used"))
    assert P.download_aa5(cache, sites=[SITE], progress=False) == [SITE]
