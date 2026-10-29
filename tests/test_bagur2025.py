"""Fast tests for Bagur2025Dataset on a synthetic mini-release (no real data)."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

from tests.bagur2025_fake import write_fake_release
from deepSTRF.datasets.audio import _bagur2025_stims as bs


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("bagur")
    info = write_fake_release(root)
    for z, d in (("Code.zip", "code"), ("Sounds.zip", "sounds")):
        with zipfile.ZipFile(root / z) as f:
            f.extractall(root / d)
    return root, info


def _table(root):
    return bs.build_stim_table(root / "code", root / "sounds" / "Sounds")


def test_stim_table_categories_levels(release):
    root, info = release
    st = _table(root)
    assert [m["name"] for m in st] == info["names"]
    by = {m["name"]: m for m in st}
    assert by["blank"]["category"] == "blank" and by["blank"]["level_db"] is None
    assert by["tono80dB_4kHz"]["category"] == "pure_tones" and by["tono80dB_4kHz"]["level_db"] == 80.0
    assert by["tono70dB_9kHz"]["frequency_label_khz"] == [9.0]
    assert by["Sin7Hz9kHz"]["category"] == "am" and by["Sin7Hz9kHz"]["am_rate_hz"] == 7.0
    assert by["Sin7Hz9kHz"]["carrier"] == 9.0 and by["Sin7Hz9kHz"]["level_db"] == 70.0
    assert by["Sin3HzWhitenoise"]["carrier"] == "white_noise"
    assert by["Sin3HzWhitenoise"]["frequency_khz"] == []
    assert by["ChirpUp4kHz60dB100ms"]["category"] == "chirp_durations"
    assert by["ChirpUp4kHz60dB100ms"]["level_db"] == 60.0
    assert by["ChirpUp4kHz60dB100ms"]["duration_ms"] == pytest.approx(100.0)
    assert by["ChirpDownclose9to4kHz60dB"]["category"] == "chirps"
    assert by["ChirpDownclose9to4kHz60dB"]["direction"] == "down"
    assert by["ChirpDownclose9to4kHz60dB"]["frequency_khz"] == [9.0, 4.0]
    assert by["Up4+9kHz"]["category"] == "up_ramps" and by["Down4+9kHz"]["direction"] == "down"
    assert [m["index_in_release"] for m in st] == list(range(info["S"]))
    assert {m["category"] for m in st} == set(bs.CATEGORIES)


def test_ramp_tone_set_from_audio(release):
    root, _ = release
    by = {m["name"]: m for m in _table(root)}
    assert by["Up4+9kHz"]["frequency_khz"] == pytest.approx([4.0, 9.0], abs=0.01)
    assert by["Down4+9kHz"]["frequency_khz"] == pytest.approx([4.0, 9.0], abs=0.01)
    assert by["Up4+9kHz"]["frequency_label_khz"] == [4.0, 9.0]
    assert by["tono70dB_24kHz"]["frequency_khz"] == pytest.approx([24.0], abs=0.01)
    assert by["Sin7Hz9kHz"]["frequency_khz"] == pytest.approx([9.0], abs=0.01)    # sidebands merged


def test_find_duplicates_on_fake(release):
    root, info = release
    groups = bs.find_duplicates(root / "sounds" / "Sounds", info["names"])
    assert ["tono60dB_4kHz", "tono70dB_4kHz", "tono80dB_4kHz"] in groups
    assert ["ChirpUp4kHz60dB500ms", "ChirpUpclose4to9kHz60dB", "ChirpUpclose4to9kHz80dB"] in groups
    assert not any("Up4+9kHz" in g for g in groups)                # time reversal: not a copy
    assert not any("blank" in g for g in groups)


def test_shipped_duplicate_json():
    groups = bs.load_duplicate_groups()
    sizes = sorted(len(g) for g in groups)
    assert len(groups) == 31 and sizes == [2] * 16 + [3] * 15
    dur = [g for g in groups if any(n.endswith("500ms") for n in g)]
    assert len(dur) == 4 and all(len(g) == 3 for g in dur)


# ---------------------------------------------------------------- Task 3: cache
from deepSTRF.datasets.audio import _bagur2025_prepare as bp   # noqa: E402


def test_hdf5_axes_order(release, tmp_path):
    root, info = release
    bp.prepare_population(root, tmp_path, "AC", "2p")
    a = bp.load_population(tmp_path, "AC", "2p")
    assert a.shape == (3, info["S"], 14, 47) and a.dtype == np.float32
    import h5py
    with h5py.File(root / "Data_AC.mat") as h:                     # MATLAB Data(n,t,s,r) = h5[r,s,t,n]
        src = h["Data"][()]
    np.testing.assert_allclose(a, src.transpose(3, 1, 0, 2), atol=2e-3)
    assert a[2, 11, 13, 40] == pytest.approx(src[13, 11, 40, 2], abs=2e-3)
    np.testing.assert_allclose(bp.load_twop_time(tmp_path), -0.4683 + 0.0317 * np.arange(47))


def test_count_recovery_exact(release, tmp_path):
    root, info = release
    bp.prepare_population(root, tmp_path, "IC", "ephys")
    c = bp.load_population(tmp_path, "IC", "ephys")               # (N, S, R, 300)
    planted = info["counts"]["IC"].transpose(0, 2, 3, 1)          # (N, T, S, R) -> (N, S, R, T)
    np.testing.assert_array_equal(c, planted)


def test_count_recovery_rejects_non_integer():
    x = np.zeros((1, 300, 2, 3), np.float32)
    x[0, 150, 1, 2] = 1.0
    x[0, :, 0, 1] += 0.3 + np.linspace(0, 0.2, 300)                # residuals not integer
    with pytest.raises(ValueError, match="integer"):
        bp.recover_counts(x, time_axis=1)


def test_anatomy_keys(release):
    root, _ = release
    ac = bp.load_anatomy(root, "AC", "2p")
    assert ac["subfield"] == ["AI", None, "SRAF"]
    assert ac["x_um"][0] == 850.0 and ac["depth_um"][2] == 450.0
    ic = bp.load_anatomy(root, "IC", "ephys")
    assert ic["subdivision"] == ["IC-C", None, "outside IC"]
    assert ic["ap"][0] == pytest.approx(5.3) and np.isnan(ic["ml"][1])
    assert bp.load_anatomy(root, "TH", "2p") is None and bp.load_anatomy(root, "TH", "ephys") is None


def test_arrays_not_required_after_cache(release, tmp_path):
    root, _ = release
    f = bp.required_files(("IC",), "ephys", cochlear=False)
    assert "Data_ICE.mat" in f and "AnatInfo_ICE.mat" in f and "Sounds.zip" in f
    g = bp.required_files(("IC",), "ephys", cochlear=False, prepared=("IC-ephys",),
                          have_wavs=True, have_code=True)
    assert "Data_ICE.mat" not in g and "Sounds.zip" not in g and "Code.zip" not in g
    assert "AnatInfo_ICE.mat" in g                                 # metadata read at load time
    assert "Data_CC.mat" in bp.required_files(("AC",), "2p", cochlear=True)


# ------------------------------------------------------------- Task 4: cochlea
def test_cochlear_mean_and_bf(release, tmp_path):
    root, info = release
    from scipy.io import loadmat
    st = _table(root)
    bp.prepare_cochlear(root, tmp_path, st)
    cc, bf, cf = bp.load_cochlear(tmp_path)
    raw = loadmat(root / "Data_CC.mat")["Data"]                        # (F, T, S, 15)
    assert cc.shape == (12, 300, info["S"]) and cc.dtype == np.float32
    np.testing.assert_allclose(cc, raw.mean(-1), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(bf, [4.0] * 4 + [9.0] * 4 + [24.0] * 4, atol=0.01)
    assert np.all(np.diff(cf) > 0)                                     # tonotopic label
    assert cf[0] == pytest.approx(4.0, rel=0.25) and cf[-1] == pytest.approx(24.0, rel=0.25)


# ------------------------------------------------------------- Task 5: class
import torch                                                            # noqa: E402

from deepSTRF.datasets.audio.bagur2025 import Bagur2025Dataset        # noqa: E402


@pytest.fixture(scope="module")
def ds_ephys(release, tmp_path_factory):
    root, _ = release
    return Bagur2025Dataset(str(root), areas=("IC", "TH"), method="ephys",
                            cache_dir=str(tmp_path_factory.mktemp("ce")))


@pytest.fixture(scope="module")
def ds_2p(release, tmp_path_factory):
    root, _ = release
    return Bagur2025Dataset(str(root), areas=("AC", "TH", "IC"), method="2p",
                            cache_dir=str(tmp_path_factory.mktemp("c2")))


def test_ephys_counts_and_shapes(ds_ephys, release, tmp_path):
    root, info = release
    ds = ds_ephys
    assert ds.N_neurons == 5 and ds.dt == 5.0 and len(ds.stim_meta) == info["S"]
    assert [m["area"] for m in ds.nrn_meta] == ["IC"] * 3 + ["TH"] * 2
    r = ds.responses[3][1]                                          # IC unit 1, sound 3
    assert r.shape == (14, 300) and torch.equal(r, r.round())
    np.testing.assert_array_equal(r.numpy(), info["counts"]["IC"][1, :, 3, :].T)
    assert ds.responses[0][4].shape == (15, 300)                     # TH: 15 repeats
    coarse = Bagur2025Dataset(str(root), areas="IC", method="ephys", dt_ms=25,
                              cache_dir=str(tmp_path))
    assert coarse.responses[3][1].shape == (14, 60)
    assert float(coarse.responses[3][1].sum()) == float(r.sum())


def test_2p_shapes_and_values(ds_2p, release):
    root, info = release
    ds = ds_2p
    assert ds.N_neurons == 3 + 2 + 4 and ds.responses[0][0].shape == (14, 47)
    assert ds.responses[0][3].shape == (15, 47) and ds.responses[0][5].shape == (13, 47)
    assert ds.dt == pytest.approx(6086 / 192) and ds.recording_type == "2p-calcium"
    import h5py
    with h5py.File(root / "Data_TH.mat") as h:
        src = h["Data"][()]                                          # (R, S, T, N)
    np.testing.assert_allclose(ds.responses[7][4].numpy(), src[:, 7, :, 1], atol=2e-3)


def test_metadata_keys_gosselin_lacked(ds_2p, ds_ephys):
    ac = [m for m in ds_2p.nrn_meta if m["area"] == "AC"]
    assert [m["subfield"] for m in ac] == ["AI", None, "SRAF"]
    assert ac[0]["x_um"] == 850.0 and ac[0]["y_um"] == 1150.0 and ac[2]["depth_um"] == 450.0
    assert {m["unit_type"] for m in ds_2p.nrn_meta if m["area"] == "TH"} == {"axon bouton"}
    assert {m["unit_type"] for m in ds_2p.nrn_meta if m["area"] != "TH"} == {"soma ROI"}
    assert all(m["spont_rate"] is None for m in ds_2p.nrn_meta)
    ic = [m for m in ds_ephys.nrn_meta if m["area"] == "IC"]
    assert [m["subdivision"] for m in ic] == ["IC-C", None, "outside IC"]
    assert ic[0]["ap"] == pytest.approx(5.3) and ic[1]["ml"] is None
    assert {m["unit_type"] for m in ds_ephys.nrn_meta} == {"single unit"}
    assert ds_ephys.nrn_meta[3]["cell_id"] == "TH-ephys_00000"
    assert all(isinstance(m["spont_rate"], float) for m in ds_ephys.nrn_meta)
    assert set(ds_ephys.nrn_meta[0]) >= {"cell_id", "area", "method", "recording_type", "unit_type",
                                        "index_in_release", "best_freq_khz", "spont_rate"}


@pytest.mark.parametrize("kw, err, msg", [
    (dict(areas="AC", method="ephys"), ValueError, "AC"),
    (dict(areas="IC", method="calcium"), ValueError, "method"),
    (dict(areas="XX", method="2p"), ValueError, "unknown area"),
    (dict(areas="IC", method="ephys", dt_ms=3), ValueError, "dt_ms"),
    (dict(areas="IC", method="2p", dt_ms=5), ValueError, "dt_ms"),
    (dict(areas="IC", method="2p", smooth=True), ValueError, "already smoothed"),
    (dict(method="2p"), TypeError, "areas"),
    (dict(areas="IC"), TypeError, "method"),
])
def test_bad_arguments(release, tmp_path, kw, err, msg):
    root, _ = release
    with pytest.raises(err, match=msg):
        Bagur2025Dataset(str(root), cache_dir=str(tmp_path), **kw)


def test_areas_plain_string(release, tmp_path):
    root, _ = release
    ds = Bagur2025Dataset(str(root), areas="TH", method="ephys", cache_dir=str(tmp_path))
    assert ds.areas == ("TH",) and ds.N_neurons == 2 and ds.method == "ephys"


def test_smooth_opt_in_ephys(release, ds_ephys, tmp_path):
    root, _ = release
    sm = Bagur2025Dataset(str(root), areas=("IC", "TH"), method="ephys", smooth=True,
                          cache_dir=str(tmp_path))
    r, q = ds_ephys.responses[3][1], sm.responses[3][1]
    assert q.shape == r.shape and not torch.equal(q, r) and q.max() < r.max()


# ----------------------------------------------------------- Task 6: stimuli
def _first_loud_bin(spec, margin_db=10.0):
    x = spec[0].numpy()
    return int(np.flatnonzero((x.max(axis=0) - x.min()) > margin_db)[0])


def test_ephys_onset_bin_100(ds_ephys):
    s = [m["name"] for m in ds_ephys.stim_meta].index("tono70dB_9kHz")
    assert _first_loud_bin(ds_ephys.stims[s]) == 100


def test_2p_onset_frame_matches_t(ds_2p):
    s = [m["name"] for m in ds_2p.stim_meta].index("tono70dB_9kHz")
    t, dt = ds_2p.twop_frame_times_s * 1000.0, ds_2p.dt
    expected = int(np.flatnonzero(t + dt / 2 > 0)[0])                 # first frame reaching +0 ms
    assert _first_loud_bin(ds_2p.stims[s]) == expected


def test_levels_kept(release, ds_ephys, tmp_path):
    names = [m["name"] for m in ds_ephys.stim_meta]
    a, b = ds_ephys.stims[names.index("ChirpUpclose4to9kHz60dB")], ds_ephys.stims[names.index("ChirpUpclose4to9kHz80dB")]
    assert float(b.max() - a.max()) == pytest.approx(20.0, abs=0.5)
    root, _ = release
    w = Bagur2025Dataset(str(root), areas="IC", method="ephys", return_waveform=True,
                         cache_dir=str(tmp_path))
    ra = w.stims[names.index("ChirpUpclose4to9kHz60dB")].pow(2).mean().sqrt()
    rb = w.stims[names.index("ChirpUpclose4to9kHz80dB")].pow(2).mean().sqrt()
    assert float(20 * torch.log10(rb / ra)) == pytest.approx(20.0, abs=0.05)
    assert w.audio_fs == 192000 and w.stims[0].shape == (1, 300 * 960)


def test_blank_is_floor(ds_ephys):
    x = ds_ephys.stims[[m["name"] for m in ds_ephys.stim_meta].index("blank")]
    assert float(x.max() - x.min()) == 0.0


def test_cochlear_input_shapes_both_time_bases(release, tmp_path):
    root, info = release
    e = Bagur2025Dataset(str(root), areas="IC", method="ephys", spec_backend="cochlear_model",
                         cache_dir=str(tmp_path))
    assert e.stims[0].shape == (1, 12, 300) and e.F == 12
    np.testing.assert_allclose(e.cochlear_bf_khz, [4.0] * 4 + [9.0] * 4 + [24.0] * 4, atol=0.01)
    assert np.all(np.diff(e.cochlear_cf_khz) > 0)
    s = info["names"].index("tono70dB_9kHz")
    assert float(e.stims[s][0, 5, 110]) > float(e.stims[s][0, 0, 110]) + 100     # BF fibre driven
    p = Bagur2025Dataset(str(root), areas="IC", method="2p", spec_backend="cochlear_model",
                         cache_dir=str(tmp_path))
    assert p.stims[0].shape == (1, 12, 47)
    with pytest.raises(ValueError, match="exclude"):
        Bagur2025Dataset(str(root), areas="IC", method="ephys", spec_backend="cochlear_model",
                         return_waveform=True, cache_dir=str(tmp_path))


def test_waveform_grid_lock_2p(release, tmp_path):
    root, _ = release
    w = Bagur2025Dataset(str(root), areas="TH", method="2p", return_waveform=True,
                         cache_dir=str(tmp_path))
    assert w.hop == 6086 and w.stims[0].shape == (1, 47 * 6086)
