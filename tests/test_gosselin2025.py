"""Fast tests for Gosselin2025Dataset on a synthetic mini-release (no real data)."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest
import torch

from tests.gosselin2025_fake import write_fake_release
from deepSTRF.datasets.audio import _gosselin2025_stims as gs
from deepSTRF.datasets.audio import _gosselin2025_cochlear as gc
from deepSTRF.datasets.audio import _gosselin2025_prepare as gp
from deepSTRF.datasets.audio.gosselin2025 import Gosselin2025Dataset


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("gosselin")
    info = write_fake_release(root)
    with zipfile.ZipFile(root / "Soundset.zip") as z:
        z.extractall(root)
    return root, info


def test_read_sound_table_fills_categories_and_fixes_names(release):
    root, info = release
    rows = gs.read_sound_table(root / "Sound Parameters.csv")
    assert len(rows) == info["S"]
    assert rows[1]["category_raw"] == "PURE TONES"            # filled down
    assert rows[2]["name"] == "Ramps_down7050_16kHz"         # typo fixed
    assert rows[2]["intensity_db"] == [70.0, 50.0]
    assert rows[5]["intensity_db"] == [70.0, 50.0]           # "70-50" separator
    assert rows[3]["frequency_khz"] == []
    assert rows[4]["frequency_khz"] == [6.0, 12.0]
    assert [r["index_cn_ic"] for r in rows] == info["idx_cnic"]


def test_build_stim_table_derived_fields(release):
    root, _ = release
    st = gs.build_stim_table(root / "Sound Parameters.csv", root / "Soundset")
    by = {m["name"]: m for m in st}
    assert by["PT_70dB_8kHz"]["category"] == "pure_tones"
    assert by["PT_70dB_8kHz"]["level_db"] == 70.0
    assert by["Ramps_down7050_16kHz"]["level_db"] is None
    assert by["Ramps_down7050_16kHz"]["direction"] == "down"
    assert by["Chirpup_6kHz_to_12kHz_in_25_ms"]["direction"] == "up"
    assert by["WN_ramps_500_ms_Down_70dB_50dB"]["direction"] == "down"
    assert by["AM_10Hz_WN"]["am_rate_hz"] == 10.0
    assert by["AM_10Hz_WN"]["carrier"] == "white_noise"
    assert by["bird_inverted50dB"]["reversed"] is True
    assert by["bird_inverted50dB"]["source_clip"] == "bird"
    assert by["bird_50dB"]["decomposition_role"] == "original"
    assert by["bird_inverted50dB"]["decomposition_role"] is None
    assert by["bird_reconstructed"]["decomposition_role"] == "reconstructed"
    assert by["bird_Chirp_50to25kHz_0.02s_80dB"]["decomposition_role"] == "component"
    assert by["dassin_50dB"]["decomposition_role"] is None
    assert by["bird_reconstructed"]["duration_ms"] == pytest.approx(480.0)   # from the wav
    assert by["bird_reconstructed"]["duration_ms_csv"] == 460.0


def test_trial_waveform_places_onset_and_pads(release):
    root, _ = release
    wav = gs.load_wav_int16(root / "Soundset", "PT_50dB_8kHz")
    buf = gs.trial_waveform(wav, n_bins=180, hop=960)
    assert buf.shape == (180 * 960,)
    assert np.all(buf[:38400] == 0)
    np.testing.assert_array_equal(buf[38400:38400 + len(wav)], wav.astype(np.float64))
    assert np.all(buf[38400 + len(wav):] == 0)
    ac = gs.trial_waveform(wav, n_bins=24, hop=8352)          # 43.5 ms bins
    assert ac.shape == (24 * 8352,)


def test_logspec_keeps_absolute_level_and_floor(release):
    root, _ = release
    lo = gs.logspec_trial(gs.load_wav_int16(root / "Soundset", "PT_50dB_8kHz"),
                          n_bins=180, dt_ms=5.0, fmin=2000.0, fmax=64000.0, bins_per_octave=6)
    hi = gs.logspec_trial(gs.load_wav_int16(root / "Soundset", "PT_70dB_8kHz"),
                          n_bins=180, dt_ms=5.0, fmin=2000.0, fmax=64000.0, bins_per_octave=6)
    assert lo.shape == (31, 180) and lo.dtype == np.float32
    band = int(np.argmax(hi[:, 100]))                         # the 8 kHz band, mid-sound
    assert hi[band, 100] - lo[band, 100] == pytest.approx(20.0, abs=0.1)
    floor = gs.spec_floor_db(10.0)                            # window = 2 * dt
    assert lo[:, :30].min() == pytest.approx(floor) and lo[:, :30].max() == pytest.approx(floor)


def test_find_gain_copies_on_fake(release):
    root, _ = release
    names = [m["name"] for m in gs.build_stim_table(root / "Sound Parameters.csv", root / "Soundset")]
    groups = gs.find_gain_copies(root / "Soundset", names)
    assert groups == [["PT_50dB_8kHz", "PT_70dB_8kHz"], ["bird_50dB", "bird_70dB"]]


def test_shipped_gain_copies_json():
    groups = gs.load_gain_copies()
    assert len(groups) == 44 and all(len(g) == 2 for g in groups)
    assert ["PT_50dB_2kHz", "PT_70dB_2kHz"] in groups
    assert ["bird_inverted50dB", "bird_inverted70dB"] in groups


def test_ihc_groups_from_spont_resets():
    spont = np.array([1, 5, 20, 1, 5, 20, 0.5, 3, 2, 9])
    np.testing.assert_array_equal(gc.ihc_groups_from_spont(spont), [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])


def test_approx_cf():
    cf = gc.approx_cf_khz(51)
    assert cf[0] == pytest.approx(5.0) and cf[-1] == pytest.approx(50.0)
    assert np.allclose(np.diff(np.log(cf)), np.log(10) / 50)


def test_rebin_integer_and_fractional_with_spont_pad():
    x = np.zeros((1, 900))
    x[0, :200] = 2.0                  # spontaneous
    x[0, 200:700] = 10.0
    x[0, 700:] = 2.0
    r5 = gc.rebin_cochlear(x, 5.0, 180)
    assert r5.shape == (1, 180) and r5[0, 39] == 2.0 and r5[0, 40] == 10.0
    rac = gc.rebin_cochlear(x, 43.5, 24)
    assert rac.shape == (1, 24)
    assert rac[0, 0] == pytest.approx(2.0)
    assert rac[0, 4] == pytest.approx((26.0 * 2 + 17.5 * 10) / 43.5)   # bin 4 = 174..217.5 ms
    assert rac[0, 23] == pytest.approx(2.0)                           # 1000.5..1044 ms: padded


def test_overlap_weights_rows_sum_to_one():
    W = gc.overlap_weights(1044, 43.5, 24)
    np.testing.assert_allclose(W.sum(1), 1.0)


def test_prepare_and_load_ephys_reorders_and_rebins(release, tmp_path):
    root, info = release
    for area in ("CN", "IC"):
        gp.prepare_area(root, tmp_path, area)
        assert gp.is_prepared(tmp_path, area)
        x = gp.load_ephys(tmp_path, area, 5)
        n_units = 3 if area == "CN" else 2
        assert x.shape == (n_units, info["S"], 12, 180)
        for s in range(info["S"]):
            for n in range(n_units):
                reps = range(11) if (area == "CN" and n == 0) else range(12)
                for r in reps:
                    assert x[n, s, r].sum() == 1 and x[n, s, r].argmax() == 40 + s
        if area == "CN":
            assert x[0, :, 11].sum() == 0
        x1 = gp.load_ephys(tmp_path, area, 1)
        assert x1.sum() == x.sum()                         # counts conserved


def test_prepare_and_load_ac(release, tmp_path):
    root, info = release
    gp.prepare_area(root, tmp_path, "AC")
    a = gp.load_ac(tmp_path)
    assert a.shape == (3, info["S"], 12, 24) and a.dtype == np.float32
    for s in range(info["S"]):
        assert a[1, s, 0, 6] == pytest.approx(s + 0.01, abs=1e-2)
    assert np.isnan(a[:, :, 10:]).all() and np.isnan(a[2, 0, 9]).all()


def test_prepare_cochlear(release, tmp_path):
    root, info = release
    gp.prepare_cochlear(root, tmp_path)
    ihc, groups = gp.load_cochlear_ihc(tmp_path)
    assert ihc.shape == (4, 900, info["S"])
    np.testing.assert_array_equal(groups, np.repeat(np.arange(4), 3))
    assert ihc[2, 300, 0] == pytest.approx(np.mean([1.0, 5.0, 20.0]) + 300.0)


def test_required_files():
    f = gp.required_files(["IC"], cochlear=False, fibres=False)
    assert set(f) == {"Sound Parameters.csv", "Soundset.zip", "Info_IC.csv", "Dataset_IC.npy"}
    assert "Dataset_CC.npy" in gp.required_files(["AC"], cochlear=True, fibres=False)


@pytest.fixture(scope="module")
def ds_ephys(release, tmp_path_factory):
    root, _ = release
    return Gosselin2025Dataset(str(root), cache_dir=str(tmp_path_factory.mktemp("c1")))


def test_ephys_shapes_and_order(ds_ephys, release):
    _, info = release
    ds = ds_ephys
    assert ds.N_neurons == 5 and len(ds.stim_meta) == info["S"]
    assert [m["area"] for m in ds.nrn_meta] == ["CN"] * 3 + ["IC"] * 2
    assert ds.nrn_meta[3]["cell_id"] == "IC_0000" and ds.nrn_meta[4]["session_id"] == "IC-s2"
    for s in range(info["S"]):
        r = ds.responses[s][1]                         # CN neuron 1
        assert r.shape == (12, 180) and r.dtype == torch.float32
        assert int(r[0].argmax()) == 40 + s
    assert ds.responses[0][0][11].sum() == 0           # faded repeat kept as zeros
    assert ds.stims[0].shape == (1, 31, 180)
    assert ds.F == 31 and ds.dt == 5.0


def test_waveform_mode_levels(release, tmp_path):
    root, _ = release
    ds = Gosselin2025Dataset(str(root), areas=("IC",), return_waveform=True, cache_dir=str(tmp_path))
    assert ds.audio_fs == 192000 and ds.hop == 960
    w50, w70 = ds.stims[0], ds.stims[1]                # PT_50dB_8kHz, PT_70dB_8kHz
    assert w50.shape == (1, 180 * 960)
    ratio = w70.pow(2).sum().sqrt() / w50.pow(2).sum().sqrt()
    assert float(20 * torch.log10(ratio)) == pytest.approx(20.0, abs=0.05)


def test_ac_drops_nan_repeats(release, tmp_path):
    root, info = release
    ds = Gosselin2025Dataset(str(root), areas=("AC",), cache_dir=str(tmp_path))
    assert ds.dt == 43.5 and ds.recording_type == "2p-calcium" and ds.indicator == "GCaMP6s"
    assert ds.responses[3][1].shape == (10, 24)
    assert ds.responses[0][2].shape == (9, 24)
    assert not any(torch.isnan(r).any() for row in ds.responses for r in row)
    assert ds.stims[0].shape[-1] == 24


@pytest.mark.parametrize("kw, msg", [
    (dict(areas=("CN", "AC")), "AC"),
    (dict(areas=("XX",)), "unknown area"),
    (dict(dt_ms=3), "dt_ms"),
    (dict(areas=("AC",), dt_ms=5.0), "43.5"),
    (dict(audio_fs=48000, return_waveform=True), "audio_fs"),
    (dict(spec_backend="mel"), "spec_backend"),
])
def test_bad_arguments(release, tmp_path, kw, msg):
    root, _ = release
    with pytest.raises(ValueError, match=msg):
        Gosselin2025Dataset(str(root), cache_dir=str(tmp_path), **kw)


def test_missing_files_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset_IC.npy"):
        Gosselin2025Dataset(str(tmp_path), areas=("IC",))


def test_derived_neuron_meta(ds_ephys):
    m = ds_ephys.nrn_meta[0]
    assert set(m) >= {"cell_id", "area", "session", "session_id", "index_in_release",
                      "best_freq_khz", "spont_rate"}
    assert m["best_freq_khz"] == 8.0                   # only one 70 dB tone in the fake set
