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
