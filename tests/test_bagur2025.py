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
