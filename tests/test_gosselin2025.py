"""Fast tests for Gosselin2025Dataset on a synthetic mini-release (no real data)."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

from tests.gosselin2025_fake import write_fake_release
from deepSTRF.datasets.audio import _gosselin2025_stims as gs


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
