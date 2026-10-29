"""Real-data tests for Gosselin2025Dataset. Need GOSSELIN2025_DATA (auto-marked slow).

The Fig 3I test also needs the figure source data of the Zenodo record
(``Fig3_data.ods``, version 17271813) under ``$GOSSELIN2025_DATA/v2_extras/``.
"""
from __future__ import annotations

import os
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = os.environ.get("GOSSELIN2025_DATA")
pytestmark = pytest.mark.skipif(not ROOT, reason="set GOSSELIN2025_DATA to the release folder")

from deepSTRF.datasets.audio import Gosselin2025Dataset
from deepSTRF.datasets.audio import _gosselin2025_stims as gs

_ODS = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
# Fig 3I categories, as CSV rows (the ``PopFRSlc`` of the authors' Codes.py)
_FIG3I_CATS = {"PT70": range(14, 28), "Chords": range(54, 102), "AM": range(102, 150),
               "WN": range(170, 200), "Complex70": range(262, 277)}


def _ods_rows(path, sheet):
    root = ET.fromstring(zipfile.ZipFile(path).read("content.xml"))
    for t in root.iter(_ODS + "table"):
        if t.get(_ODS + "name") == sheet:
            return [["".join(c.itertext()) for c in r.iter(_ODS + "table-cell")]
                    for r in t.iter(_ODS + "table-row")]
    raise KeyError(sheet)


def _stack(rows):
    """(N, R_max, T) float array from one sound's responses, NaN-padded on R."""
    R = max(r.shape[0] for r in rows)
    out = np.full((len(rows), R, rows[0].shape[-1]), np.nan, np.float32)
    for n, r in enumerate(rows):
        out[n, :r.shape[0]] = r.numpy()
    return out


def _pop_fr(ds, base_bins, resp_bins):
    """Population rate per sound, as in Codes.py: each neuron scaled by its max
    over all sounds, baseline subtracted per trial, averaged over neurons, time
    window and repeats, then divided by the mean over the 70-dB complex sounds."""
    mx = np.full(ds.N_neurons, -np.inf, np.float32)
    for s in range(len(ds.stim_meta)):
        mx = np.fmax(mx, np.nanmax(_stack(ds.responses[s]), axis=(1, 2)))
    out = {}
    for cat, rows in _FIG3I_CATS.items():
        vals = []
        for s in rows:
            x = _stack(ds.responses[s]) / mx[:, None, None]
            with warnings.catch_warnings():          # NaN-padded missing AC repeats
                warnings.simplefilter("ignore", RuntimeWarning)
                base = np.nanmean(x[:, :, base_bins], axis=2, keepdims=True)
            vals.append(np.nanmean((x - base)[:, :, resp_bins]))
        out[cat] = np.array(vals)
    norm = out["Complex70"].mean()
    return {c: v / norm for c, v in out.items()}


@pytest.mark.parametrize("areas, dt_ms, N, T", [(("CN",), 5, 1421, 180),     # 1 ms CN is ~19 GB
                                                (("IC",), None, 551, 900),     # default: native 1 ms
                                                (("AC",), None, 4217, 24)])
def test_full_load(areas, dt_ms, N, T):
    ds = Gosselin2025Dataset(ROOT, areas=areas, dt_ms=dt_ms)
    assert ds.N_neurons == N and len(ds.stim_meta) == 307 and ds.stims[0].shape[-1] == T
    assert not any(torch.isnan(r).any() for row in ds.responses for r in row
                   if tuple(r.shape) != (1, 1))


def test_gain_copy_json_matches_audio():
    st = gs.build_stim_table(f"{ROOT}/Sound Parameters.csv", f"{ROOT}/soundset/Soundset")
    names = [m["name"] for m in st]
    assert gs.find_gain_copies(f"{ROOT}/soundset/Soundset", names) == gs.load_gain_copies()


def test_waveform_mode_validates():
    ds = Gosselin2025Dataset(ROOT, areas=("IC",), dt_ms=5, return_waveform=True)
    assert ds.stims[0].shape == (1, 180 * 960)


@pytest.mark.parametrize("area, dt_ms, base_bins, resp_bins, min_r", [
    ("AC", None, slice(0, 4), slice(5, 16), 0.999),      # 43.5 ms bins
    ("IC", 1, slice(0, 200), slice(200, 700), 0.98),     # 1 ms bins
])
def test_reproduces_fig3i_population_rate(area, dt_ms, base_bins, resp_bins, min_r):
    ods = Path(ROOT) / "v2_extras" / "Fig3_data.ods"
    if not ods.exists():
        pytest.skip(f"{ods} not found (figure source data of Zenodo 17271813)")
    rows = [r for r in _ods_rows(ods, "Fig 3I")[1:] if len(r) >= 4 and r[0] != ""]
    ref = {c: np.array([float(r[1].replace(",", ".")) for r in rows
                        if r[2] == c and r[3] == area]) for c in _FIG3I_CATS}
    ds = Gosselin2025Dataset(ROOT, areas=(area,), dt_ms=dt_ms)
    ours = _pop_fr(ds, base_bins, resp_bins)
    a = np.concatenate([ours[c] for c in _FIG3I_CATS])
    b = np.concatenate([ref[c] for c in _FIG3I_CATS])
    assert len(a) == len(b) == 155
    assert np.corrcoef(a, b)[0, 1] >= min_r
