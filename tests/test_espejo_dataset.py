"""Tests for ``deepSTRF.datasets.audio.espejo``.

The structural / data-paradigm checks need actual recording archives —
they're skipped automatically when the local data dir is missing
(typical in CI). The cell-id parser is exercised in isolation.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


# Local-data sentinel. CI runners don't have the archives; loading tests
# skip when the directory is absent.
ESPEJO_LOCAL = os.path.join(
    os.path.dirname(__file__), "..",
    "deepSTRF", "datasets", "audio", "Espejo", "data",
)
HAS_VMN = os.path.isdir(os.path.join(ESPEJO_LOCAL, "A1_voc_mod_noise", "VMN"))
HAS_NAT = os.path.isdir(os.path.join(ESPEJO_LOCAL, "A1_natural_sounds", "NAT"))


# ============================================================
# Cell-id parsing (network-free, pure function)
# ============================================================

def test_parse_nat_cell_id():
    from deepSTRF.datasets.audio.espejo import _parse_espejo_cell_id

    out = _parse_espejo_cell_id("AMT003c-11-1")
    assert out == {
        "site": "AMT003c", "animal_id": "AMT",
        "channel": "11", "unit": "1",
    }


def test_parse_vmn_cell_id():
    from deepSTRF.datasets.audio.espejo import _parse_espejo_cell_id

    # 2-segment id (VMN's <site>-<chan_letter><unit_num>); unit ends up None
    # because the parser only splits a trailing -<digits> unit.
    out = _parse_espejo_cell_id("btn144a-c1")
    assert out["site"] == "btn144a"
    assert out["animal_id"] == "btn"
    assert out["channel"] == "c1"
    assert out["unit"] is None


def test_parse_unparseable_cell_id_returns_nones():
    from deepSTRF.datasets.audio.espejo import _parse_espejo_cell_id

    out = _parse_espejo_cell_id("totally-malformed-id-here")
    # site falls back to first dash-segment even when full parse fails
    assert out["site"] == "totally"


# ============================================================
# VMN end-to-end (fast: ~10 s on local data)
# ============================================================

@pytest.fixture(scope="module")
def vmn_dataset():
    if not HAS_VMN:
        pytest.skip("VMN data dir missing — skip integration test")
    from deepSTRF.datasets.audio.espejo import EspejoDataset
    return EspejoDataset(path=ESPEJO_LOCAL, stimuli="vmn")


def test_vmn_shape_invariants(vmn_dataset):
    ds = vmn_dataset
    assert ds.F == 2
    assert ds.species == "ferret"
    assert ds.behavioral_state == "awake-passive"
    assert ds.N_neurons > 0
    assert ds.N_neurons == len(ds.nrn_meta)
    S = len(ds.stim_meta)
    assert S > 0
    assert len(ds.stims) == S
    assert len(ds.responses) == S
    for s in range(min(3, S)):
        stim = ds.stims[s]
        assert stim.ndim == 3 and stim.shape[0] == 1 and stim.shape[1] == ds.F
        assert not stim.isnan().any(), "stim tensors must never contain NaN"
        for n in range(ds.N_neurons):
            r = ds.responses[s][n]
            assert r.ndim == 2
            if r.isnan().any():
                assert tuple(r.shape) == (1, 1), \
                    "NaN-sentinel must be (1, 1)"


def test_vmn_split_meta_fields(vmn_dataset):
    ds = vmn_dataset
    for m in ds.stim_meta:
        assert m["type"] == "vmn"
        assert m["split"] in ("estimation", "test")
        assert m["n_repeats"] >= 1
        assert m["duration_s"] > 0
        assert m["n_samples"] > 0
    # both splits should be non-empty
    splits = {m["split"] for m in ds.stim_meta}
    assert splits == {"estimation", "test"}


def test_vmn_subset_filter():
    if not HAS_VMN:
        pytest.skip("VMN data dir missing")
    from deepSTRF.datasets.audio.espejo import EspejoDataset

    ds_test = EspejoDataset(path=ESPEJO_LOCAL, stimuli="vmn", subset="test")
    assert all(m["split"] == "test" for m in ds_test.stim_meta)

    ds_est = EspejoDataset(path=ESPEJO_LOCAL, stimuli="vmn", subset="estimation")
    assert all(m["split"] == "estimation" for m in ds_est.stim_meta)

    # subsets sum to the 'all' total (or the 'all' total is at least their sum)
    ds_all = EspejoDataset(path=ESPEJO_LOCAL, stimuli="vmn", subset="all")
    assert len(ds_test.stim_meta) + len(ds_est.stim_meta) == len(ds_all.stim_meta)


def test_vmn_collate_produces_correct_shapes(vmn_dataset):
    from torch.utils.data import DataLoader
    from deepSTRF.utils.data import neural_collate

    loader = DataLoader(vmn_dataset, batch_size=2, shuffle=False,
                        collate_fn=neural_collate)
    stims, resps, mask, metas = next(iter(loader))
    assert stims.shape[:3] == (2, 1, vmn_dataset.F), stims.shape
    assert resps.shape[:2] == (2, vmn_dataset.N_neurons), resps.shape
    assert mask.shape == resps.shape
    assert not stims.isnan().any(), "stims must be NaN-free"
    assert len(metas) == 2


def test_vmn_nat_concat_rejected():
    """NAT (F=18) and VMN (F=2) cannot be concatenated."""
    if not (HAS_VMN and HAS_NAT):
        pytest.skip("Need both NAT and VMN data for this test")
    from deepSTRF.datasets.audio.espejo import EspejoDataset

    vmn = EspejoDataset(path=ESPEJO_LOCAL, stimuli="vmn", subset="test")
    nat = EspejoDataset(path=ESPEJO_LOCAL, stimuli="nat", subset="test")
    with pytest.raises(AssertionError, match="F mismatch"):
        _ = vmn + nat


# ============================================================
# Raw-waveform branch
# ============================================================

def test_nat_waveform_rejects_vmn():
    """VMN has no raw waveform (synthesized envelopes) -> return_waveform=True
    raises before any data access (no archive needed)."""
    from deepSTRF.datasets.audio.espejo import EspejoDataset
    with pytest.raises(AssertionError, match="stimuli='nat'"):
        EspejoDataset(stimuli="vmn", return_waveform=True)


def test_nat_waveform_rejects_bad_audio_fs():
    """A non-integer audio_fs * dt / 1000 grid is rejected at construction
    (before any data access)."""
    from deepSTRF.datasets.audio.espejo import EspejoDataset
    with pytest.raises(AssertionError, match="grid-lock|integer"):
        EspejoDataset(stimuli="nat", return_waveform=True, audio_fs=97656)


def test_nat_waveform_download_helper_uses_cache(tmp_path):
    """The waveform helper strips the STIM_ prefix and returns already-cached
    files without hitting the network."""
    from deepSTRF.datasets.audio.espejo import download_espejo_nat_waveforms
    wav_dir = tmp_path / "nat_waveforms"
    wav_dir.mkdir()
    (wav_dir / "00cat172_rec1_geese_excerpt1.wav").write_bytes(b"RIFFfake")
    out = download_espejo_nat_waveforms(
        ["STIM_00cat172_rec1_geese_excerpt1.wav"],
        dest=str(tmp_path), progress=False,
    )
    assert out["STIM_00cat172_rec1_geese_excerpt1.wav"].endswith(
        os.path.join("nat_waveforms", "00cat172_rec1_geese_excerpt1.wav")
    )


@pytest.mark.skipif(not HAS_NAT, reason="Espejo NAT local data missing")
def test_nat_waveform_branch():
    """NAT return_waveform=True hands out grid-locked raw waveforms (fetched
    from the bitbucket mirror, inset at the 0.5 s pre-stim offset); responses
    stay identical to cochleagram mode. subset='test' bounds the download."""
    from deepSTRF.datasets.audio.espejo import EspejoDataset
    kw = dict(path=ESPEJO_LOCAL, stimuli="nat", subset="test", dt_ms=10.0)
    ds_spec = EspejoDataset(**kw)
    ds_wav = EspejoDataset(return_waveform=True, **kw)

    assert ds_spec.return_waveform is False and ds_wav.return_waveform is True
    assert ds_spec.audio_fs is None and ds_spec.hop is None
    assert ds_wav.audio_fs == 44100 and ds_wav.hop == 441
    assert ds_wav.hearing_range_hz == (200.0, 40000.0)
    assert len(ds_wav.stims) == len(ds_spec.stims)

    ds_spec.validate()
    ds_wav.validate()
    for s in range(len(ds_wav.stims)):
        stim = ds_wav.stims[s]
        assert stim.dim() == 2 and stim.shape[0] == 1
        assert not stim.isnan().any()
        assert stim.shape[-1] == ds_spec.stims[s].shape[-1] * ds_wav.hop

    for s in range(len(ds_wav.stims)):
        for n in range(ds_wav.N_neurons):
            assert torch.allclose(ds_spec.responses[s][n], ds_wav.responses[s][n],
                                  equal_nan=True)


# ============================================================
# Native loader unit (independent of dataset class)
# ============================================================

def test_rasterize_pointprocess_basic():
    from deepSTRF.datasets.audio._espejo_native import rasterize_pointprocess

    # 3 spikes in [0, 0.1) at fs=100 -> bins 0, 5, 9
    spikes = np.array([0.001, 0.055, 0.099])
    out = rasterize_pointprocess(spikes, start_s=0.0, end_s=0.1, bin_s=0.01)
    assert out.shape == (10,)
    assert out[0] == 1.0 and out[5] == 1.0 and out[9] == 1.0
    assert out.sum() == 3.0


def test_rasterize_pointprocess_clamps_to_window():
    from deepSTRF.datasets.audio._espejo_native import rasterize_pointprocess

    # spike at 0.15 is past the window; should be dropped
    spikes = np.array([0.01, 0.15])
    out = rasterize_pointprocess(spikes, start_s=0.0, end_s=0.1, bin_s=0.01)
    assert out.sum() == 1.0


def test_rasterize_pointprocess_empty():
    from deepSTRF.datasets.audio._espejo_native import rasterize_pointprocess

    out = rasterize_pointprocess(np.array([]), 0.0, 0.1, 0.01)
    assert out.shape == (10,) and out.sum() == 0.0
