"""Tests for ``deepSTRF.datasets.audio.crcns_ac1``.

CRCNS-AC1 is auth-walled (free CRCNS account required), so the
structural / end-to-end checks here are skipped automatically in CI
when the local archive is missing. The unit-level checks of the
spectrogram primitive, the response-cleanup helpers, and the Asari
sequence parser run network-free.

Override the local path with ``$CRCNS_AC1_DATA`` if needed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


CRCNS_AC1_LOCAL = os.environ.get(
    "CRCNS_AC1_DATA",
    "/home/ulysse/Documents/NRFdatasets/Audio/CRCNS_AC1",
)
HAS_DATA = (
    os.path.isfile(os.path.join(CRCNS_AC1_LOCAL, "crcns-ac1.zip"))
    or os.path.isdir(os.path.join(CRCNS_AC1_LOCAL, "crcns-ac1"))
)


# ============================================================
# logspectrogram (network-free)
# ============================================================

def test_logspectrogram_synthetic_tone_peaks_at_target_freq():
    from deepSTRF.datasets.audio._logspectrogram import logspectrogram, n_bands_for

    sf = 97656.0
    t = np.arange(int(sf)) / sf
    y = np.sin(2 * np.pi * 440.0 * t)
    S, f = logspectrogram(y, sf, dt_ms=5.0)
    # Most frames should peak in the band containing 440 Hz (~band 13 at 6/oct)
    peak_band = np.bincount(np.argmax(S, axis=0)).argmax()
    assert abs(f[peak_band] - 440.0) < 80.0, (
        f"peak at {f[peak_band]:.1f} Hz, expected ~440 Hz"
    )


def test_logspectrogram_output_T_matches_dt_ms():
    """T_out should equal ceil(L_wave * dt_ms / 1000) for any dt_ms (no two-step
    downsample)."""
    from deepSTRF.datasets.audio._logspectrogram import logspectrogram

    sf = 10000.0
    y = np.random.randn(int(sf))  # 1 second of noise
    for dt_ms in (1.0, 2.0, 5.0, 10.0):
        S, _ = logspectrogram(y, sf, dt_ms=dt_ms)
        # 1000 ms / dt_ms ± 1 frame for boundary handling
        expected_T = int(np.ceil(1000.0 / dt_ms))
        assert abs(S.shape[1] - expected_T) <= 1, (
            f"dt_ms={dt_ms}: got T={S.shape[1]}, expected ~{expected_T}"
        )


def test_n_bands_for_default_matches_logspectrogram_F():
    """The convenience size helper must agree with what the kernel produces."""
    from deepSTRF.datasets.audio._logspectrogram import logspectrogram, n_bands_for

    # Asari paper layout: 100 Hz – 45 kHz at 6/oct
    expected = n_bands_for(100.0, 45000.0, 6)
    y = np.random.randn(1000)
    S, _ = logspectrogram(y, sf=10000.0, fmin=100.0, fmax=45000.0, bins_per_octave=6)
    assert S.shape[0] == expected

    # Wehr 2024 layout: 100 Hz – 25.6 kHz at 6/oct → F=49
    S2, _ = logspectrogram(y, sf=10000.0, fmin=100.0, fmax=25600.0, bins_per_octave=6)
    assert S2.shape[0] == 49


# ============================================================
# MedGauss detrend + repeat gating (network-free)
# ============================================================

def test_medgauss_detrend_removes_linear_drift():
    from deepSTRF.datasets.audio._crcns_ac1_native import medgauss_detrend

    sf = 4000.0
    t = np.arange(int(2 * sf)) / sf
    # Slow linear drift (10 mV across 2 s) + small oscillation
    drift = 10.0 * t
    oscillation = 1.0 * np.sin(2 * np.pi * 50 * t)
    detrended = medgauss_detrend(drift + oscillation, sf)

    # The 50 Hz content should survive; the drift should not.
    assert abs(np.mean(detrended)) < 1.0
    # After detrend, the trace should not span the full ±10 mV linear range.
    assert detrended.max() - detrended.min() < 5.0


def test_prepare_repeats_drops_saturated_trial():
    from deepSTRF.datasets.audio._crcns_ac1_native import (
        prepare_repeats, RepeatGating,
    )

    sf = 4000.0
    n = int(2 * sf)
    clean = 1.0 * np.sin(2 * np.pi * 5 * np.arange(n) / sf)  # tiny 5 Hz
    saturated = clean.copy()
    saturated[n // 2 : n // 2 + 200] += 250.0  # 250 mV step → way past abs_mv_max

    kept, reasons = prepare_repeats(
        [clean, clean.copy(), saturated],
        sf,
        gating=RepeatGating(abs_mv_max=150.0, min_xcorr=0.0),
    )
    # The artifact may trip either the range or the step test depending on
    # how much the MedGauss baseline tracks the step. Both are valid;
    # what we care about is that the trial is dropped.
    assert reasons[:2] == ["kept", "kept"], reasons
    assert reasons[2] in ("range", "step"), reasons
    assert len(kept) == 2


def test_prepare_repeats_drops_outlier_via_xcorr():
    from deepSTRF.datasets.audio._crcns_ac1_native import (
        prepare_repeats, RepeatGating,
    )

    sf = 4000.0
    n = int(2 * sf)
    base = 1.0 * np.sin(2 * np.pi * 5 * np.arange(n) / sf)
    # 3 clean repeats that agree, 1 anti-correlated outlier
    kept, reasons = prepare_repeats(
        [base, base + 0.01 * np.random.randn(n), base + 0.01 * np.random.randn(n), -base],
        sf,
        gating=RepeatGating(min_xcorr=0.3),
    )
    assert reasons[-1] == "xcorr", reasons
    assert len(kept) == 3


# ============================================================
# Asari sequence parsing (network-free)
# ============================================================

def test_asari_sequence_regex_handles_variable_whitespace():
    from deepSTRF.datasets.audio._crcns_ac1_native import _asari_seq_segments

    assert _asari_seq_segments("Sequence 1: 2  1  3  1  4") == [2, 1, 3, 1, 4]
    assert _asari_seq_segments("Sequence 12: 9  8") == [9, 8]
    assert _asari_seq_segments("not a sequence") is None
    assert _asari_seq_segments("Tuning curve") is None


# ============================================================
# Structural / end-to-end (require local data)
# ============================================================

@pytest.mark.skipif(not HAS_DATA, reason="CRCNS-AC1 local archive missing")
def test_wehr_loads_and_validates():
    from deepSTRF.datasets.audio import CRCNSAC1Dataset

    ds = CRCNSAC1Dataset(path=CRCNS_AC1_LOCAL, experimenter="wehr", dt_ms=5.0)
    assert ds.N_neurons == 25
    assert ds.F == 53
    assert len(ds.stims) > 0
    # stim shape contract
    assert ds.stims[0].dim() == 3 and ds.stims[0].shape[0] == 1
    assert ds.stims[0].shape[1] == ds.F
    # response paradigm: list-of-lists with NaN sentinels
    assert len(ds.responses) == len(ds.stims)
    assert len(ds.responses[0]) == ds.N_neurons
    # nrn_masks shape + sparsity (most cells don't hear most stims)
    assert tuple(ds.nrn_masks.shape) == (len(ds.stims), ds.N_neurons)
    assert ds.nrn_masks.float().mean().item() < 1.0
    # Wehr meta convention
    for meta in ds.nrn_meta:
        assert meta["experimenter"] == "wehr"
        assert meta["site"] == "A1"
        assert meta["animal_id"] == "mw"
        assert "_wehr_cell_idx" in meta


@pytest.mark.skipif(not HAS_DATA, reason="CRCNS-AC1 local archive missing")
def test_asari_a1_loads_and_validates():
    from deepSTRF.datasets.audio import CRCNSAC1Dataset

    ds = CRCNSAC1Dataset(
        path=CRCNS_AC1_LOCAL, experimenter="asari", sites="A1", dt_ms=5.0,
    )
    assert ds.N_neurons > 0
    assert ds.F == 53
    for meta in ds.nrn_meta:
        assert meta["experimenter"] == "asari"
        assert meta["site"] == "A1"
    for sm in ds.stim_meta:
        # Asari stims carry the spliced segment paths for provenance
        assert "segment_files" in sm
        assert len(sm["segments"]) == len(sm["segment_files"])


@pytest.mark.skipif(not HAS_DATA, reason="CRCNS-AC1 local archive missing")
def test_filter_api_round_trips():
    """``select_pop_by_nrn_attr`` on experimenter must reproduce the matching
    subset. The filter API stores selected indices in ``ds.I`` rather than
    mutating ``nrn_meta`` — verify both the count and the selection."""
    from deepSTRF.datasets.audio import CRCNSAC1Dataset

    ds = CRCNSAC1Dataset(path=CRCNS_AC1_LOCAL, dt_ms=5.0)
    n_total = ds.N_neurons
    selected = ds.select_pop_by_nrn_attr("experimenter", "wehr")
    assert len(selected) > 0
    assert all(ds.nrn_meta[i]["experimenter"] == "wehr" for i in selected)
    assert len(selected) < n_total  # asari + wehr should outnumber wehr alone
