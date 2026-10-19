"""Tests for the raw-waveform branch of the Le 2025 dataset.

Integration test — needs the unpacked figshare archive. Loading nat8b in
waveform mode peaks near 1.2 GB (measured 2026-10-14). The earlier ~14 GB peak,
and the RAM gate that went with it, came from responses binned at the audio
rate instead of ``dt_ms``, a bug fixed since.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


_LE2025_LOCAL = os.environ.get(
    "LE2025_DATA",
    str(Path.home() / "Library/Caches/deepSTRF/Le_2025/zebf-auditory-restoration-1"),
)
_HAS_DATA = os.path.isdir(os.path.join(_LE2025_LOCAL, "nat8b-stimuli"))


@pytest.mark.skipif(not _HAS_DATA,
                    reason=f"Le2025 data not at {_LE2025_LOCAL!r}; set $LE2025_DATA.")
def test_le_waveform_branch():
    """Le 2025's waveform branch hands out the 48 kHz source wavs grid-locked to
    the gammatone-gram frames (hop=240 at dt=5 ms, offset 0 — the wav is the full
    stimulus). Skips unless the data is present."""
    from deepSTRF.datasets.audio.le_2025 import Le2025Dataset

    ds = Le2025Dataset(path=_LE2025_LOCAL, experiment="nat8b",
                           return_waveform=True, compute_reliability=False)
    assert ds.audio_fs == 48000 and ds.hop == 240
    assert ds.hearing_range_hz == (250.0, 8000.0)

    # grid-lock holds (also enforced by validate() at construction, vs responses)
    ds.validate()
    for s in range(ds.get_S()):
        stim = ds.stims[s]
        assert stim.dim() == 2 and stim.shape[0] == 1, \
            f"Le 2025 stim {s} must be (1, T_audio); got {tuple(stim.shape)}"
        assert stim.shape[-1] % ds.hop == 0

    # the wav aligns from t=0: the loaded waveform is the file itself from its first
    # sample (no offset), zero-padded at the end. (Testing for sound in the first
    # frame is wrong: 65/96 nat8b files are digitally silent for their first ~7 ms.)
    import soundfile as sf
    raw, sr = sf.read(os.path.join(_LE2025_LOCAL, "nat8b-stimuli",
                                   ds.stim_meta[0]["name"] + ".wav"))
    assert sr == ds.audio_fs
    w0 = ds.stims[0][0]
    n = min(len(raw), w0.shape[-1])
    assert torch.allclose(w0[:n], torch.as_tensor(raw[:n], dtype=torch.float32))
    assert float(w0[n:].abs().sum()) == 0.0


@pytest.mark.skipif(not os.path.isdir(os.path.join(_LE2025_LOCAL, "synth8b-stimuli")),
                    reason=f"Le2025 synth8b data not at {_LE2025_LOCAL!r}; set $LE2025_DATA.")
def test_le_waveform_responses_binned_at_dt():
    """Regression: in waveform mode the responses were binned to the number of AUDIO
    samples (the per-stim length was read after the spectrograms were swapped for
    waveforms), so validate() rejected every waveform dataset."""
    from deepSTRF.datasets.audio.le_2025 import Le2025Dataset

    wav = Le2025Dataset(path=_LE2025_LOCAL, experiment="synth8b",
                        return_waveform=True, compute_reliability=False)
    spec = Le2025Dataset(path=_LE2025_LOCAL, experiment="synth8b",
                         compute_reliability=False)
    for s in range(len(wav.stims)):
        assert wav.stims[s].shape[-1] == spec.stims[s].shape[-1] * wav.hop
        for a, b in zip(wav.responses[s], spec.responses[s]):
            if a.numel() > 1:
                assert torch.equal(a, b)
