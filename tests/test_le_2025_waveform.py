"""Tests for the raw-waveform branch of the Le 2025 dataset.

Integration test — needs the unpacked figshare archive AND skips when the box
has too little RAM: nat8b loads ~445 cells of response data and peaks near 14 GB,
so we gate on MemAvailable to avoid OOM-killing the runner. The waveform-branch
*core logic* (grid-lock + offset-0 + gammatone build) is validated memory-light in
``untracked/meliza_wav_light.py``; this test exercises it through the real loader
when the hardware allows.
"""
from __future__ import annotations

import os

import pytest
import torch


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


_LE2025_LOCAL = os.environ.get(
    "LE2025_DATA",
    "/home/ulysse/Documents/NRFdatasets/Audio/ZF_Restore_Bao/zebf-auditory-restoration-1",
)
_HAS_DATA = os.path.isdir(os.path.join(_LE2025_LOCAL, "nat8b-stimuli"))


@pytest.mark.skipif(not _HAS_DATA,
                    reason=f"Le2025 data not at {_LE2025_LOCAL!r}; set $LE2025_DATA.")
@pytest.mark.skipif(_mem_available_gb() < 18.0,
                    reason="nat8b loads ~14 GB of responses; need >=18 GB free to avoid OOM.")
def test_le_waveform_branch():
    """Le 2025's waveform branch hands out the 48 kHz source wavs grid-locked to
    the gammatone-gram frames (hop=240 at dt=5 ms, offset 0 — the wav is the full
    stimulus). Skips unless the data is present and RAM is sufficient."""
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

    # the wav aligns from t=0 (no silence flank): sound energy starts immediately
    w0 = ds.stims[0][0]
    assert float((w0[:ds.hop] ** 2).sum()) > 0.0


@pytest.mark.skipif(not os.path.isdir(os.path.join(_LE2025_LOCAL, "synth8b-stimuli")),
                    reason=f"Le2025 synth8b data not at {_LE2025_LOCAL!r}; set $LE2025_DATA.")
def test_le_waveform_responses_binned_at_dt():
    """Regression: in waveform mode the responses were binned to the number of AUDIO
    samples (the per-stim length was read after the spectrograms were swapped for
    waveforms), so validate() rejected every waveform dataset. synth8b is small enough
    to run without the nat8b RAM gate."""
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
