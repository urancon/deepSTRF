"""Tests for the raw-waveform branch of the NAT4 dataset.

Integration test — needs the unpacked NAT4 release (incl. the ``wav/`` folder
from ``wav.zip``) under the deepSTRF cache and skips automatically when missing.
The base-class grid-lock validation itself is unit-tested without data in
``tests/test_ns1_waveform.py`` (via ``_ToyAudioWav``); here we check NAT4
specifically produces a correctly grid-locked waveform branch, with the 1 s
source sound inset at the trial's pre-silence offset and responses unchanged
vs spec mode.
"""
from __future__ import annotations

import os

import pytest
import torch


def _has_nat4() -> bool:
    from deepSTRF.utils.data_download import default_cache_dir
    root = str(default_cache_dir("NAT4"))
    return (os.path.isdir(os.path.join(root, "wav"))
            and os.path.isdir(os.path.join(root, "A1_NAT4_ozgf.fs100.ch18")))


def test_nat4_waveform_branch():
    """NAT4's waveform branch reads the 44.1 kHz source wavs and embeds each
    1 s sound at the recording's pre-silence offset inside the 1.5 s trial
    window, grid-locked to ``T_neural * hop`` (hop=441, no resampling). Skips
    unless NAT4 + its wav/ folder are in the cache."""
    if not _has_nat4():
        pytest.skip("NAT4 cache (with wav/) missing — skip integration test")
    from deepSTRF.datasets.audio.nat4 import NAT4Dataset

    # est subset: 575 stims, skips the expensive per-site spike pass
    ds_spec = NAT4Dataset(area="A1", subset="est")
    ds_wav = NAT4Dataset(area="A1", subset="est", return_waveform=True)

    assert ds_wav.get_S() == ds_spec.get_S() and ds_wav.get_N() == ds_spec.get_N()
    assert ds_spec.audio_fs is None and ds_spec.hop is None
    assert ds_wav.audio_fs == 44100 and ds_wav.hop == 441
    assert ds_wav.hearing_range_hz == (200.0, 40000.0)
    assert ds_wav._pre_samples == int(round(0.25 * 44100))   # 0.25 s pre-silence

    # grid-lock holds (also exercised by validate(), called at construction)
    ds_wav.validate()
    for s in range(ds_wav.get_S()):
        stim = ds_wav.stims[s]
        assert stim.dim() == 2 and stim.shape[0] == 1, \
            f"NAT4 stim {s} must be (1, T_audio); got {tuple(stim.shape)}"
        assert stim.shape[-1] == ds_spec.stims[s].shape[-1] * ds_wav.hop

    # the inset sound carries all the energy; the pre/post silence pads are zero
    pre = ds_wav._pre_samples
    w0 = ds_wav.stims[0][0]
    assert float((w0[:pre] ** 2).sum()) == 0.0
    assert float((w0[pre:pre + 44100] ** 2).sum()) > 0.0
    assert float((w0[pre + 44100:] ** 2).sum()) == 0.0

    # responses are untouched by the input representation
    for s in range(ds_wav.get_S()):
        for n in range(ds_wav.get_N()):
            assert torch.allclose(ds_spec.responses[s][n], ds_wav.responses[s][n],
                                  equal_nan=True)
