"""Espejo NAT waveform inset: the sound must start where the cochleagram's sound starts.

The Zenodo cochleagrams come in two layouts: 5.0 s (0.5 s pre-stim + 4 s sound + 0.5 s
post-stim) and 5.5 s (2.0 s pre-stim + 3 s sound + 0.5 s post-stim; the 93 sounds whose
mirror wav is 3 s at 48 kHz). The waveform must be inset at 0.5 s and 2.0 s respectively.
Synthetic wavs, no data needed.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from deepSTRF.datasets.audio.espejo import EspejoDataset


class _Stub:
    audio_fs = 44100
    hop = 441                       # 10-ms bins


def _onset_s(tmp_path, dur_s, sr, T_neural):
    x = np.ones(int(round(dur_s * sr)), dtype=np.float32) * 0.5
    f = tmp_path / f"s{dur_s}_{sr}.wav"
    sf.write(f, x, sr)
    w = EspejoDataset._nat_waveform_for_stim(_Stub(), str(f), T_neural)
    assert w.shape == (1, T_neural * _Stub.hop)
    nz = torch.nonzero(w[0].abs() > 0.25)
    return float(nz[0]) / _Stub.audio_fs


@pytest.mark.parametrize("dur_s, sr, T_neural, expected", [
    (4.0, 44100, 500, 0.5),         # 5.0-s cochleagram, 4-s sound
    (3.7, 44100, 500, 0.5),         # slightly short 4-s file: same layout
    (3.0, 48000, 550, 2.0),         # 5.5-s cochleagram, 3-s sound
])
def test_nat_waveform_inset_matches_cochleagram_layout(tmp_path, dur_s, sr, T_neural, expected):
    assert _onset_s(tmp_path, dur_s, sr, T_neural) == pytest.approx(expected, abs=2e-3)
