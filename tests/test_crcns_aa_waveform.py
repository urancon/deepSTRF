"""Tests for the raw-waveform branch of the CRCNS-AA datasets (AA1, AA2).

Integration tests — they need the unpacked CRCNS-AA archives under the deepSTRF
cache and skip automatically when those are missing. The base-class grid-lock
validation itself is unit-tested without data in ``tests/test_ns1_waveform.py``
(via ``_ToyAudioWav``); here we check the AA datasets specifically produce a
correctly grid-locked waveform branch with responses unchanged vs spec mode.
"""
from __future__ import annotations

import os

import pytest
import torch


def _has_aa(name: str) -> bool:
    from deepSTRF.utils.data_download import default_cache_dir
    root = str(default_cache_dir(name))
    return os.path.isdir(os.path.join(root, "all_stims"))


# (label, import path, class, narrow kwargs for a fast load)
AA_CASES = [
    ("AA1", "deepSTRF.datasets.audio.crcns_aa1", "CRCNSAA1Dataset",
     dict(areas=("MLd",), stimuli=("flatrip",))),
    ("AA2", "deepSTRF.datasets.audio.crcns_aa2", "CRCNSAA2Dataset",
     dict(areas=("mld",), stimuli=("flatrip",))),
]


@pytest.mark.parametrize("label,module,cls,kw", AA_CASES,
                         ids=[c[0] for c in AA_CASES])
def test_aa_waveform_branch(label, module, cls, kw):
    if not _has_aa(label):
        pytest.skip(f"CRCNS-{label} cache missing — skip integration test")
    import importlib
    Dataset = getattr(importlib.import_module(module), cls)

    ds_spec = Dataset(**kw)
    ds_wav = Dataset(return_waveform=True, **kw)

    # same stims / neurons; spec mode advertises no audio_fs, wav mode does
    assert ds_wav.get_S() == ds_spec.get_S() and ds_wav.get_N() == ds_spec.get_N()
    assert ds_spec.audio_fs is None and ds_spec.hop is None
    assert ds_wav.audio_fs == 32000 and ds_wav.hop == 32
    assert ds_wav.hearing_range_hz == (250.0, 8000.0)

    # grid-lock holds (also exercised by validate(), called at construction)
    ds_wav.validate()
    for s in range(ds_wav.get_S()):
        stim = ds_wav.stims[s]
        assert stim.dim() == 2 and stim.shape[0] == 1, \
            f"{label} stim {s} must be (1, T_audio); got {tuple(stim.shape)}"
        assert stim.shape[-1] == ds_spec.stims[s].shape[-1] * ds_wav.hop

    # responses are untouched by the input representation
    for s in range(ds_wav.get_S()):
        for n in range(ds_wav.get_N()):
            assert torch.allclose(ds_spec.responses[s][n], ds_wav.responses[s][n],
                                  equal_nan=True)
