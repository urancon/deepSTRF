"""Tests for ``deepSTRF.datasets.audio.alice_eeg``.

End-to-end tests depend on the Brodbeck 2023 restructure being unpacked
under ``deepSTRF/datasets/audio/Alice_EEG/data/brodbeck_eelbrain_elife/``.
They skip automatically when that directory is missing (typical in CI).
The ERB filterbank helper is exercised in isolation since it's pure.
"""

from __future__ import annotations

import os

import pytest
import torch


ALICE_DATA = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..",
    "deepSTRF", "datasets", "audio", "Alice_EEG", "data",
    "brodbeck_eelbrain_elife",
))


def _has_local_alice() -> bool:
    if not os.path.isdir(ALICE_DATA):
        return False
    if not os.path.exists(os.path.join(ALICE_DATA, "stimuli", "1.wav")):
        return False
    try:
        from deepSTRF.datasets.audio.alice_eeg import _discover_subjects
    except ImportError:
        return False
    return bool(_discover_subjects(ALICE_DATA))


HAS_LOCAL = _has_local_alice()

# Optional dep — gate the whole module if mne isn't installed (it's an
# `[eeg]` extra, not a runtime dep). The pure ERB-filterbank test would
# still work without mne, but skipping the whole module is simpler.
mne = pytest.importorskip("mne")


# ============================================================
# Pure helpers (no I/O, no data dependency)
# ============================================================

def test_erb_filterbank_shape_and_positive():
    from deepSTRF.datasets.audio.alice_eeg import _erb_filterbank
    fb = _erb_filterbank(n_bands=8, sr=44100, n_fft=1024)
    assert fb.shape == (8, 513)
    # Gaussians peak at their center freq; each band must have positive mass
    assert (fb > 0).any(dim=-1).all().item()


def test_erb_filterbank_monotone_centers():
    """Successive ERB bands have monotonically increasing center frequencies."""
    import torch
    from deepSTRF.datasets.audio.alice_eeg import _erb_filterbank
    fb = _erb_filterbank(n_bands=8, sr=44100, n_fft=1024)
    peaks = torch.argmax(fb, dim=-1)  # (n_bands,)
    assert (peaks[1:] >= peaks[:-1]).all().item()


# ============================================================
# End-to-end (require local data)
# ============================================================

@pytest.fixture(scope="module")
def alice_s01():
    if not HAS_LOCAL:
        pytest.skip("Alice EEG local data missing — skip integration test")
    from deepSTRF.datasets.audio.alice_eeg import AliceEEGDataset
    return AliceEEGDataset(path=ALICE_DATA, subjects=["S01"])


def test_alice_shape_invariants(alice_s01):
    ds = alice_s01
    assert ds.F == 8
    assert ds.species == "human"
    assert ds.behavioral_state == "passive-listening"
    assert ds.N_neurons == 61
    assert len(ds.stims) == 12
    assert len(ds.stim_meta) == 12
    assert len(ds.responses) == 12

    # stims: list of (1, F, T_s) tensors
    for stim in ds.stims:
        assert stim.dim() == 3
        assert stim.shape[:2] == (1, 8)

    # responses: list[S] of list[N=61] tensors with R=1 in neurons mode
    for s in range(12):
        assert len(ds.responses[s]) == 61
        for r in ds.responses[s]:
            assert r.dim() == 2
            assert r.shape[0] == 1  # R = 1
            # Either real (1, T_s) or (1, 1) sentinel for structural NaN
            assert r.shape[1] in (1, ds.stims[s].shape[-1])


def test_alice_nrn_masks_derived(alice_s01):
    """nrn_masks is a property derived from response NaNs (data paradigm §3.1)."""
    ds = alice_s01
    assert ds.nrn_masks.shape == (12, 61)
    # S01 has 7 documented bad channels — they're masked across all 12 stims.
    bad_count = (~ds.nrn_masks.any(dim=0)).sum().item()
    assert bad_count == 7


def test_alice_total_duration(alice_s01):
    """The 12 audio segments should total ~12.4 min (Brodbeck 2023 §Tutorial)."""
    total_s = sum(m["duration_s"] for m in alice_s01.stim_meta)
    # Brodbeck cites 12.4 min = 744 s
    assert 700.0 < total_s < 800.0


def test_alice_nrn_meta(alice_s01):
    """Per-(subject, channel) entries with xyz from the standard montage."""
    md = alice_s01.nrn_meta
    assert len(md) == 61
    sample = md[0]
    assert sample["subject"] == "S01"
    assert sample["area"] == "EEG"
    assert "channel_id" in sample
    # xyz can be None for channels not in the montage, but at least most
    # should be populated
    n_with_xyz = sum(1 for m in md if m["xyz"] is not None)
    assert n_with_xyz >= 50


def test_alice_dataloader_integration(alice_s01):
    from torch.utils.data import DataLoader
    from deepSTRF.utils.data import neural_collate

    loader = DataLoader(alice_s01, batch_size=4, collate_fn=neural_collate)
    batch = next(iter(loader))
    stims, responses, valid_mask, stim_metas = (
        batch['stims'], batch['responses'], batch['valid_mask'], batch['stim_meta'])

    assert stims.dim() == 4 and stims.shape[1:3] == (1, 8)        # (B, 1, F, T)
    assert responses.dim() == 4 and responses.shape[1] == 61      # (B, N, R, T)
    assert valid_mask.shape == responses.shape
    assert not stims.isnan().any().item()                          # stims never NaN
    # mask coverage should be substantial (most positions are valid)
    assert valid_mask.float().mean().item() > 0.5


def test_alice_repeats_mode():
    """Multi-subject 'repeats' mode: N = montage channels, R = n_subjects."""
    if not HAS_LOCAL:
        pytest.skip("Alice EEG local data missing — skip integration test")
    from deepSTRF.datasets.audio.alice_eeg import AliceEEGDataset, _discover_subjects

    available = sorted(_discover_subjects(ALICE_DATA))
    if len(available) < 2:
        pytest.skip("Need ≥2 subjects unpacked for repeats-mode test")
    subjects = available[:2]

    ds = AliceEEGDataset(path=ALICE_DATA, subjects=subjects,
                           treat_subjects_as="repeats")
    assert ds.N_neurons == 61
    # at least one channel should be valid for both subjects -> R=2 slab
    for resp in ds.responses[0]:
        import torch
        if not torch.isnan(resp).all():
            assert resp.shape[0] == 2
            break


def test_alice_subjects_filter_error():
    """Requesting a nonexistent subject raises FileNotFoundError."""
    if not HAS_LOCAL:
        pytest.skip("Alice EEG local data missing — skip integration test")
    from deepSTRF.datasets.audio.alice_eeg import AliceEEGDataset
    with pytest.raises(FileNotFoundError):
        AliceEEGDataset(path=ALICE_DATA, subjects=["S999"])


def test_alice_invalid_treat_mode():
    if not HAS_LOCAL:
        pytest.skip("Alice EEG local data missing — skip integration test")
    from deepSTRF.datasets.audio.alice_eeg import AliceEEGDataset
    with pytest.raises(ValueError):
        AliceEEGDataset(path=ALICE_DATA, subjects=["S01"],
                          treat_subjects_as="bogus")


# ============================================================
# Raw-waveform branch
# ============================================================

@pytest.mark.skipif(not HAS_LOCAL, reason="Alice EEG data not unpacked under repo path")
def test_alice_waveform_branch():
    """Alice EEG's waveform branch hands out the 44.1 kHz audiobook waveforms,
    grid-locked to the spectrogram frames (hop=441 at dt=10 ms, offset 0 —
    continuous audio). The EEG responses must bin to the spec frame count, NOT
    the waveform length, and stay identical to spectrogram mode."""
    from deepSTRF.datasets.audio.alice_eeg import AliceEEGDataset

    kw = dict(path=ALICE_DATA, subjects=["S01"], dt_ms=10.0, n_frequency_bands=8)
    ds_spec = AliceEEGDataset(**kw)
    ds_wav = AliceEEGDataset(return_waveform=True, **kw)

    assert ds_wav.get_S() == ds_spec.get_S() and ds_wav.get_N() == ds_spec.get_N()
    assert ds_spec.audio_fs is None and ds_spec.hop is None
    assert ds_wav.audio_fs == 44100 and ds_wav.hop == 441
    assert ds_wav.hearing_range_hz == (20.0, 20000.0)

    ds_wav.validate()
    for s in range(ds_wav.get_S()):
        stim = ds_wav.stims[s]
        assert stim.dim() == 2 and stim.shape[0] == 1
        assert stim.shape[-1] == ds_spec.stims[s].shape[-1] * ds_wav.hop

    # responses bin to the spec frame count (the T_per_stim fix), not T_audio,
    # and are untouched by the input representation.
    for s in range(ds_wav.get_S()):
        for n in range(ds_wav.get_N()):
            assert torch.allclose(ds_spec.responses[s][n], ds_wav.responses[s][n],
                                  equal_nan=True)
