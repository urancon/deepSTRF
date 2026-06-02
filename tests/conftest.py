"""Shared pytest fixtures and collection hooks for deepSTRF tests."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Fast vs. slow split.
#
# These modules construct *real* datasets (parse .mat/.h5, resample wavs,
# compute spectrograms) and dominate local runtime — the full suite is ~50 min
# on a machine that has the data, almost all of it here. They are gated on
# local data, so they already SKIP in CI; the cost is purely a local-dev one.
#
# Every test collected from one of these modules is auto-marked ``slow`` so the
# inner dev loop can skip them:
#
#     pytest -m "not slow"     # fast inner loop: 440/593 tests in ~17 s
#     pytest                   # full: everything (~50 min locally with data;
#                              #       CI runs this but the integration tests
#                              #       skip without data)
#
# Keep model/metric/logic tests (synthetic data) OUT of this list — they are
# the core fast suite.
# ---------------------------------------------------------------------------
_SLOW_TEST_MODULES = {
    "test_crcns_ac1",
    "test_crcns_aa_waveform",
    "test_ns1_waveform",
    "test_nat4_waveform",
    "test_le_2025_waveform",
    "test_downer2025_dataset",
    "test_wingert2026",
    "test_espejo_dataset",
    "test_alice_eeg",
    "test_audio_spec_pipeline",
    "test_alice_eeg_spec_pipeline",
}


def pytest_collection_modifyitems(config, items):
    import pytest

    slow = pytest.mark.slow
    for item in items:
        module = item.module.__name__.rsplit(".", 1)[-1]
        if module in _SLOW_TEST_MODULES:
            item.add_marker(slow)
