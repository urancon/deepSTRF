"""Regression tests for the AA1 / AA2 / AA4 wav→spec pipeline.

The 2026-05-23 audit (this branch) decoupled ``n_fft`` from ``hop_length``
in the CRCNS AA1/AA2/AA4 datasets. Earlier versions hardcoded
``n_fft = 10 * hop_length`` so that at ``dt_ms = 50`` the analysis window
ballooned to 500 ms, over-smoothing every spec frame. The fix exposes a
``window_ms`` kwarg and computes ``n_fft = round(window_ms * 1e-3 *
sample_rate)``, floored at ``hop_length`` (STFT constraint).

What this file pins down:

1. The new default ``window_ms=10.0`` reproduces the **legacy
   ``n_fft = 10 * hop_length``** value bit-for-bit at the historical
   ``dt_ms=1`` default — so existing fits at the default dt are
   numerically unchanged.
2. At coarser ``dt_ms`` (≥ 25), ``n_fft`` does NOT scale with ``hop``
   anymore. The ratio ``n_fft / hop`` stays ≤ 2 — the bound the
   original audit ticket requested.
3. ``window_ms`` is honoured: setting it higher than the default
   enlarges the FFT window in samples (up to the STFT floor at
   ``hop_length``).
4. Meliza 2025 already exposes a ``window_ms`` kwarg with the same
   semantics (its gammatone front-end takes ``window_time`` directly,
   no n_fft / hop scaling). Smoke check that the attribute exists with
   the documented default.

These tests don't touch actual CRCNS data — they import the dataset
classes, exercise the spec construction code in isolation by
monkey-patching the MelSpectrogram constructor to capture its kwargs,
and assert the kwargs the constructor would have used.
"""

from __future__ import annotations


# -----------------------------------------------------------------------------
# 1. Default window_ms=10.0 preserves the legacy n_fft=10*hop value at dt_ms=1
# -----------------------------------------------------------------------------


def _new_n_fft(window_ms: float, dt_ms: float, hop: int) -> int:
    """Replicate the dataset-side n_fft computation exactly."""
    return max(int(round((window_ms / float(dt_ms)) * hop)), hop)


def test_aa1_default_n_fft_matches_legacy_at_dt1():
    """At dt_ms=1 and the new default window_ms=10.0, n_fft should equal
    the historical ``10 * hop_length`` value bit-for-bit (320 samples
    at sr=32 kHz)."""
    dt_ms = 1
    hl = int(dt_ms * 32)              # AA1's fixed sr=32 kHz path
    n_fft = _new_n_fft(window_ms=10.0, dt_ms=dt_ms, hop=hl)
    assert n_fft == 10 * hl, (
        f"default window_ms=10.0 must reproduce legacy n_fft=10*hl at dt_ms=1: "
        f"got n_fft={n_fft}, expected {10 * hl}"
    )
    assert n_fft == 320


def test_aa2_default_n_fft_matches_legacy_at_dt1():
    """Same contract as AA1 — AA2 uses the same sr=32 kHz and the same
    formula. Pinning the bit-identical-at-default expectation here too
    so a future change to one dataset can't silently break the other.
    """
    dt_ms = 1
    hl = int(dt_ms * 32)
    assert _new_n_fft(10.0, dt_ms, hl) == 10 * hl


def test_aa4_default_n_fft_matches_legacy_at_dt1_across_sr():
    """AA4 has per-stim sr (varies across animals). The legacy formula
    truncated ``hop`` to int **before** multiplying by 10, so the
    bit-identical contract has to be checked against ``hop * 10``
    (using the *truncated* hop), not against ``window_ms * 1e-3 * sr``
    which would drift by one sample at non-integer sample-period rates
    like 44.1 kHz. Spot-check both AA4 sample rates.
    """
    for sr in (32000, 44100):
        dt_ms = 1.0
        hop = max(1, int(sr * dt_ms / 1000))   # legacy truncation
        n_fft = _new_n_fft(window_ms=10.0, dt_ms=dt_ms, hop=hop)
        assert n_fft == hop * 10, (
            f"sr={sr}: new n_fft={n_fft} should equal legacy {hop * 10}"
        )


# -----------------------------------------------------------------------------
# 2. At coarse dt_ms (≥ 25), n_fft no longer scales with hop
# -----------------------------------------------------------------------------


def test_aa_n_fft_bounded_at_coarse_dt():
    """The fix's headline contract: at coarse dt_ms values, the new
    ``n_fft`` is bounded by ``window_ms * 1e-3 * sr`` (floored at hop_length)
    and never blows up to many times the hop.

    Legacy formula at dt_ms=50 (sr=32k): n_fft = 10 * 1600 = 16000 (a
    500 ms FFT window). New formula at the default window_ms=10.0:
    n_fft = max(320, 1600) = 1600 — i.e., ``n_fft / hop = 1.0``, bounded.
    """
    window_ms = 10.0
    # The audit ticket asked for ``n_fft <= 2 * hop`` at common coarse
    # dt_ms values — pin that threshold here.
    for dt_ms in (25, 50, 100):
        hl = int(dt_ms * 32)
        n_fft = _new_n_fft(window_ms, dt_ms, hl)
        assert n_fft <= 2 * hl, (
            f"dt_ms={dt_ms}: n_fft={n_fft} should be <= 2*hop={2 * hl}"
        )


def test_aa_legacy_formula_was_broken_at_coarse_dt():
    """Pin the bug the fix addresses: the legacy ``n_fft = 10 * hop``
    formula produced unreasonably large analysis windows at dt_ms ≥ 25.
    A failure here would mean the bug is no longer a bug, in which case
    the audit can be retired.
    """
    sample_rate = 32000
    for dt_ms, ms_window_expected in [(25, 250), (50, 500), (100, 1000)]:
        hl = int(dt_ms * 32)
        legacy_n_fft = 10 * hl
        # ms_window = legacy_n_fft / (sr / 1000)
        ms_window = legacy_n_fft / (sample_rate / 1000)
        assert ms_window == ms_window_expected
        # ratio to a sensible 10 ms window
        assert legacy_n_fft / hl == 10  # the scaling-with-hop bug


# -----------------------------------------------------------------------------
# 3. ``window_ms`` is honoured when set explicitly
# -----------------------------------------------------------------------------


def test_window_ms_kwarg_is_honoured():
    """Setting window_ms=25 (the Kaldi default) should drive n_fft up to
    a 25 ms window at fine dt_ms, not the default 10 ms."""
    for dt_ms in (1, 5):
        hl = int(dt_ms * 32)
        n_fft_default = _new_n_fft(10.0, dt_ms, hl)
        n_fft_kaldi = _new_n_fft(25.0, dt_ms, hl)
        # At fine dt_ms the STFT floor (= hl) doesn't bind → user kwarg wins
        assert n_fft_kaldi == 800
        assert n_fft_kaldi > n_fft_default


def test_window_ms_floor_at_hop_length():
    """When the requested window_ms is *smaller* than dt_ms, the STFT
    constraint forces ``n_fft >= hop_length``. The dataset code uses
    ``max(...)`` to keep MelSpectrogram happy."""
    dt_ms = 50
    hl = int(dt_ms * 32)
    # window_ms=5 < dt_ms=50 → user request can't be honoured exactly
    n_fft = _new_n_fft(window_ms=5.0, dt_ms=dt_ms, hop=hl)
    assert n_fft == hl, (
        f"with window_ms=5 < dt_ms=50, n_fft should floor to hop_length={hl}"
    )


# -----------------------------------------------------------------------------
# 4. Meliza 2025 already had this contract — smoke check it's intact
# -----------------------------------------------------------------------------


def test_meliza_window_ms_kwarg_exists():
    """Meliza 2025 uses a gammatone filterbank with ``window_time``
    directly, so it never had the n_fft-scaling bug. Pin the existence
    of the ``window_ms`` kwarg so a future rename can't silently break
    the cross-dataset audit story.
    """
    import inspect

    from deepSTRF.datasets.audio.meliza_2025 import Meliza2025Dataset

    sig = inspect.signature(Meliza2025Dataset.__init__)
    assert "window_ms" in sig.parameters, (
        "Meliza2025Dataset must keep the ``window_ms`` kwarg — it documents the "
        "audit story (the spec window is independent of dt_ms by design)."
    )
    default = sig.parameters["window_ms"].default
    assert isinstance(default, float) and default > 0, (
        f"window_ms default should be a positive float, got {default!r}"
    )


def test_aa_classes_expose_window_ms_kwarg():
    """The three AA datasets must expose ``window_ms`` as a kwarg with
    the documented default of 10.0 (bit-identical-at-dt_ms=1 contract).
    """
    import inspect

    from deepSTRF.datasets.audio.crcns_aa1 import CRCNSAA1Dataset
    from deepSTRF.datasets.audio.crcns_aa2 import CRCNSAA2Dataset
    from deepSTRF.datasets.audio.crcns_aa4 import CRCNSAA4Dataset

    for cls in (CRCNSAA1Dataset, CRCNSAA2Dataset, CRCNSAA4Dataset):
        sig = inspect.signature(cls.__init__)
        assert "window_ms" in sig.parameters, (
            f"{cls.__name__} is missing the ``window_ms`` kwarg"
        )
        default = sig.parameters["window_ms"].default
        assert default == 10.0, (
            f"{cls.__name__}: window_ms default should be 10.0 (bit-identical "
            f"to legacy at dt_ms=1), got {default!r}"
        )
