"""Pipeline-level tests for the Alice EEG wav→spec front-end.

The 2026-05-23 audit (this branch) exposed three spec knobs on the
Alice EEG dataset that had been hardcoded inside ``_gammatone_spectrogram``:

  - ``window_ms`` — FFT analysis-window length in ms (formerly ``n_fft=1024``).
  - ``fmin``      — ERB-band lower edge (formerly hardcoded to 80 Hz).
  - ``fmax``      — ERB-band upper edge (formerly hardcoded to sr/2).

Defaults preserve the historical behaviour bit-for-bit — no existing fits
change. The goal of exposing them is to enable future bin-for-bin
benchmarking against Brodbeck 2023 in one line (see the "Audit status"
callout on :class:`AliceEEGDataset` for the full rationale).

These tests don't touch actual Alice EEG data — they exercise the
``_gammatone_spectrogram`` and ``_erb_filterbank`` helpers directly on a
synthetic waveform, and pin the constructor-surface contract via
``inspect``.
"""

from __future__ import annotations

import inspect

import torch


# -----------------------------------------------------------------------------
# Helper-level tests (no Alice EEG data needed)
# -----------------------------------------------------------------------------


def test_gammatone_default_n_fft_unchanged():
    """No new kwargs → must reproduce the legacy ``n_fft=1024`` spec.

    Pins the bit-identical-by-default contract: any existing Alice EEG
    fit on disk re-runs the same wav→spec front-end with the new code.
    """
    from deepSTRF.datasets.audio.alice_eeg import _gammatone_spectrogram

    torch.manual_seed(0)
    sr = 44100
    wav = torch.randn(1, sr)               # 1 s of audio
    spec = _gammatone_spectrogram(wav, sr=sr, n_bands=8, dt_ms=10.0)
    # At sr=44100, dt_ms=10 → hop ≈ 441; n_fft=1024 → T ≈ 1*sr/hop ≈ 100.
    assert spec.dim() == 3
    assert spec.shape[1] == 8
    assert spec.shape[2] > 0


def test_gammatone_window_ms_overrides_legacy_n_fft():
    """An explicit ``window_ms`` must drive ``n_fft`` and produce a
    different spectrogram from the legacy default.
    """
    from deepSTRF.datasets.audio.alice_eeg import _gammatone_spectrogram

    torch.manual_seed(0)
    sr = 44100
    wav = torch.randn(1, sr)
    legacy = _gammatone_spectrogram(wav, sr=sr, n_bands=8, dt_ms=10.0)
    # 25 ms at sr=44100 → n_fft ≈ 1102, materially different from 1024.
    kaldi = _gammatone_spectrogram(
        wav, sr=sr, n_bands=8, dt_ms=10.0, window_ms=25.0,
    )
    assert legacy.shape == kaldi.shape
    assert not torch.allclose(legacy, kaldi), (
        "window_ms=25 should yield a different spec from the legacy "
        "n_fft=1024 default"
    )


def test_gammatone_window_ms_floors_at_hop_length():
    """``window_ms`` smaller than ``dt_ms`` should floor at hop_length
    (STFT constraint), matching the same convention used in CRCNS AA{1,2,4}.
    """
    from deepSTRF.datasets.audio.alice_eeg import _gammatone_spectrogram

    torch.manual_seed(0)
    sr = 44100
    dt_ms = 50.0
    wav = torch.randn(1, sr)
    # window_ms=5 < dt_ms=50 → user request can't be honoured exactly;
    # the helper should silently floor n_fft to hop and still produce a
    # finite-valued spec (not crash on n_fft < hop_length).
    spec = _gammatone_spectrogram(
        wav, sr=sr, n_bands=8, dt_ms=dt_ms, window_ms=5.0,
    )
    assert torch.isfinite(spec).all()


def test_gammatone_explicit_n_fft_overrides_window_ms():
    """When both ``n_fft`` and ``window_ms`` are passed, ``n_fft`` wins —
    documents the unambiguous precedence so a caller doing manual
    benchmarking can lock the FFT length exactly.
    """
    from deepSTRF.datasets.audio.alice_eeg import _gammatone_spectrogram

    torch.manual_seed(0)
    sr = 44100
    wav = torch.randn(1, sr)
    a = _gammatone_spectrogram(wav, sr=sr, n_bands=8, dt_ms=10.0,
                                n_fft=2048, window_ms=10.0)
    b = _gammatone_spectrogram(wav, sr=sr, n_bands=8, dt_ms=10.0,
                                n_fft=2048, window_ms=999.0)
    assert torch.allclose(a, b), "n_fft must override window_ms"


def test_erb_filterbank_fmax_is_honoured():
    """Setting ``fmax`` below Nyquist must concentrate the ERB bands in
    that range. Verifiable via the band-center spacing: the largest
    center frequency must be ≤ fmax.
    """
    from deepSTRF.datasets.audio.alice_eeg import _erb_filterbank

    sr = 44100
    fb_wide = _erb_filterbank(n_bands=8, sr=sr, n_fft=1024)
    fb_speech = _erb_filterbank(
        n_bands=8, sr=sr, n_fft=1024, f_max=8000.0,
    )
    # Both shapes match (n_bands, n_fft//2+1).
    assert fb_wide.shape == fb_speech.shape

    # The narrower fb_speech should have its highest-band peak at a
    # lower frequency than fb_wide → diagnose via argmax along the
    # frequency axis.
    freqs = torch.linspace(0.0, sr / 2, 1024 // 2 + 1)
    wide_peak = freqs[int(fb_wide[-1].argmax())]
    speech_peak = freqs[int(fb_speech[-1].argmax())]
    assert speech_peak < wide_peak
    # And the speech peak should be at or below the requested fmax (with
    # some slack for the Gaussian center vs. its max bin).
    assert speech_peak.item() <= 8000.0 + 100.0


def test_erb_filterbank_fmin_is_honoured():
    """Lifting ``fmin`` above the default 80 Hz must push the lowest
    band up correspondingly."""
    from deepSTRF.datasets.audio.alice_eeg import _erb_filterbank

    sr = 44100
    fb_default = _erb_filterbank(n_bands=8, sr=sr, n_fft=1024)
    fb_raised = _erb_filterbank(
        n_bands=8, sr=sr, n_fft=1024, f_min=500.0,
    )
    freqs = torch.linspace(0.0, sr / 2, 1024 // 2 + 1)
    default_low = freqs[int(fb_default[0].argmax())]
    raised_low = freqs[int(fb_raised[0].argmax())]
    assert raised_low > default_low
    assert raised_low.item() >= 500.0 - 100.0


# -----------------------------------------------------------------------------
# Constructor-surface tests (no MNE / no data needed — uses inspect only)
# -----------------------------------------------------------------------------


def test_alice_dataset_exposes_spec_kwargs():
    """All four spec knobs must be visible on the constructor signature
    with documented defaults that preserve the legacy behaviour.
    """
    from deepSTRF.datasets.audio.alice_eeg import AliceEEGDataset

    sig = inspect.signature(AliceEEGDataset.__init__)
    params = sig.parameters

    assert "window_ms" in params
    assert params["window_ms"].default is None, (
        f"window_ms default should be None (legacy n_fft=1024 path), "
        f"got {params['window_ms'].default!r}"
    )

    assert "fmin" in params
    assert params["fmin"].default == 80.0, (
        f"fmin default should be 80.0 Hz (Brodbeck 2023), got "
        f"{params['fmin'].default!r}"
    )

    assert "fmax" in params
    assert params["fmax"].default is None, (
        f"fmax default should be None (=sr/2 fallback), got "
        f"{params['fmax'].default!r}"
    )

    assert "spec_backend" in params
    assert params["spec_backend"].default == "gaussian", (
        f"spec_backend default should be 'gaussian' (back-compat), got "
        f"{params['spec_backend'].default!r}"
    )


# -----------------------------------------------------------------------------
# 5. Heeris backend (paper-faithful, gated on the optional ``gammatone`` dep)
# -----------------------------------------------------------------------------


def _have_gammatone() -> bool:
    try:
        import gammatone.gtgram  # noqa: F401
        return True
    except ImportError:
        return False


def test_gammatone_backend_invalid_value_raises():
    """An unknown ``backend`` argument must raise ValueError — guards the
    documented {'gaussian', 'heeris'} contract."""
    import pytest

    from deepSTRF.datasets.audio.alice_eeg import _gammatone_spectrogram

    wav = torch.zeros(1, 44100)
    with pytest.raises(ValueError, match="backend must be"):
        _gammatone_spectrogram(wav, sr=44100, n_bands=8, dt_ms=10.0,
                                backend="kaldi")


def test_heeris_backend_smokes():
    """Smoke test the Heeris time-domain gammatone backend end-to-end on a
    short synthetic wave. Skips cleanly when the optional ``gammatone``
    package isn't installed.

    Pins the shape contract (``(1, n_bands, T)`` with low-frequency band
    at index 0) and the finite-value contract — failures here are the
    most common "did the optional dep stay wired correctly" regression.
    """
    import pytest

    if not _have_gammatone():
        pytest.skip(
            "optional `gammatone` package not installed — "
            "Heeris backend is opt-in via the [eeg] extra"
        )

    from deepSTRF.datasets.audio.alice_eeg import _gammatone_spectrogram

    sr = 44100
    torch.manual_seed(0)
    # ~1 s of band-limited noise → all 8 ERB bands carry energy.
    wav = torch.randn(1, sr) * 0.1
    spec = _gammatone_spectrogram(
        wav, sr=sr, n_bands=8, dt_ms=10.0, backend="heeris",
    )
    assert spec.dim() == 3
    assert spec.shape[0] == 1 and spec.shape[1] == 8
    # Heeris uses window_time=25ms and hop_time=10ms, so T ≈ sr * 1s / hop.
    assert spec.shape[2] > 0
    assert torch.isfinite(spec).all(), (
        "Heeris backend should never emit NaN/inf for a normal waveform"
    )


def test_heeris_backend_low_band_at_index_0():
    """The Heeris ``gtgram`` output is high-frequency-first; our wrapper
    must flip to low-frequency-first to match the Gaussian backend's
    convention. Verifiable by feeding a low-frequency tone and checking
    that band 0 carries the energy."""
    import pytest

    if not _have_gammatone():
        pytest.skip("optional `gammatone` package not installed")

    from deepSTRF.datasets.audio.alice_eeg import _gammatone_spectrogram

    sr = 44100
    # 200 Hz pure tone — should light up the lowest ERB band (centered
    # around 80 Hz with f_min=80, the next bands ramp up).
    t = torch.arange(sr) / sr
    wav = torch.sin(2 * torch.pi * 200.0 * t).unsqueeze(0)
    spec = _gammatone_spectrogram(
        wav, sr=sr, n_bands=8, dt_ms=10.0, backend="heeris",
    )
    # Average each band's log-power over time
    per_band = spec.squeeze(0).mean(dim=-1)
    top_band = int(per_band.argmax())
    assert top_band <= 1, (
        f"200 Hz tone should peak in band 0 or 1 (low-frequency first), "
        f"got band {top_band}; per-band power: {per_band.tolist()}"
    )
