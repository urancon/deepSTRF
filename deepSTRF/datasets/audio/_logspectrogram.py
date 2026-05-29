"""Hamming-windowed STFT at explicit log-spaced frequencies (Goertzel).

This is the Python port of ``wehr/Tools/logspectrogram.m`` and its sibling
in ``asari/Tools/`` (Christian Machens / Hiroki Asari / Tomas Hromadka,
Zador Lab, CSHL). MATLAB's ``spectrogram(y, win, noverlap, f, fs)`` with
an explicit frequency vector evaluates the DFT only at the requested
frequencies (a Goertzel-style sum), rather than computing a full FFT and
interpolating. We do the same.

Unlike the MATLAB pipeline used in Rançon 2024/2025 (which computed at
the native ``dt=1 ms`` and then time-downsampled), this primitive
produces the spectrogram **directly at the target resolution** by
setting ``hop = round(dt_ms * sf / 1000)``. That avoids the aliasing
that a naive resample-after-the-fact would induce.

Only used internally by the CRCNS AC1 loader; not part of the public
API.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def logspectrogram(
    y: np.ndarray,
    sf: float,
    *,
    dt_ms: float = 5.0,
    fmin: float = 100.0,
    fmax: float = 45000.0,
    bins_per_octave: int = 6,
    window_ms: Optional[float] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a Hamming STFT at log-spaced frequencies in dB.

    Parameters
    ----------
    y : (T_wave,) array_like
        Time-domain waveform. Coerced to float64 internally.
    sf : float
        Sampling rate of ``y`` in Hz.
    dt_ms : float, default 5.0
        Output bin width in milliseconds. The hop is set to
        ``round(dt_ms * sf / 1000)`` so that output frame ``i`` covers
        time ``[i*dt_ms, (i+1)*dt_ms)`` ms.
    fmin, fmax : float
        Frequency range in Hz.
    bins_per_octave : int, default 6
        Spectral density. With the defaults (100 Hz, 45 kHz, 6/oct) the
        output has 54 bands — the Rançon 2025 Asari setting. Pass
        ``fmax=25600.0`` to recover the 49-band Wehr setting.
    window_ms : float, optional
        STFT analysis-window length in ms. Defaults to ``2 * dt_ms``
        (the MATLAB ``overlap=2`` convention: window = overlap × dt).
    eps : float
        Floor added before ``log10`` to avoid -inf on silent bins.

    Returns
    -------
    S : (F, T_out) float32 ndarray
        Log-power spectrogram in dB (``20 * log10(|X| + eps)``).
    freqs : (F,) float64 ndarray
        Band-center frequencies in Hz.

    Notes
    -----
    The DFT kernel is built explicitly as ``exp(-2πj · f · n / sf) * w[n]``
    and multiplied against a sliding view of the windowed signal. This is
    O(F × W × T_out) — for the AC1 settings (F=54, W≈1k samples,
    T_out≈2k) that's a few-million-element matmul per stim, well under a
    second.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    sf = float(sf)
    if window_ms is None:
        window_ms = 2.0 * dt_ms

    hop = max(int(round(dt_ms * sf / 1000.0)), 1)
    win_size = max(int(round(window_ms * sf / 1000.0)), 4)

    # log-spaced bin centers (inclusive of fmin)
    dx = 1.0 / bins_per_octave
    n_bands = int(np.floor(np.log2(fmax / fmin) / dx)) + 1
    freqs = fmin * 2.0 ** (np.arange(n_bands) * dx)

    win = np.hamming(win_size)

    # Center frame i on sample i*hop in the original signal: pad by win/2
    # on both sides so frames[i] = y_pad[i*hop : i*hop + win].
    half = win_size // 2
    y_pad = np.concatenate([
        np.zeros(half, dtype=np.float64),
        y,
        np.zeros(win_size, dtype=np.float64),
    ])
    n_frames = int(np.ceil(len(y) / hop))

    # DFT kernel (F, W): Goertzel-style sum at the target frequencies only.
    n_idx = np.arange(win_size, dtype=np.float64)
    phase = -2.0j * np.pi * np.outer(freqs, n_idx) / sf
    kernel = np.exp(phase) * win[None, :]

    # Sliding view, picking every hop-th frame.
    from numpy.lib.stride_tricks import sliding_window_view
    starts = np.arange(n_frames) * hop
    frames = sliding_window_view(y_pad, win_size)[starts]  # (n_frames, W)

    spec = kernel @ frames.T                              # (F, n_frames) complex
    S_db = 20.0 * np.log10(np.abs(spec) + eps)
    return S_db.astype(np.float32), freqs


def n_bands_for(fmin: float, fmax: float, bins_per_octave: int) -> int:
    """Return the ``F`` that ``logspectrogram`` will produce for these args.

    Useful for sizing arrays / validating constructor args before any
    waveform is loaded.
    """
    return int(np.floor(np.log2(fmax / fmin) * bins_per_octave)) + 1
