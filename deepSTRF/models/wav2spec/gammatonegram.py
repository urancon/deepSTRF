"""Faithful causal gammatone-gram (Heeris / Slaney / Ellis) wav2spec front-end.

This reproduces the *exact* gammatone-gram of the ``gammatone`` PyPI package
(``gammatone.gtgram.gtgram``) — the one used by NEMS and by the Meliza 2025
dataset's in-loader spectrogram — but as a differentiable, strictly-causal
``torch`` module that lives in a model's ``wav2spec`` slot.

It is distinct from :class:`~deepSTRF.models.wav2spec.gammatone.CausalGammatone`,
which is a deepSTRF *reimplementation* with FIR gammatone kernels + an envelope
pooling window ``>= hop``. That reimplementation over-smooths datasets whose
native gammatone-gram uses a **sub-hop analysis window** (e.g. Meliza's 2.5 ms
window at a 5 ms hop), matching them at only ~0.27 band-corr. This module instead
uses the canonical Slaney ERB IIR filterbank (the package's own
``make_erb_filters`` coefficients) and the package's windowed-RMS integration, so
``Gammatonegram(wav)`` reproduces the native spectrogram to numerical precision.

Why it is causal: the Slaney ERB filterbank is a cascade of causal 2nd-order IIR
sections, and the windowed-RMS frame ``t`` integrates filterbank power over
``[t*hop, t*hop + nwin)`` with ``nwin <= hop`` — strictly inside ``[0, (t+1)*hop)``.
Sub-hop windows (``window_ms <= dt_ms``) are therefore required and enforced.

Needs the ``gammatone`` package (``pip install 'deepSTRF[meliza]'``) at
construction time only, to compute the ERB filter coefficients; the forward
pass is pure ``torch`` / ``torchaudio``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class Gammatonegram(nn.Module):
    """Canonical causal gammatone-gram (Slaney ERB filters + windowed RMS).

    Parameters
    ----------
    audio_fs : int
        Input sample rate (Hz).
    n_filters : int, default 50
        Number of ERB-spaced gammatone channels (= ``out_channels``).
    hop_ms : float, default 5.0
        Neural bin width in ms (= dataset ``dt_ms``); ``hop = round(audio_fs*hop_ms/1000)``.
    window_ms : float, default 2.5
        Analysis-window length in ms for the windowed-RMS integration. Must be
        ``<= hop_ms`` so each frame stays causal. The default 2.5 ms matches the
        Meliza 2025 paper spectrogram.
    f_min, f_max : float
        ERB filterbank frequency limits (Hz). ``f_max`` defaults to Nyquist.
    compression : {'log1p', 'log', 'none'}, default 'log1p'
        Post-integration compression. ``'log1p'`` = ``log(1 + clip(x, 0))`` is
        the Meliza / NEMS convention.
    log_floor : float, default 1e-8
        Pre-log clamp for ``compression='log'``.
    """

    def __init__(self, audio_fs: int, n_filters: int = 50, hop_ms: float = 5.0,
                 window_ms: float = 2.5, f_min: float = 1000.0,
                 f_max: Optional[float] = 8000.0, compression: str = "log1p",
                 log_floor: float = 1e-8):
        super().__init__()
        try:
            from gammatone.filters import make_erb_filters, centre_freqs
        except ImportError as e:  # pragma: no cover - import-guard
            raise ImportError(
                "Gammatonegram needs the `gammatone` package for the Slaney ERB "
                "filter coefficients. Install with `pip install 'deepSTRF[meliza]'` "
                "or `pip install gammatone`."
            ) from e
        if compression not in ("log1p", "log", "none"):
            raise ValueError(f"compression must be 'log1p'/'log'/'none', got {compression!r}")

        self.audio_fs = int(audio_fs)
        self.n_filters = int(n_filters)
        self.out_channels = self.n_filters
        self.hop = max(1, int(round(audio_fs * hop_ms / 1000.0)))
        self.nwin = max(1, int(round(audio_fs * window_ms / 1000.0)))
        if self.nwin > self.hop:
            raise ValueError(
                f"window_ms ({window_ms}) must be <= hop_ms ({hop_ms}) for strict "
                f"causality: a frame may not integrate audio past its bin boundary "
                f"(got nwin={self.nwin} > hop={self.hop})."
            )
        self.compression = compression
        self.log_floor = float(log_floor)

        if f_max is None:
            f_max = audio_fs / 2.0
        # canonical gtgram coefficients; gtgram_xe flips channel order (low->high)
        cfs = centre_freqs(audio_fs, n_filters, f_min, f_max)
        fcoefs = np.flipud(make_erb_filters(audio_fs, cfs)).astype(np.float64)
        # numerator triplets share the denominator Bs; final divide by gain.
        a = torch.as_tensor(fcoefs[:, 6:9], dtype=torch.float32)          # Bs (denominator)
        self.register_buffer("a_coeffs", a)
        self.register_buffer("b1", torch.as_tensor(fcoefs[:, (0, 1, 5)], dtype=torch.float32))
        self.register_buffer("b2", torch.as_tensor(fcoefs[:, (0, 2, 5)], dtype=torch.float32))
        self.register_buffer("b3", torch.as_tensor(fcoefs[:, (0, 3, 5)], dtype=torch.float32))
        self.register_buffer("b4", torch.as_tensor(fcoefs[:, (0, 4, 5)], dtype=torch.float32))
        self.register_buffer("gain", torch.as_tensor(fcoefs[:, 9], dtype=torch.float32))

    def extra_repr(self) -> str:
        return (f"audio_fs={self.audio_fs}, n_filters={self.n_filters}, hop={self.hop}, "
                f"nwin={self.nwin}, compression={self.compression!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"expected (B, 1, T_audio); got {tuple(x.shape)}")
        if x.shape[-1] % self.hop != 0:
            raise ValueError(
                f"input length {x.shape[-1]} is not a multiple of hop={self.hop}; "
                f"check audio_fs / dt_ms vs the dataset (grid lock C1)."
            )
        B, _, T = x.shape
        C = self.n_filters
        xe = x.expand(B, C, T)                                   # (B, C, T)
        # Slaney ERB filterbank: 4 cascaded causal 2nd-order IIR sections.
        y = torchaudio.functional.lfilter(xe, self.a_coeffs, self.b1, batching=True, clamp=False)
        y = torchaudio.functional.lfilter(y, self.a_coeffs, self.b2, batching=True, clamp=False)
        y = torchaudio.functional.lfilter(y, self.a_coeffs, self.b3, batching=True, clamp=False)
        y = torchaudio.functional.lfilter(y, self.a_coeffs, self.b4, batching=True, clamp=False)
        y = y / self.gain.view(1, C, 1)
        power = y * y                                            # (B, C, T)
        # windowed RMS: frame t integrates [t*hop, t*hop+nwin) -> sqrt(mean(power)).
        mean_p = F.avg_pool1d(power, kernel_size=self.nwin, stride=self.hop)  # (B, C, T_n)
        spec = torch.sqrt(mean_p.clamp_min(0.0))
        if self.compression == "log1p":
            spec = torch.log1p(spec.clamp_min(0.0))
        elif self.compression == "log":
            spec = torch.log(spec.clamp_min(self.log_floor))
        return spec.unsqueeze(1)                                 # (B, 1, F, T_n)
