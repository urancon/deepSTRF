"""Decoding metrics: quality of a reconstructed audio spectrogram.

Inputs follow the audio-stim layout ``(B, 1, F, T)``: a predicted and a target
**log-power** spectrogram. NaN in the target marks padded / missing frames.

- :func:`spectrogram_corrcoef` — Pearson r per frequency band over time.
- :func:`mel_cepstral_distortion` — mean distance between mel-cepstra (MCD).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np
import torch

from deepSTRF.metrics.performance import corrcoef

# dB scaling used by much of the speech-synthesis / voice-conversion literature;
# Kubichek (1993) used 1. See mel_cepstral_distortion.
MCD_DB_SCALE = 10.0 * math.sqrt(2.0) / math.log(10.0)


def _bands_as_channels(x: torch.Tensor) -> torch.Tensor:
    """``(B, 1, F, T)`` → ``(B, F, 1, T)``: frequency bands on the 'neuron' axis."""
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"expected a (B, 1, F, T) spectrogram, got {tuple(x.shape)}")
    return x.transpose(1, 2)


def spectrogram_corrcoef(pred: torch.Tensor, target: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         reduction: str = "mean") -> torch.Tensor:
    """Pearson correlation between predicted and target spectrogram, per band.

    For each frequency band, correlates the predicted and target values over
    all valid ``(B, T)`` positions (NaN target positions are dropped), then
    reduces over bands. Apply it to **log** spectrograms — the standard
    reconstruction-accuracy measure in neural speech decoding (e.g. Pasley et
    al. 2012, PLoS Biol; Angrick et al. 2019, J Neural Eng).

    Parameters
    ----------
    pred, target : torch.Tensor
        ``(B, 1, F, T)`` spectrograms.
    mask : torch.Tensor, optional
        Bool, broadcastable to ``(B, 1, F, T)``; replaces the NaN-derived mask.
    reduction : {'none', 'mean', 'sum'}, default 'mean'
        Over bands. ``'none'`` returns ``(F,)``.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    m = None if mask is None else _bands_as_channels(mask.expand_as(target))
    return corrcoef(_bands_as_channels(pred), _bands_as_channels(target),
                    mask=m, reduction=reduction)


def mel_filterbank(freqs_hz: Union[Sequence[float], np.ndarray, torch.Tensor],
                   n_mels: int) -> torch.Tensor:
    """Triangular (HTK-mel) filters evaluated at the given bin frequencies.

    Returns an ``(n_mels, F)`` matrix mapping linear-frequency power to mel-band
    power. Filter edges are equally spaced on the mel scale between the first
    and last bin. Raises if a filter falls between bins (too many mels for the
    frequency resolution).
    """
    f = torch.as_tensor(np.asarray(freqs_hz, dtype=np.float64))
    mel = lambda hz: 2595.0 * torch.log10(1.0 + hz / 700.0)          # noqa: E731
    edges = torch.linspace(float(mel(f[0])), float(mel(f[-1])), n_mels + 2, dtype=torch.float64)
    m = mel(f)
    lo, ce, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    fb = torch.clamp(torch.minimum((m - lo) / (ce - lo), (hi - m) / (hi - ce)), min=0.0)
    if (fb.sum(1) == 0).any():
        raise ValueError(f"n_mels={n_mels} is too fine for {len(f)} bins; use fewer mel bands")
    return fb.float()


def _dct2_ortho(n_in: int, n_out: int) -> torch.Tensor:
    """Orthonormal DCT-II basis, ``(n_out, n_in)``."""
    k = torch.arange(n_out, dtype=torch.float64)[:, None]
    n = torch.arange(n_in, dtype=torch.float64)[None, :]
    D = torch.cos(math.pi * k * (2 * n + 1) / (2 * n_in)) * math.sqrt(2.0 / n_in)
    D[0] /= math.sqrt(2.0)
    return D.float()


def mel_cepstral_distortion(pred: torch.Tensor, target: torch.Tensor,
                            mask: Optional[torch.Tensor] = None, *,
                            freqs_hz=None, n_mels: int = 32, n_ceps: int = 13,
                            alpha: float = MCD_DB_SCALE) -> torch.Tensor:
    """Mel-cepstral distortion (MCD) between two log-power spectrograms.

    Per frame ``t``: mel-cepstra ``c = DCT-II(log mel power)`` (orthonormal),
    distance ``alpha * sqrt(sum_{d=1}^{n_ceps} (c_d - c'_d)^2)`` — the energy
    term ``c_0`` is excluded, so overall loudness offsets are ignored. Returns
    the mean over valid frames (a scalar; lower is better).

    Parameters
    ----------
    pred, target : torch.Tensor
        ``(B, 1, F, T)`` **natural-log power** spectrograms.
    mask : torch.Tensor, optional
        Bool ``(B, T)`` (or broadcastable) frame mask; default = frames whose
        target has no NaN.
    freqs_hz : sequence of float, optional
        Centre frequency of each of the ``F`` bins. If given, the inputs are
        treated as *linear-frequency* log spectrograms and warped to ``n_mels``
        mel bands first (``exp`` → triangular mel filters → ``log``). If
        ``None``, the inputs are assumed to be mel-scaled already.
    n_mels : int, default 32
        Mel bands used when ``freqs_hz`` is given.
    n_ceps : int, default 13
        Cepstral coefficients compared (``c_1 … c_{n_ceps}``).
    alpha : float, default ``10 * sqrt(2) / ln(10)``
        Scaling. The default is the dB convention common in speech-synthesis
        and voice-conversion work; Kubichek (1993), who introduced the measure,
        used ``alpha = 1``. Report which one you use.

    References
    ----------
    Kubichek, R. (1993). "Mel-cepstral distance measure for objective speech
    quality assessment." *Proc. IEEE Pacific Rim Conference on Communications,
    Computers and Signal Processing*, 1, 125–128.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    p, g = pred[:, 0], target[:, 0]                              # (B, F, T)
    if mask is None:
        frame_ok = ~g.isnan().any(dim=1)                         # (B, T)
    else:
        frame_ok = mask.reshape(mask.shape[0], -1) if mask.dim() > 2 else mask
        frame_ok = frame_ok.expand(g.shape[0], g.shape[-1]).bool()
    g = torch.nan_to_num(g, nan=0.0)
    p = torch.nan_to_num(p, nan=0.0)
    if freqs_hz is not None:
        fb = mel_filterbank(freqs_hz, n_mels).to(p.device, p.dtype)   # (M, F)
        to_mel = lambda s: torch.log(torch.einsum("mf,bft->bmt", fb, torch.exp(s)).clamp(min=1e-10))  # noqa: E731
        p, g = to_mel(p), to_mel(g)
    M = p.shape[1]
    if n_ceps >= M:
        raise ValueError(f"n_ceps={n_ceps} must be < number of (mel) bands {M}")
    D = _dct2_ortho(M, n_ceps + 1)[1:].to(p.device, p.dtype)     # drop c0
    dc = torch.einsum("cm,bmt->bct", D, p - g)                   # (B, n_ceps, T)
    per_frame = alpha * torch.sqrt((dc ** 2).sum(1))             # (B, T)
    if not frame_ok.any():
        return per_frame.new_full((), float("nan"))
    return per_frame[frame_ok].mean()
