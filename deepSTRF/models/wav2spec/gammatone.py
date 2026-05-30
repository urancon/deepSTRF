"""Strictly-causal gammatone filterbank front-end for the ``wav2spec`` slot.

The gammatone is the standard linear approximation to the auditory periphery's
frequency analysis (Patterson et al. 1992). Unlike :class:`SincNet` its filters
are *fixed* (not learnable), so it cannot suffer the frozen-cutoff failure mode
SincNet shows on small datasets — it is a principled, biologically-grounded
cochleagram that, paired with half-wave rectification + compression, is directly
comparable to the mel baseline.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepSTRF.models.scales import Hz_to_ERB, ERB_to_Hz, ERB_bandwidth


class CausalGammatone(nn.Module):
    """Strictly-causal gammatone cochleagram, for the ``wav2spec`` slot.

    Pipeline: fixed gammatone bandpass bank (causal impulse responses) →
    rectify → causal envelope pooling → compression. Produces a
    ``(B, 1, n_filters, T_neural)`` log/compressed cochleagram aligned to the
    response bins, exactly like :class:`CausalMelSpectrogram`.

    The gammatone impulse response of order ``n`` at centre frequency ``fc`` is

        g(t) = t^(n-1) · exp(-2π b t) · cos(2π fc t),   t ≥ 0

    with ``b = b_factor · ERB(fc)``. It is zero for ``t < 0``, so the filterbank
    is causal **by construction** — no padding trick is needed beyond the usual
    left-pad to keep the output length a clean multiple of ``hop``.

    Parameters
    ----------
    audio_fs : int
        Input audio sample rate in Hz.
    n_filters : int, default 34
        Number of gammatone channels (= ``out_channels``). Centre frequencies
        are spaced uniformly on the ERB-rate scale between ``f_min`` and
        ``f_max``.
    hop_ms : float, default 5.0
        Hop in ms (= dataset ``dt_ms``); ``hop = round(audio_fs·hop_ms/1000)``.
    f_min, f_max : float
        Centre-frequency range (Hz). Defaults (500, 22627) match the NS1
        Rahman cochleagram. ``f_max`` is clamped to Nyquist.
    order : int, default 4
        Gammatone order ``n`` (4 is the classic cochlear value).
    b_factor : float, default 1.019
        Bandwidth scaling ``b = b_factor · ERB(fc)`` (Holdsworth/Patterson).
    kernel_ms : float, default 20.0
        Impulse-response length in ms. Must be long enough for the lowest-fc
        filter to decay (≈ 20 ms covers a 500 Hz gammatone at order 4).
    rectify : {'halfwave', 'full'}, default 'halfwave'
        Hair-cell rectification: ``relu`` (half-wave) or ``abs`` (full-wave).
    env_window_ms : float, optional
        Width of the causal averaging window that turns the rectified bandpass
        into an envelope. ``None`` → ``2·hop_ms`` (light smoothing). Must be
        ``>= hop_ms``.
    compression : {'log', 'cuberoot', 'pcen', 'none'}, default 'log'
        Output nonlinearity. ``'log'`` matches the mel baseline
        (``log(max(env, log_floor))``); ``'cuberoot'`` is the classic
        loudness power-law ``env**(1/3)``; ``'pcen'`` is causal Per-Channel
        Energy Normalization (Wang et al. 2017), an adaptive automatic-gain
        control that divides each channel by a causal running-mean of its own
        energy before root compression — emphasises onsets / transients.
    log_floor : float, default 1e-4
        Pre-log clamp for ``compression='log'`` (Rahman threshold-clip form).
    pcen_s, pcen_alpha, pcen_delta, pcen_r, pcen_eps : float
        PCEN parameters (only used when ``compression='pcen'``). Defaults are
        the standard fixed values: ``s=0.025`` (smoother coeff), ``alpha=0.98``
        (gain-normalisation strength), ``delta=2.0``, ``r=0.5`` (root), and
        ``eps=1e-6``. ``PCEN = (E / (eps + M)**alpha + delta)**r - delta**r``
        with ``M(t) = (1-s) M(t-1) + s E(t)`` (causal IIR, run at the neural
        rate on the pooled envelope).
    """

    def __init__(self, audio_fs: int, n_filters: int = 34, hop_ms: float = 5.0,
                 f_min: float = 500.0, f_max: Optional[float] = 22627.0,
                 order: int = 4, b_factor: float = 1.019,
                 kernel_ms: float = 20.0, rectify: str = "halfwave",
                 env_window_ms: Optional[float] = None,
                 compression: str = "log", log_floor: float = 1e-4,
                 pcen_s: float = 0.025, pcen_alpha: float = 0.98,
                 pcen_delta: float = 2.0, pcen_r: float = 0.5,
                 pcen_eps: float = 1e-6):
        super().__init__()
        if audio_fs <= 0:
            raise ValueError(f"audio_fs must be positive (got {audio_fs})")
        if rectify not in ("halfwave", "full"):
            raise ValueError(f"rectify must be 'halfwave' or 'full' (got {rectify!r})")
        if compression not in ("log", "cuberoot", "pcen", "none"):
            raise ValueError(
                f"compression must be 'log', 'cuberoot', 'pcen', or 'none' "
                f"(got {compression!r})"
            )
        self.pcen_s = float(pcen_s)
        self.pcen_alpha = float(pcen_alpha)
        self.pcen_delta = float(pcen_delta)
        self.pcen_r = float(pcen_r)
        self.pcen_eps = float(pcen_eps)

        self.audio_fs = int(audio_fs)
        self.n_filters = int(n_filters)
        self.out_channels = self.n_filters
        self.order = int(order)
        self.b_factor = float(b_factor)
        self.rectify = rectify
        self.compression = compression
        self.log_floor = float(log_floor)
        self.hop = max(1, int(round(audio_fs * hop_ms / 1000.0)))
        self.kernel_size = max(self.hop, int(round(audio_fs * kernel_ms / 1000.0)))

        nyquist = audio_fs / 2.0
        self.f_min = float(f_min)
        self.f_max = min(float(f_max), nyquist) if f_max is not None else nyquist

        env_ms = float(env_window_ms) if env_window_ms is not None else 2.0 * hop_ms
        self.env_window = max(1, int(round(audio_fs * env_ms / 1000.0)))
        if self.env_window < self.hop:
            raise ValueError(f"env_window_ms ({env_ms}) must be >= hop_ms ({hop_ms})")
        self.pool_left_pad = self.env_window - self.hop
        # gammatone is one-sided → causal; left-pad K-1 so the stride-1 conv
        # output has length T_audio (each frame reads orig[t-(K-1):t+1]).
        self.left_pad = self.kernel_size - 1

        # ---- centre frequencies on the ERB-rate scale ----
        e_lo = Hz_to_ERB(torch.tensor(self.f_min))
        e_hi = Hz_to_ERB(torch.tensor(self.f_max))
        fc = ERB_to_Hz(torch.linspace(e_lo.item(), e_hi.item(), self.n_filters))  # (F,)
        self.register_buffer("_fc", fc)

        # ---- build + peak-normalise the fixed gammatone kernels ----
        kernels = self._build_gammatone(fc)            # (F, 1, K)
        self.register_buffer("_kernels", kernels)

    def extra_repr(self) -> str:
        return (f"audio_fs={self.audio_fs}, n_filters={self.n_filters}, "
                f"hop={self.hop}, kernel_size={self.kernel_size}, order={self.order}, "
                f"f_range=({self.f_min:.0f}, {self.f_max:.0f}), rectify={self.rectify!r}, "
                f"compression={self.compression!r}")

    def _build_gammatone(self, fc: torch.Tensor) -> torch.Tensor:
        """Build peak-gain-normalised gammatone kernels of shape (F, 1, K)."""
        fs = self.audio_fs
        K = self.kernel_size
        t = torch.arange(K, dtype=torch.float32) / fs            # (K,) seconds, t>=0
        fc = fc.view(-1, 1)                                       # (F, 1)
        b = self.b_factor * ERB_bandwidth(fc)                     # (F, 1) Hz
        env = (t ** (self.order - 1)) * torch.exp(-2.0 * math.pi * b * t)  # (F, K)
        g = env * torch.cos(2.0 * math.pi * fc * t)              # (F, K) real gammatone
        # normalise each filter to unit peak frequency-response magnitude so
        # bands are comparably scaled before compression.
        H = torch.fft.rfft(g, n=4096, dim=1).abs()               # (F, n_fft//2+1)
        peak = H.amax(dim=1, keepdim=True).clamp(min=1e-12)      # (F, 1)
        g = g / peak
        return g.unsqueeze(1)                                    # (F, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"CausalGammatone expects (B, 1, T_audio); got {tuple(x.shape)}"
            )
        if x.shape[-1] % self.hop != 0:
            raise ValueError(
                f"CausalGammatone: input length T_audio={x.shape[-1]} is not a "
                f"multiple of hop={self.hop}. This usually means the wav2spec's "
                f"audio_fs/dt_ms disagree with the dataset's — construct it with "
                f"audio_fs=dataset.audio_fs and hop_ms=dataset.dt (or via "
                f"make_wav2spec(..., audio_fs=ds.audio_fs, dt_ms=ds.dt))."
            )
        x = F.pad(x, (self.left_pad, 0))
        y = F.conv1d(x, self._kernels, stride=1)                 # (B, F, T_audio)
        y = y.abs() if self.rectify == "full" else F.relu(y)     # rectify
        if self.pool_left_pad > 0:
            y = F.pad(y, (self.pool_left_pad, 0))
        y = F.avg_pool1d(y, kernel_size=self.env_window, stride=self.hop)  # (B, F, T_neural)

        if self.compression == "log":
            y = torch.log(y.clamp(min=self.log_floor))
        elif self.compression == "cuberoot":
            y = y.clamp(min=0.0) ** (1.0 / 3.0)
        elif self.compression == "pcen":
            y = self._pcen(y.clamp(min=0.0))
        # 'none' -> identity
        return y.unsqueeze(1)  # (B, 1, F, T_neural)

    def _pcen(self, E: torch.Tensor) -> torch.Tensor:
        """Causal Per-Channel Energy Normalization on ``E`` of shape
        ``(B, F, T_neural)``. The smoother ``M(t) = (1-s) M(t-1) + s E(t)`` is a
        first-order IIR run forward in time (uses only past + present), so the
        whole transform is strictly causal. Computed with
        ``torchaudio.functional.lfilter`` (zero initial state) for speed."""
        import torchaudio.functional as taF
        s = self.pcen_s
        B, F, T = E.shape
        a = torch.tensor([1.0, -(1.0 - s)], dtype=E.dtype, device=E.device)
        b = torch.tensor([s, 0.0], dtype=E.dtype, device=E.device)
        M = taF.lfilter(E.reshape(B * F, T), a, b, clamp=False).reshape(B, F, T)
        smooth = (self.pcen_eps + M) ** self.pcen_alpha
        return (E / smooth + self.pcen_delta) ** self.pcen_r - self.pcen_delta ** self.pcen_r
