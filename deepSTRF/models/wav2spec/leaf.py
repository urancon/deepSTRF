"""Strictly-causal LEAF front-end (Zeghidour et al. 2021) for the ``wav2spec``
slot.

LEAF ("LEarnable Audio Frontend") is a fully-learnable cochleagram with three
learnable stages:

1. a complex **Gabor** filterbank (learnable centre frequency + bandwidth per
   channel) whose complex magnitude is a smooth envelope (no rectification);
2. a learnable **Gaussian lowpass pooling** that downsamples to the frame rate;
3. learnable per-channel **sPCEN** (PCEN with learnable α, δ, r and smoother).

This module is the strictly-causal variant: the Gabor / pooling convolutions
use left-only padding and the sPCEN smoother is a forward IIR, so output frame
``t`` depends only on audio up to ``(t+1)·hop``. Reference:
Zeghidour, Teboul, de Chaumont Quitry & Tagliasacchi, "LEAF: A Learnable
Frontend for Audio Classification", ICLR 2021.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepSTRF.models.scales import Hz_to_mel, mel_to_Hz, ERB_bandwidth


def _inv_softplus(y: float) -> float:
    """Raw value r such that softplus(r) == y (for positive-param reparam)."""
    return math.log(math.expm1(y))


class CausalLEAF(nn.Module):
    """Strictly-causal learnable LEAF front-end for the ``wav2spec`` slot.

    Output shape ``(B, 1, n_filters, T_neural)`` like the other front-ends.

    Parameters
    ----------
    audio_fs : int
        Input audio sample rate (Hz).
    n_filters : int, default 34
        Number of Gabor channels (= ``out_channels``).
    hop_ms : float, default 5.0
        Hop in ms (= dataset ``dt_ms``).
    f_min, f_max : float
        Centre-frequency range (Hz) for the mel-spaced Gabor initialisation.
    kernel_ms : float, default 15.0
        Gabor impulse-response length in ms.
    pool_ms : float, optional
        Gaussian-pooling window length in ms. ``None`` → ``4·hop_ms``.
    learn_filters, learn_pooling, learn_pcen : bool, default True
        Toggle learnability of each stage (all on = full LEAF).

    Notes
    -----
    Causal divergences from the published (non-causal) LEAF: the Gabor and
    Gaussian-pooling kernels use left-only padding (so each output frame reads
    only past+present audio), and the Gaussian pool is a one-sided
    (recent-weighted) window rather than symmetric. The sPCEN smoother is the
    standard causal forward IIR. Everything else (complex Gabor, learnable
    bandwidths, learnable per-channel PCEN) matches the paper.
    """

    def __init__(self, audio_fs: int, n_filters: int = 34, hop_ms: float = 5.0,
                 f_min: float = 60.0, f_max: Optional[float] = 22627.0,
                 kernel_ms: float = 15.0, pool_ms: Optional[float] = None,
                 learn_filters: bool = True, learn_pooling: bool = True,
                 learn_pcen: bool = True):
        super().__init__()
        if audio_fs <= 0:
            raise ValueError(f"audio_fs must be positive (got {audio_fs})")

        self.audio_fs = int(audio_fs)
        self.n_filters = int(n_filters)
        self.out_channels = self.n_filters
        self.hop = max(1, int(round(audio_fs * hop_ms / 1000.0)))
        self.kernel_size = max(self.hop, int(round(audio_fs * kernel_ms / 1000.0)))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1                      # odd, symmetric n-grid
        pool_ms = pool_ms if pool_ms is not None else 4.0 * hop_ms
        self.pool_size = max(self.hop, int(round(audio_fs * pool_ms / 1000.0)))
        self.left_pad = self.kernel_size - 1           # causal gabor padding
        self.pool_left_pad = self.pool_size - self.hop  # causal pool padding
        self.eps = 1e-6

        nyq = audio_fs / 2.0
        f_max = min(float(f_max), nyq) if f_max is not None else nyq
        self.f_min, self.f_max = float(f_min), float(f_max)

        # ---- Gabor init: mel-spaced centres, ERB-matched bandwidths ----
        fc = mel_to_Hz(torch.linspace(Hz_to_mel(torch.tensor(self.f_min)).item(),
                                      Hz_to_mel(torch.tensor(self.f_max)).item(),
                                      self.n_filters))             # (F,) Hz
        omega = 2.0 * math.pi * fc / audio_fs                      # rad/sample
        sigma = audio_fs / (2.0 * math.pi * ERB_bandwidth(fc))     # time-std, samples
        self.center_freq_ = nn.Parameter(omega.clone(), requires_grad=learn_filters)
        self.gabor_sigma_raw = nn.Parameter(
            torch.tensor([_inv_softplus(float(s)) for s in sigma]),
            requires_grad=learn_filters)

        # ---- Gaussian pooling init (one-sided, recent-weighted) ----
        self.pool_sigma_raw = nn.Parameter(
            torch.full((self.n_filters,), _inv_softplus(self.pool_size / 4.0)),
            requires_grad=learn_pooling)

        # ---- sPCEN init (the values that worked for gammatone+PCEN) ----
        self.pcen_alpha_raw = nn.Parameter(
            torch.full((self.n_filters,), _inv_softplus(0.96)), requires_grad=learn_pcen)
        self.pcen_delta_raw = nn.Parameter(
            torch.full((self.n_filters,), _inv_softplus(2.0)), requires_grad=learn_pcen)
        self.pcen_root_raw = nn.Parameter(
            torch.full((self.n_filters,), _inv_softplus(0.5)), requires_grad=learn_pcen)
        # smoother coeff s in (0,1) via sigmoid; init 0.04
        self.pcen_s_logit = nn.Parameter(
            torch.full((self.n_filters,), math.log(0.04 / (1 - 0.04))),
            requires_grad=learn_pcen)

        n = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2.0
        self.register_buffer("_n", n.view(1, -1))                  # (1, K) centred
        pj = torch.arange(self.pool_size, dtype=torch.float32)     # 0..L-1
        self.register_buffer("_pool_j", pj.view(1, -1))            # (1, L)

    def extra_repr(self) -> str:
        return (f"audio_fs={self.audio_fs}, n_filters={self.n_filters}, hop={self.hop}, "
                f"kernel_size={self.kernel_size}, pool_size={self.pool_size}")

    # ---- learnable params via positivity reparam ----
    @property
    def gabor_sigma(self):
        return F.softplus(self.gabor_sigma_raw).clamp(min=1.0).view(-1, 1)

    @property
    def pool_sigma(self):
        return F.softplus(self.pool_sigma_raw).clamp(min=0.5).view(-1, 1)

    def _build_gabor(self):
        n = self._n                                               # (1, K)
        gauss = torch.exp(-0.5 * (n / self.gabor_sigma) ** 2)     # (F, K)
        omega = self.center_freq_.clamp(0.01, math.pi).view(-1, 1)
        norm = gauss.sum(dim=1, keepdim=True).clamp(min=1e-8)
        real = (torch.cos(omega * n) * gauss) / norm              # (F, K)
        imag = (torch.sin(omega * n) * gauss) / norm
        return real.unsqueeze(1), imag.unsqueeze(1)               # (F,1,K) each

    def _gaussian_pool(self, env: torch.Tensor) -> torch.Tensor:
        """Causal, learnable per-channel Gaussian lowpass + stride-hop pooling.
        Window is one-sided: the most-recent tap (rightmost) has the highest
        weight, older taps decay with the learnable per-channel sigma."""
        # distance from the most-recent tap (j = L-1) going into the past
        dist = (self.pool_size - 1) - self._pool_j                # (1, L)
        w = torch.exp(-0.5 * (dist / self.pool_sigma) ** 2)       # (F, L)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernel = w.unsqueeze(1)                                   # (F, 1, L)
        env = F.pad(env, (self.pool_left_pad, 0))
        return F.conv1d(env, kernel, stride=self.hop, groups=self.n_filters)

    def _spcen(self, E: torch.Tensor) -> torch.Tensor:
        """Causal per-channel learnable sPCEN on E of shape (B, F, T).

        The forward EMA ``M(t) = (1-s) M(t-1) + s E(t)`` (per-channel learnable
        ``s``) is a first-order causal IIR; computed with batched
        ``torchaudio.lfilter`` (per-channel coeffs) for speed. lfilter uses a
        zero initial state, so ``M(0) = s·E(0)`` — a negligible transient at the
        first frame."""
        import torchaudio.functional as taF
        s = torch.sigmoid(self.pcen_s_logit)                      # (F,) in (0,1)
        a = torch.stack([torch.ones_like(s), -(1.0 - s)], dim=1)  # (F, 2)
        b = torch.stack([s, torch.zeros_like(s)], dim=1)          # (F, 2)
        M = taF.lfilter(E, a, b, clamp=False)                     # (B, F, T)
        alpha = F.softplus(self.pcen_alpha_raw).view(1, -1, 1)
        delta = F.softplus(self.pcen_delta_raw).view(1, -1, 1)
        root = F.softplus(self.pcen_root_raw).view(1, -1, 1).clamp(min=1e-3)
        smooth = (self.eps + M) ** alpha
        return (E / smooth + delta) ** root - delta ** root

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"CausalLEAF expects (B, 1, T_audio); got {tuple(x.shape)}")
        if x.shape[-1] % self.hop != 0:
            raise ValueError(
                f"CausalLEAF: input length T_audio={x.shape[-1]} is not a multiple "
                f"of hop={self.hop}. Construct it with audio_fs=dataset.audio_fs and "
                f"hop_ms=dataset.dt (or via make_wav2spec(..., dt_ms=ds.dt))."
            )
        real, imag = self._build_gabor()
        xp = F.pad(x, (self.left_pad, 0))
        xr = F.conv1d(xp, real)                                   # (B, F, T_audio)
        xi = F.conv1d(xp, imag)
        env = torch.sqrt(xr * xr + xi * xi + 1e-9)               # complex magnitude
        env = self._gaussian_pool(env)                           # (B, F, T_neural)
        out = self._spcen(env.clamp(min=0.0))
        return out.unsqueeze(1)                                   # (B, 1, F, T_neural)
