"""Learnable parametric bandpass front-end (Ravanelli & Bengio 2018)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepSTRF.models.scales import Hz_to_mel, mel_to_Hz


class SincNet(nn.Module):
    """Strictly-causal SincNet front-end for the ``wav2spec`` slot.

    Each of the ``n_filters`` channels is a parametric bandpass filter with
    two learnable parameters — a low cutoff ``f1`` and a high cutoff ``f2``
    in Hz. Time-domain impulse response (Ravanelli & Bengio 2018, eq. 4):

        g[n; f1, f2] = 2 f2 · sinc(2π f2 n / fs) - 2 f1 · sinc(2π f1 n / fs)

    multiplied by a Hamming window. The bank is convolved across the input
    audio with stride equal to the dataset's hop, producing one frame per
    neural bin.

    Causality
    ---------
    Strict: the input is left-padded by ``kernel_size - hop`` zeros and a
    standard ``Conv1d(stride=hop, padding=0)`` runs the filterbank. Output
    frame ``t`` then reads audio samples ``[t·hop - K + hop : (t+1)·hop]``
    of the original waveform — ends exactly at the last sample of neural
    bin ``t``, never crosses the boundary into bin ``t+1``.

    Parameters
    ----------
    audio_fs : int
        Input audio sample rate in Hz.
    n_filters : int, default 34
        Number of bandpass filters (= ``out_channels``). Default matches
        NS1's precomputed-spec layout.
    kernel_size : int, default 251
        Time-domain filter length in samples. Must be ``>= hop`` (else the
        causal-padding scheme can't cover the full neural bin). 251 samples
        ≈ 15.7 ms at 16 kHz is the Ravanelli default; ICNet uses 64.
    hop_ms : float, default 5.0
        Hop length in ms (matches dataset ``dt_ms``).
    f_min : float, default 300.0
    f_max : float, optional
        Frequency limits (Hz) used for initial filter spacing. ``None`` →
        Nyquist. Filters can drift beyond these during training; the only
        hard clamps are at 0 Hz and audio_fs/2.
    init : {'mel', 'linear'}, default 'mel'
        Initial filter edge spacing.
    activation : {'symlog', 'logabs', 'none'}, default 'symlog'
        Output nonlinearity. ``'symlog'`` = ``sgn(x)·log(|x|+1)`` (the ICNet
        variant — sign-preserving log-compression); ``'logabs'`` = ``log(|x|+1)``
        (standard SincNet, half-wave rectified); ``'none'`` = identity.

    References
    ----------
    Ravanelli & Bengio (2018). "Speaker Recognition from Raw Waveform with
    SincNet." IEEE SLT.

    Drakopoulos et al. (2025). "Modelling neural coding in the auditory
    midbrain with high resolution and accuracy." Nat. Mach. Intell.
    (ICNet's first layer is a SincNet with the ``symlog`` activation.)
    """

    def __init__(self, audio_fs: int, n_filters: int = 34,
                 kernel_size: int = 251, hop_ms: float = 5.0,
                 f_min: float = 300.0, f_max: float | None = None,
                 init: str = "mel", activation: str = "symlog"):
        super().__init__()
        if audio_fs <= 0:
            raise ValueError(f"audio_fs must be positive (got {audio_fs})")
        if activation not in ("symlog", "logabs", "none"):
            raise ValueError(f"activation must be 'symlog', 'logabs', or 'none' (got {activation!r})")
        if init not in ("mel", "linear"):
            raise ValueError(f"init must be 'mel' or 'linear' (got {init!r})")

        self.audio_fs = int(audio_fs)
        self.n_filters = int(n_filters)
        self.out_channels = self.n_filters
        self.kernel_size = int(kernel_size)
        self.hop = max(1, int(round(audio_fs * hop_ms / 1000.0)))
        # left-pad = max(0, K - hop). When K < hop the output frames are
        # genuinely causal but leave a (hop - K)-sample gap at the *end* of
        # each neural bin (the filter just doesn't reach there). Stand-alone
        # wav2spec users typically want K >= hop; the K < hop regime is
        # supported for internal-block reuse (e.g. SincNet stage of ICNet,
        # where the surrounding conv stack does the temporal aggregation).
        self.left_pad = max(0, self.kernel_size - self.hop)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else audio_fs / 2.0
        self.activation = activation

        # ---- initial filter edges (n_filters + 1 edges -> n_filters bands) ----
        f_max_t = torch.tensor(self.f_max, dtype=torch.float32)
        f_min_t = torch.tensor(self.f_min, dtype=torch.float32)
        if init == "mel":
            edges = mel_to_Hz(torch.linspace(
                Hz_to_mel(f_min_t).item(), Hz_to_mel(f_max_t).item(),
                self.n_filters + 1,
            ))
        else:
            edges = torch.linspace(self.f_min, self.f_max, self.n_filters + 1)
        f1_init = edges[:-1].contiguous()
        f2_init = edges[1:].contiguous()

        # Reparameterise as (low_hz, band_hz) so f1 >= 0 and f2 > f1 by
        # construction (we take abs() at forward time on band_hz).
        self.low_hz_ = nn.Parameter(f1_init.clone())
        self.band_hz_ = nn.Parameter((f2_init - f1_init).clone())

        # n axis for the sinc, centered at 0 so the analytic value at n=0
        # falls inside the kernel. Length = kernel_size.
        half = (self.kernel_size - 1) / 2.0
        n = torch.arange(self.kernel_size, dtype=torch.float32) - half
        self.register_buffer("_n", n.view(1, -1))      # (1, K)

        # Hamming window
        win = 0.54 - 0.46 * torch.cos(
            2 * math.pi * torch.arange(self.kernel_size, dtype=torch.float32)
            / (self.kernel_size - 1)
        )
        self.register_buffer("_window", win.view(1, -1))  # (1, K)

    def extra_repr(self) -> str:
        return (f"audio_fs={self.audio_fs}, n_filters={self.n_filters}, "
                f"kernel_size={self.kernel_size}, hop={self.hop}, "
                f"f_range=({self.f_min}, {self.f_max}), activation={self.activation!r}")

    @property
    def f1(self) -> torch.Tensor:
        """Lower cutoffs (Hz), clamped to [1, fs/2 - 1]."""
        return torch.clamp(self.low_hz_.abs(), min=1.0, max=self.audio_fs / 2.0 - 1.0)

    @property
    def f2(self) -> torch.Tensor:
        """Upper cutoffs (Hz), guaranteed > f1 and <= fs/2."""
        return torch.clamp(self.f1 + self.band_hz_.abs(), max=self.audio_fs / 2.0)

    def _build_filters(self) -> torch.Tensor:
        """Build the time-domain bandpass kernels.

        Returns
        -------
        torch.Tensor
            Shape ``(n_filters, 1, kernel_size)`` ready for Conv1d.
        """
        fs = self.audio_fs
        f1 = self.f1.view(-1, 1)  # (n_filters, 1)
        f2 = self.f2.view(-1, 1)
        n = self._n  # (1, K)
        # 2 f * sinc(2π f n / fs) = sin(2π f n / fs) / (π n / fs)  for n != 0
        # at n = 0, returns 2 f.
        two_pi_over_fs = 2.0 * math.pi / fs
        # sin terms (nonzero-n branch)
        sin1 = torch.sin(two_pi_over_fs * f1 * n)  # (n_filters, K)
        sin2 = torch.sin(two_pi_over_fs * f2 * n)
        # safe denominator: replace n=0 with 1 to avoid div0; we overwrite the
        # corresponding row afterwards.
        denom = math.pi * n / fs                   # (1, K)
        mask = (n.abs() < 0.5)  # (1, K) — True only at the centre tap
        denom_safe = torch.where(mask, torch.ones_like(denom), denom)
        h_lp1 = sin1 / denom_safe
        h_lp2 = sin2 / denom_safe
        h = h_lp2 - h_lp1                            # bandpass = lp(f2) - lp(f1)
        # overwrite centre-tap with the analytic limit value 2(f2 - f1)
        centre = 2.0 * (f2 - f1).view(-1, 1)  # (n_filters, 1)
        h = torch.where(mask, centre.expand_as(h), h)
        # window + reshape for Conv1d
        h = h * self._window
        return h.unsqueeze(1)  # (n_filters, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"SincNet expects (B, 1, T_audio); got {tuple(x.shape)}"
            )
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        kernels = self._build_filters()
        y = F.conv1d(x, kernels, stride=self.hop)   # (B, n_filters, T_neural)

        if self.activation == "symlog":
            y = torch.sign(y) * torch.log1p(y.abs())
        elif self.activation == "logabs":
            y = torch.log1p(y.abs())
        # 'none' -> identity

        return y.unsqueeze(1)  # (B, 1, n_filters, T_neural) — explicit C_in axis
