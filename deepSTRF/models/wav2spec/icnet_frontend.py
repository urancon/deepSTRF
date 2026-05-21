"""ICNet's convolutional encoder, packaged as a ``wav2spec`` module.

Drakopoulos et al. (2025) define ICNet as a shared SincNet+conv encoder
mapping a sound waveform to a low-dimensional ``(N_b, T_b)`` bottleneck,
followed by an animal-specific linear decoder. The encoder generalises
across animals and sounds and is positioned by the paper as a generic
auditory front-end (their ASR experiment in Fig. 3e uses the bottleneck
features in place of a mel spectrogram). This module exposes that
encoder so other deepSTRF audio models can mount it in their ``wav2spec``
slot — see :class:`~deepSTRF.models.audio.icnet.ICNet` for the full
end-to-end model.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sincnet import SincNet


def _factor_into_strides(total: int, n_layers: int) -> list[int]:
    """Choose a length-``n_layers`` stride list that multiplies to ``total``.

    Default heuristic: emit as many 2s as possible, put any remaining factor
    at the end. Matches the paper's ``[2,2,2,2,2]`` for ``total = 32`` and
    NS1's ``[2,2,2,2,5]`` for ``total = 80``.
    """
    if total < 1 or n_layers < 1:
        raise ValueError(f"total ({total}) and n_layers ({n_layers}) must be >= 1")
    strides = [1] * n_layers
    remaining = total
    # peel off 2s left-to-right while remaining is even
    for i in range(n_layers - 1):
        if remaining % 2 == 0 and remaining // 2 >= 1:
            strides[i] = 2
            remaining //= 2
        else:
            break
    strides[-1] = remaining
    product = 1
    for s in strides:
        product *= s
    if product != total:
        raise ValueError(
            f"Cannot factor total={total} into {n_layers} strides; "
            f"got {strides} with product {product}. Pass an explicit "
            f"``encoder_strides`` list."
        )
    return strides


class ICNetFrontend(nn.Module):
    """Convolutional encoder of ICNet (Drakopoulos et al. 2025).

    ``SincNet(48 filters, K=64, stride=1, symlog)`` → 5× ``CausalConv1d(
    128 ch, K=64, stride=s_i, PReLU)`` → bottleneck ``CausalConv1d(64 ch,
    K=64, stride=1, PReLU)``. All convs are strictly causal (left-only
    padding); output shape is ``(B, 1, 64, T_neural)`` so the module slots
    into the ``wav2spec`` contract of :class:`AudioEncodingModel`.

    The 5 encoder strides multiply to ``audio_fs * dt_ms / 1000`` (samples
    per neural bin). Paper defaults — 24 414 Hz / 1.31 ms — give the canonical
    ``[2,2,2,2,2]`` (total ÷32). NS1's 16 kHz / 5 ms config has total ÷80
    and uses ``[2,2,2,2,5]``. Pass ``encoder_strides=...`` to override.

    Parameters
    ----------
    audio_fs : int
        Audio sample rate (Hz).
    dt_ms : float, default 5.0
        Target neural bin width in ms. The encoder strides will multiply to
        ``round(audio_fs * dt_ms / 1000)``.
    n_filters : int, default 48
        SincNet filter count (frontend bandpass channels).
    sincnet_kernel_size : int, default 64
        SincNet time-domain kernel length (samples).
    encoder_channels : int, default 128
        Channel count of each 1D conv layer in the encoder stack.
    encoder_kernel_size : int, default 64
        Kernel length (samples) of each encoder conv layer.
    n_encoder_layers : int, default 5
        Number of strided conv layers between SincNet and the bottleneck.
    bottleneck_channels : int, default 64
        Output channel count of the bottleneck conv — equals
        ``out_channels`` of this module.
    encoder_strides : sequence of int, optional
        Per-layer strides. Default: auto-factor from
        ``audio_fs * dt_ms / 1000``.
    """

    def __init__(self, audio_fs: int, dt_ms: float = 5.0,
                 n_filters: int = 48,
                 sincnet_kernel_size: int = 64,
                 encoder_channels: int = 128,
                 encoder_kernel_size: int = 64,
                 n_encoder_layers: int = 5,
                 bottleneck_channels: int = 64,
                 encoder_strides: Optional[Sequence[int]] = None):
        super().__init__()
        if audio_fs <= 0 or dt_ms <= 0:
            raise ValueError(f"audio_fs ({audio_fs}) and dt_ms ({dt_ms}) must be positive")

        self.audio_fs = int(audio_fs)
        self.dt_ms = float(dt_ms)
        total = int(round(audio_fs * dt_ms / 1000.0))
        if encoder_strides is None:
            strides = _factor_into_strides(total, n_encoder_layers)
        else:
            strides = [int(s) for s in encoder_strides]
            product = 1
            for s in strides:
                product *= s
            if product != total:
                raise ValueError(
                    f"encoder_strides {strides} multiply to {product}, but "
                    f"audio_fs ({audio_fs}) × dt_ms ({dt_ms}) ÷ 1000 = {total}. "
                    f"The product must equal the per-bin sample count."
                )
        self.encoder_strides = strides

        # ------ SincNet front (stride 1, no envelope: ICNet relies on the
        # downstream conv stack to extract envelopes from the signed bandpass) ------
        self.sincnet = SincNet(
            audio_fs=audio_fs, n_filters=n_filters,
            kernel_size=sincnet_kernel_size,
            hop_ms=1000.0 / audio_fs,   # stride 1 in samples
            init="mel", activation="symlog", envelope=False,
        )
        # SincNet emits (B, 1, n_filters, T_audio); we need (B, n_filters, T_audio)
        # for the Conv1d stack. Strip the C_in axis right after the SincNet call.

        # ------ 5 strided conv layers ------
        in_ch = n_filters
        layers = []
        for s in strides:
            layers.append(_CausalConv1dBlock(
                in_ch, encoder_channels,
                kernel_size=encoder_kernel_size, stride=s,
            ))
            in_ch = encoder_channels
        self.encoder = nn.ModuleList(layers)

        # ------ bottleneck conv (stride 1) ------
        self.bottleneck = _CausalConv1dBlock(
            encoder_channels, bottleneck_channels,
            kernel_size=encoder_kernel_size, stride=1,
        )

        self.bottleneck_channels = bottleneck_channels
        self.out_channels = bottleneck_channels   # wav2spec contract

    def extra_repr(self) -> str:
        return (f"audio_fs={self.audio_fs}, dt_ms={self.dt_ms}, "
                f"strides={self.encoder_strides}, "
                f"out_channels={self.out_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"ICNetFrontend expects (B, 1, T_audio); got {tuple(x.shape)}"
            )
        # SincNet returns (B, 1, n_filters, T_audio); collapse the explicit
        # C_in=1 axis for the conv stack.
        y = self.sincnet(x).squeeze(1)              # (B, n_filters, T_audio)
        for layer in self.encoder:
            y = layer(y)
        y = self.bottleneck(y)                      # (B, bottleneck, T_neural)
        return y.unsqueeze(1)                       # (B, 1, bottleneck, T_neural)


class _CausalConv1dBlock(nn.Module):
    """Conv1d with strict left-padding + PReLU. Output length = T_in // stride."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.left_pad = max(0, kernel_size - stride)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=0, bias=True)
        self.activation = nn.PReLU(num_parameters=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.activation(self.conv(x))
