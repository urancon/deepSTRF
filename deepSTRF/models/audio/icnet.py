"""ICNet — full encoder+decoder model from Drakopoulos et al. (2025)."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepSTRF.models.audio.audio_model import AudioEncodingModel
from deepSTRF.models.wav2spec.sincnet import SincNet


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
            f"Cannot factor total={total} into {n_layers} strides; got "
            f"{strides} with product {product}. Pass an explicit "
            f"``encoder_strides`` list."
        )
    return strides


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


class _ICNetEncoder(nn.Module):
    """SincNet + 5 strided causal convs + bottleneck. The wav2spec slot
    value for :class:`ICNet`. Intentionally kept module-private — users who
    want the IC encoder as a generic feature extractor should instantiate
    :class:`ICNet` and use its ``wav2spec`` attribute directly.
    """

    def __init__(self, audio_fs: int, dt_ms: float,
                 n_filters: int, sincnet_kernel_size: int,
                 encoder_channels: int, encoder_kernel_size: int,
                 n_encoder_layers: int, bottleneck_channels: int,
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

        # SincNet front (stride 1, no envelope: ICNet relies on the downstream
        # conv stack to extract envelopes from the signed bandpass).
        self.sincnet = SincNet(
            audio_fs=audio_fs, n_filters=n_filters,
            kernel_size=sincnet_kernel_size,
            hop_ms=1000.0 / audio_fs,   # stride 1 in samples
            init="mel", activation="symlog", envelope=False,
        )

        # 5 strided conv layers
        in_ch = n_filters
        layers = []
        for s in strides:
            layers.append(_CausalConv1dBlock(
                in_ch, encoder_channels,
                kernel_size=encoder_kernel_size, stride=s,
            ))
            in_ch = encoder_channels
        self.encoder = nn.ModuleList(layers)

        # Bottleneck conv (stride 1)
        self.bottleneck = _CausalConv1dBlock(
            encoder_channels, bottleneck_channels,
            kernel_size=encoder_kernel_size, stride=1,
        )

        self.bottleneck_channels = bottleneck_channels
        self.out_channels = bottleneck_channels   # wav2spec contract

    def extra_repr(self) -> str:
        return (f"audio_fs={self.audio_fs}, dt_ms={self.dt_ms}, "
                f"strides={self.encoder_strides}, out_channels={self.out_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"_ICNetEncoder expects (B, 1, T_audio); got {tuple(x.shape)}"
            )
        # SincNet returns (B, 1, n_filters, T_audio); collapse the explicit
        # C_in=1 axis for the conv stack.
        y = self.sincnet(x).squeeze(1)              # (B, n_filters, T_audio)
        for layer in self.encoder:
            y = layer(y)
        y = self.bottleneck(y)                      # (B, bottleneck, T_neural)
        return y.unsqueeze(1)                       # (B, 1, bottleneck, T_neural)


class ICNet(AudioEncodingModel):
    """End-to-end ICNet (Drakopoulos et al. 2025) ported to deepSTRF.

    Architecture: SincNet (48 filters, K=64, stride 1, symlog) → 5× causal
    ``Conv1d(128 ch, K=64, PReLU)`` at strides that multiply to
    ``audio_fs · dt_ms / 1000`` → bottleneck ``Conv1d(64 ch, K=64, stride 1,
    PReLU)`` → ``Linear(64 → N)`` → softplus (Poisson head, ``N_c = 1`` in
    paper notation).

    Cross-dataset configuration
    ---------------------------
    The paper trains on 24 414 Hz gerbil-IC audio binned at ~1.31 ms (32
    samples per bin, 5 stride-2 conv layers). To use the same architecture
    on a dataset at a different ``(audio_fs, dt_ms)``, the encoder strides
    are auto-factored so they multiply to ``audio_fs · dt_ms / 1000`` (the
    number of audio samples per neural bin). For NS1 (48 kHz / 5 ms) that's
    240 samples / bin and the default factorisation is ``[2, 2, 2, 2, 15]``.
    Pass an explicit ``encoder_strides`` list to override. The layer
    structure (kernel sizes, channel counts, activations) stays
    paper-faithful; only the strides scale with the dataset, per the
    deepSTRF policy of adapting hyperparameters to each dataset's temporal
    resolution.

    The decoder is intentionally simple — paper-faithful (the paper:
    *"the simple linear decoders in ICNet … ensure that the latent
    representation in the bottleneck is constrained to directly reflect
    the dynamics that underlie neural activity"*). The expressivity lives
    in the shared encoder.

    Differences from the paper
    --------------------------
    - Single-branch / time-invariant only. The paper's multi-branch and
      time-variant heads (animal-specific decoders, timestamp-input
      modulation) are out of scope for the deepSTRF v1 port.
    - Poisson head only. The paper's main result uses a categorical
      cross-entropy head with ``N_c = 5`` classes for spike counts in
      ``{0, 1, 2, 3, ≥4}``. The deepSTRF training stack centres on
      rate-based losses; cross-entropy can be added later.
    - No left-context crop. The paper feeds 10 240 audio samples in and
      crops the leftmost 64 frames from the bottleneck output to suppress
      edge effects. deepSTRF's convention is to keep ``T_neural`` output
      frames matching the dataset's response window; causal convs leave
      the first few frames noisier but downstream losses handle that.

    Parameters
    ----------
    audio_fs : int
        Audio sample rate (Hz). Determines the total encoder downsampling.
    out_neurons : int
        Number of output neurons ``N``.
    dt_ms : float, default 5.0
        Target neural bin width in ms. Encoder strides are factored so the
        total downsampling matches ``audio_fs · dt_ms / 1000``.
    n_filters : int, default 48
        SincNet filter count.
    sincnet_kernel_size : int, default 64
    encoder_channels : int, default 128
    encoder_kernel_size : int, default 64
    n_encoder_layers : int, default 5
    bottleneck_channels : int, default 64
        Output channel count of the bottleneck conv.
    encoder_strides : sequence of int, optional
        Per-layer encoder strides. Default: auto-factor.

    References
    ----------
    Drakopoulos, Pellatt, Sabesan, Xia, Fragner & Lesica (2025). "Modelling
    neural coding in the auditory midbrain with high resolution and
    accuracy." Nature Machine Intelligence 7:1478-1493.
    https://doi.org/10.1038/s42256-025-01104-9
    """

    def __init__(self, audio_fs: int, out_neurons: int,
                 dt_ms: float = 5.0,
                 n_filters: int = 48,
                 sincnet_kernel_size: int = 64,
                 encoder_channels: int = 128,
                 encoder_kernel_size: int = 64,
                 n_encoder_layers: int = 5,
                 bottleneck_channels: int = 64,
                 encoder_strides: Optional[Sequence[int]] = None):
        encoder = _ICNetEncoder(
            audio_fs=audio_fs, dt_ms=dt_ms,
            n_filters=n_filters,
            sincnet_kernel_size=sincnet_kernel_size,
            encoder_channels=encoder_channels,
            encoder_kernel_size=encoder_kernel_size,
            n_encoder_layers=n_encoder_layers,
            bottleneck_channels=bottleneck_channels,
            encoder_strides=encoder_strides,
        )
        super().__init__(
            n_frequency_bands=bottleneck_channels,
            # ICNet's decoder is a 1-sample 1x1 projection — there's no STRF
            # window. We set T = 1 so STRF_gradmap (which sizes its null
            # stimulus from this attribute) still returns a sensible shape.
            temporal_window_size=1,
            out_neurons=out_neurons,
            wav2spec=encoder,
        )
        # core stays Identity (set by NeuralModel.__init__).
        # readout: per-timestep linear projection from the 64-dim latent
        # to N output neurons + softplus (Poisson head).
        self.decoder = nn.Linear(bottleneck_channels, out_neurons, bias=True)
        self.readout = self.decoder  # base-class compat (validate() looks here)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Overrides the base template because the bottleneck latent is shaped
        ``(B, 1, 64, T)`` (an explicit C_in axis on top of the latent dim)
        and the paper's decoder is a per-timestep linear map — the canonical
        :class:`STRFReadout` slot doesn't fit cleanly.

        Parameters
        ----------
        x : torch.Tensor
            Mono waveform, shape ``(B, 1, T_audio)``.

        Returns
        -------
        torch.Tensor
            Predicted spike rate, shape ``(B, N, 1, T_neural)``. Non-negative
            (softplus output) — pair with :func:`~deepSTRF.metrics.poisson_loss`.
        """
        y = self.wav2spec(x)                 # (B, 1, bottleneck, T_neural)
        y = y.squeeze(1)                     # (B, bottleneck, T_neural)
        y = y.transpose(-1, -2)              # (B, T_neural, bottleneck)
        y = self.decoder(y)                  # (B, T_neural, N)
        y = F.softplus(y)                    # non-negative rate
        y = y.transpose(-1, -2)              # (B, N, T_neural)
        return y.unsqueeze(-2)               # (B, N, 1, T_neural)
