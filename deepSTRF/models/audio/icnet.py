"""ICNet — full encoder+decoder model from Drakopoulos et al. (2025)."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from deepSTRF.models.audio.audio_model import AudioEncodingModel
from deepSTRF.models.wav2spec.icnet_frontend import ICNetFrontend


class ICNet(AudioEncodingModel):
    """End-to-end ICNet (Drakopoulos et al. 2025) ported to deepSTRF.

    ``wav2spec`` = :class:`ICNetFrontend` (SincNet + 5 strided causal convs +
    bottleneck → 64-channel latent at neural rate). ``readout`` = a single
    kernel-1 linear projection from the 64-dim latent to the ``out_neurons``
    output cells, followed by softplus to enforce a non-negative rate
    (Poisson head, ``N_c = 1`` in paper notation).

    Cross-dataset configuration
    ---------------------------
    The paper trains on 24 414 Hz gerbil-IC audio binned at ~1.31 ms (32
    samples per bin, 5 stride-2 conv layers). To use the same architecture
    on a dataset at a different ``(audio_fs, dt_ms)``, the encoder strides
    are auto-factored so they multiply to ``audio_fs · dt_ms / 1000`` (the
    number of audio samples per neural bin). For NS1 (16 kHz / 5 ms) that's
    80 samples / bin and the default factorisation is ``[2, 2, 2, 2, 5]``.
    Pass an explicit ``encoder_strides`` list to override. The layer
    structure (kernel sizes, channel counts, activations) stays paper-faithful;
    only the strides scale with the dataset, per the deepSTRF policy of
    adapting hyperparameters to each dataset's temporal resolution.

    The decoder is intentionally simple — paper-faithful (``"the simple
    linear decoders in ICNet … ensure that the latent representation in the
    bottleneck is constrained to directly reflect the dynamics that underlie
    neural activity"``). The expressivity lives in the shared encoder.

    Differences from the paper
    --------------------------
    - Single-branch / time-invariant only. The paper's multi-branch and
      time-variant heads (animal-specific decoders, timestamp-input modulation)
      are out of scope for the deepSTRF v1 port — most deepSTRF datasets are
      single-session and don't have the long-duration non-stationarity ICNet
      was designed to handle.
    - Poisson head only. The paper's main result uses a categorical
      cross-entropy head with ``N_c = 5`` classes for spike counts in
      ``{0, 1, 2, 3, ≥4}``. The deepSTRF training stack centres on rate-based
      losses (PSTH prediction), so we ship the Poisson variant; cross-entropy
      can be added later if needed.
    - No left-context crop. The paper feeds 10 240 audio samples in and crops
      the leftmost 64 frames (≈ 2 048 audio samples) from the bottleneck
      output to suppress edge effects. deepSTRF's convention is to keep
      ``T_neural`` output frames matching the dataset's response window;
      causal convs leave the first few frames noisier but downstream losses
      handle that naturally.

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
        frontend = ICNetFrontend(
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
            wav2spec=frontend,
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
        y = self.wav2spec(x)                # (B, 1, bottleneck, T_neural)
        y = y.squeeze(1)                    # (B, bottleneck, T_neural)
        y = y.transpose(-1, -2)             # (B, T_neural, bottleneck)
        y = self.decoder(y)                 # (B, T_neural, N)
        y = torch.nn.functional.softplus(y) # non-negative rate
        y = y.transpose(-1, -2)             # (B, N, T_neural)
        return y.unsqueeze(-2)              # (B, N, 1, T_neural)
