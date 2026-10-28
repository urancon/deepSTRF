"""ICNet — full encoder+decoder model from Drakopoulos et al. (2025)."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepSTRF.models.audio.audio_model import AudioEncodingModel
from deepSTRF.models.wav2spec.sincnet import SincNet
from deepSTRF.models.layers import CausalLayerNorm


def _factor_into_strides(total: int, n_layers: int) -> list[int]:
    """Choose a length-``n_layers`` stride list that multiplies to ``total``.

    Strides are chosen to be as evenly *balanced* as possible: each prime factor
    of ``total`` (largest first) is assigned to the currently-smallest slot, and
    the result is returned in descending order (largest downsampling first, so
    the sequence shortens fastest). This spreads the downsampling across every
    layer for *any* ``total`` — including non-2-smooth sample-counts.

    Examples: ``total = 32`` -> ``[2, 2, 2, 2, 2]`` (the paper's uniform
    stride 2, since gerbil-IC audio at 24414 Hz / 762 Hz bins gives exactly
    2**5); ``total = 80`` -> ``[5, 2, 2, 2, 2]``; ``total = 441`` (a 44.1 kHz
    corpus binned at 10 ms) -> ``[7, 7, 3, 3, 1]`` (max stride 7, well under the
    kernel size). A greedy powers-of-two split, by contrast, degenerates on
    non-2-smooth counts — 441 would become ``[1, 1, 1, 1, 441]``, four conv
    layers running at the full audio rate plus a final stride-441/kernel-64 conv
    that skips most of its input. Pass an explicit ``encoder_strides`` list to
    :class:`ICNet` to override.

    .. versionchanged:: 0.1.1
       Balanced factorisation replaces the previous greedy powers-of-two
       heuristic, so auto-factored strides stay small for arbitrary
       ``(audio_fs, dt_ms)``. The stride *order* also changed (largest first);
       this only affects models built without an explicit ``encoder_strides``.
    """
    if total < 1 or n_layers < 1:
        raise ValueError(f"total ({total}) and n_layers ({n_layers}) must be >= 1")
    # prime-factorise total
    factors, d, n = [], 2, total
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    # assign each factor (largest first) to the currently-smallest slot -> balanced
    strides = [1] * n_layers
    for p in sorted(factors, reverse=True):
        i = min(range(n_layers), key=lambda j: strides[j])
        strides[i] *= p
    strides.sort(reverse=True)
    product = 1
    for s in strides:
        product *= s
    if product != total:  # defensive; balanced assignment always multiplies to total
        raise ValueError(
            f"Cannot factor total={total} into {n_layers} strides; got "
            f"{strides} with product {product}. Pass an explicit "
            f"``encoder_strides`` list."
        )
    return strides


# The paper's encoder: 24 414 Hz audio, five stride-2 convs, 65-tap SincConv and 64-tap convs.
_PAPER_FS = 24414.0625


def icnet_kernel_sizes(audio_fs: float, strides: Sequence[int],
                       receptive_field_ms: float = 250.0) -> tuple[int, list[int], float]:
    """Kernel lengths that give ICNet the same layer durations at any sample rate.

    Each layer keeps the paper's duration ratios (SincConv 65 taps and convs of 64 taps at
    24 414 Hz with stride-2 layers, i.e. 2.7 ms for the SincConv, then 2.6, 5.2, 10.5, 21,
    42 ms and 84 ms for the bottleneck, with 5 layers), scaled by one common factor: the largest one
    whose total receptive field is at most ``receptive_field_ms``. Each kernel is at least
    its layer's stride (2 taps for the SincConv, 1 for the bottleneck).

    Returns ``(sincnet_kernel_size, encoder_kernel_sizes, receptive_field_ms_actual)``,
    where ``encoder_kernel_sizes`` has ``len(strides) + 1`` entries (convs, then bottleneck).
    The paper's own encoder has a receptive field of about 165 ms.
    """
    strides = [int(s) for s in strides]
    n = len(strides)        # layer l of the paper runs at 24 414 / 2**l Hz
    durations = ([65 / _PAPER_FS] + [64 / (_PAPER_FS / 2 ** l) for l in range(n)]
                 + [64 / (_PAPER_FS / 2 ** n)])
    rates, jumps, r, j = [float(audio_fs)], [1], float(audio_fs), 1   # SincConv: input rate
    for s in strides:                                          # each conv: its input rate
        rates.append(r)
        jumps.append(j)
        r, j = r / s, j * s
    rates.append(r)                                            # bottleneck
    jumps.append(j)
    mins = [2] + strides + [1]

    def build(c):
        ks = [max(m, round(c * d * rt)) for d, rt, m in zip(durations, rates, mins)]
        rf = 1 + sum((k - 1) * jp for k, jp in zip(ks, jumps))
        return ks, rf / audio_fs * 1000.0

    lo, hi = 0.01, 100.0
    for _ in range(80):                                        # bisection on the factor
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if build(mid)[1] <= receptive_field_ms else (lo, mid)
    ks, rf = build(lo)
    return ks[0], ks[1:], rf


class _CausalConv1dBlock(nn.Module):
    """Conv1d with strict left-padding (+ optional norm) + PReLU. Output length = T_in // stride.

    ``norm='layer'`` inserts a :class:`CausalLayerNorm` over the channels at every time step
    between the conv and the PReLU (strictly causal, independent of the batch size)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, norm: Optional[str] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.left_pad = max(0, kernel_size - stride)
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=0, bias=True)
        if norm is None:
            self.norm = nn.Identity()
        elif norm == "layer":
            self.norm = CausalLayerNorm(out_channels, dim=1)
        else:
            raise ValueError(f"norm must be None or 'layer', got {norm!r}")
        self.activation = nn.PReLU(num_parameters=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.activation(self.norm(self.conv(x)))


class _ICNetEncoder(nn.Module):
    """SincNet + 5 strided causal convs + bottleneck. The wav2spec slot
    value for :class:`ICNet`. Intentionally kept module-private — users who
    want the IC encoder as a generic feature extractor should instantiate
    :class:`ICNet` and use its ``wav2spec`` attribute directly.
    """

    def __init__(self, audio_fs: int, dt_ms: float,
                 n_filters: int, sincnet_kernel_size: Optional[int],
                 encoder_channels: int, encoder_kernel_size,
                 n_encoder_layers: int, bottleneck_channels: int,
                 encoder_strides: Optional[Sequence[int]] = None,
                 norm: Optional[str] = "layer",
                 receptive_field_ms: Optional[float] = 250.0):
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

        # kernel lengths: explicit ints / lists win; otherwise sized in ms from
        # receptive_field_ms (deepSTRF default), or the paper's 64 taps when it is None
        if receptive_field_ms is not None and (sincnet_kernel_size is None
                                               or encoder_kernel_size is None):
            k_sinc, k_enc, _ = icnet_kernel_sizes(audio_fs, strides, receptive_field_ms)
            sincnet_kernel_size = k_sinc if sincnet_kernel_size is None else sincnet_kernel_size
            encoder_kernel_size = k_enc if encoder_kernel_size is None else encoder_kernel_size
        sincnet_kernel_size = 64 if sincnet_kernel_size is None else sincnet_kernel_size
        encoder_kernel_size = 64 if encoder_kernel_size is None else encoder_kernel_size
        self.sincnet_kernel_size = int(sincnet_kernel_size)

        # SincNet front (stride 1, no envelope: ICNet relies on the downstream
        # conv stack to extract envelopes from the signed bandpass).
        self.sincnet = SincNet(
            audio_fs=audio_fs, n_filters=n_filters,
            kernel_size=self.sincnet_kernel_size,
            hop_ms=1000.0 / audio_fs,   # stride 1 in samples
            init="mel", activation="symlog", envelope=False,
        )

        # per-layer kernels: an int (same for every conv, the paper's 64) or a list of
        # n_encoder_layers + 1 ints (encoder convs, then the bottleneck)
        if isinstance(encoder_kernel_size, int):
            ks = [int(encoder_kernel_size)] * (len(strides) + 1)
        else:
            ks = [int(k) for k in encoder_kernel_size]
            if len(ks) != len(strides) + 1:
                raise ValueError(f"encoder_kernel_size list needs {len(strides) + 1} entries "
                                 f"(encoder convs + bottleneck), got {len(ks)}")
        self.encoder_kernel_sizes = ks

        # 5 strided conv layers
        in_ch = n_filters
        layers = []
        for s, k in zip(strides, ks[:-1]):
            layers.append(_CausalConv1dBlock(
                in_ch, encoder_channels,
                kernel_size=k, stride=s, norm=norm,
            ))
            in_ch = encoder_channels
        self.encoder = nn.ModuleList(layers)

        # Bottleneck conv (stride 1)
        self.bottleneck = _CausalConv1dBlock(
            encoder_channels, bottleneck_channels,
            kernel_size=ks[-1], stride=1, norm=norm,
        )

        self.bottleneck_channels = bottleneck_channels
        self.out_channels = bottleneck_channels   # wav2spec contract

    def extra_repr(self) -> str:
        return (f"audio_fs={self.audio_fs}, dt_ms={self.dt_ms}, "
                f"strides={self.encoder_strides}, sinc_kernel={self.sincnet_kernel_size}, "
                f"kernels={self.encoder_kernel_sizes}, "
                f"out_channels={self.out_channels}")

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

    Architecture: SincNet (48 filters, stride 1, symlog) → 5× causal
    ``Conv1d(128 ch, PReLU)`` at strides that multiply to
    ``audio_fs · dt_ms / 1000`` → bottleneck ``Conv1d(64 ch, stride 1,
    PReLU)`` → ``Linear(64 → N)`` → softplus (Poisson head, ``N_c = 1`` in
    paper notation). By default each conv is followed by a
    :class:`~deepSTRF.models.layers.CausalLayerNorm` and the kernels are sized in
    milliseconds (see below).

    Cross-dataset configuration
    ---------------------------
    The paper trains on 24 414 Hz gerbil-IC audio binned at ~1.31 ms (32
    samples per bin, 5 stride-2 conv layers, 64-tap kernels, ~165 ms
    receptive field). To use the architecture on a dataset at a different
    ``(audio_fs, dt_ms)``:

    - the encoder strides are auto-factored (balanced, see
      :func:`_factor_into_strides`) so they multiply to ``audio_fs · dt_ms / 1000``
      (the number of audio samples per neural bin). For NS1 (48 kHz / 5 ms)
      that's 240 samples / bin and the factorisation ``[5, 4, 3, 2, 2]``. Pass
      an explicit ``encoder_strides`` list to override;
    - the kernel lengths are set by :func:`icnet_kernel_sizes` so that every
      layer keeps the paper's duration ratios and the whole encoder sees
      ``receptive_field_ms`` (250 ms by default) of sound, whatever the sample
      rate and strides. With fixed 64-tap kernels the receptive field would
      instead range from tens of ms to over a second across datasets. Pass
      explicit ``sincnet_kernel_size`` / ``encoder_kernel_size`` to override.

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
    - Normalisation layers (``norm='layer'``, default). The paper has none.
      Without them, training is unstable on datasets whose waveforms come at
      very different levels (Espejo: test cc_norm 0.18 without, 0.62 with).
    - Kernels sized in ms (``receptive_field_ms=250``, default) instead of
      64 taps everywhere.

    ``ICNet(..., norm=None, receptive_field_ms=None)`` gives the paper's
    layer structure (64-tap kernels, no normalisation).

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
    sincnet_kernel_size : int, optional
        SincNet kernel length. Default: from ``receptive_field_ms``
        (64 if that is ``None``).
    encoder_channels : int, default 128
    encoder_kernel_size : int or sequence of int, optional
        One kernel length for every encoder conv and the bottleneck, or a list
        of ``n_encoder_layers + 1`` lengths (encoder convs, then the
        bottleneck). Default: from ``receptive_field_ms`` (64 everywhere, the
        paper's, if that is ``None``).
    n_encoder_layers : int, default 5
    bottleneck_channels : int, default 64
        Output channel count of the bottleneck conv.
    encoder_strides : sequence of int, optional
        Per-layer encoder strides. Default: auto-factor.
    norm : {'layer', None}, default 'layer'
        ``'layer'`` adds a :class:`~deepSTRF.models.layers.CausalLayerNorm`
        over the channels after every encoder / bottleneck conv, before the
        PReLU: strictly causal and independent of the batch size. It keeps every
        layer's activations at a fixed scale (per time step, across channels),
        whatever the level of the input waveform. ``None`` = the paper (no
        normalisation).
    receptive_field_ms : float or None, default 250.0
        Total receptive field of the encoder, used to size the kernels that
        are not given explicitly (:func:`icnet_kernel_sizes`). ``None`` keeps
        the paper's 64-tap kernels.

    .. versionchanged:: 0.1.1
       ``norm='layer'`` and ``receptive_field_ms=250`` are the new defaults
       (previously no normalisation and 64-tap kernels). SincNet filters are
       now peak-normalised as in the reference code.

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
                 sincnet_kernel_size: Optional[int] = None,
                 encoder_channels: int = 128,
                 encoder_kernel_size: Union[int, Sequence[int], None] = None,
                 n_encoder_layers: int = 5,
                 bottleneck_channels: int = 64,
                 encoder_strides: Optional[Sequence[int]] = None,
                 norm: Optional[str] = "layer",
                 receptive_field_ms: Optional[float] = 250.0):
        encoder = _ICNetEncoder(
            audio_fs=audio_fs, dt_ms=dt_ms,
            n_filters=n_filters,
            sincnet_kernel_size=sincnet_kernel_size,
            encoder_channels=encoder_channels,
            encoder_kernel_size=encoder_kernel_size,
            n_encoder_layers=n_encoder_layers,
            bottleneck_channels=bottleneck_channels,
            encoder_strides=encoder_strides,
            norm=norm,
            receptive_field_ms=receptive_field_ms,
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
