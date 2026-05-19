import torch
import torch.nn as nn
import torch.nn.functional

from .audio_model import AudioEncodingModel

import deepSTRF.models.layers as layers
from deepSTRF.models.activations import ParametricSoftplus
from deepSTRF.models.dependencies.lmu import LMU
from deepSTRF.models.dependencies.mamba import MambaBlock, MambaConfig
from deepSTRF.models.prefiltering import AdapTrans
from deepSTRF.models.readouts import STRFReadout, LinearReadout
# S4Block is imported lazily inside StateNet — its module emits noisy stderr
# warnings about missing CUDA extensions and pulls a heavy dependency
# graph; users who don't pick rnn_type='S4' shouldn't pay either cost.


class Linear(AudioEncodingModel):
    """
    The canonical Linear (L) STRF model — a single SpectroTemporal Receptive
    Field convolved with the (optionally prefiltered) input spectrogram.

    All learnable parameters live in the readout (``STRFReadout``), which
    holds the kernel of shape ``(N, C_in, F, T)``, applies it causally via
    left-padding, and follows it with a per-neuron ``nn.BatchNorm1d(N)``
    before the output activation. The model's ``core`` is ``nn.Identity``
    — every learnable scalar has the neuron axis as leading dim, so the
    model is strictly no-shared-params (each neuron's parameters are
    independent of every other neuron's).

    Parameters
    ----------
    n_frequency_bands : int, default 34
        Number of input spectrogram frequency bands ``F``.
    temporal_window_size : int, default 9
        STRF temporal extent ``T`` in frames.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    output_activation : nn.Module, optional
        Pointwise nonlinearity applied at the readout output. Default
        ``nn.Identity`` (true linear model).
    prefiltering : nn.Module, optional
        Optional spectrogram prefilter (``AdapTrans``, ``ICAdaptation``,
        or any ``nn.Module`` exposing ``out_channels``). ``None``
        (default) gives ``nn.Identity`` and ``C_in = 1``.
    kernel : nn.Module, optional
        Pluggable STRF kernel for the readout. ``None`` (default) gives
        a vanilla ``nn.Conv2d``; pass ``ParametricSTRF(...)`` for DCLS,
        or a separable ``nn.Sequential`` for a rank-1 factorization.
        See ``deepSTRF.models.layers`` for the kernel module catalogue.

    References
    ----------
    The L model is a folklore baseline; canonical formulations appear in:

    Theunissen, Sen & Doupe (2000). "Spectral-Temporal Receptive Fields of
    Nonlinear Auditory Neurons Obtained Using Natural Sounds."
    J. Neurosci. 20(6): 2315–2331.
    https://doi.org/10.1523/JNEUROSCI.20-06-02315.2000

    Sahani & Linden (2003). "How Linear are Auditory Cortical Responses?"
    NIPS.

    Notes
    -----
    The per-neuron BatchNorm inside the readout absorbs into the kernel
    at inference (its running stats are frozen per-channel scalars), so
    the model remains a strict linear-affine map of the input at eval
    time and the learned STRF kernel is directly interpretable up to a
    per-neuron affine rescaling.
    """
    def __init__(self, n_frequency_bands: int = 34, temporal_window_size: int = 9,
                 out_neurons: int = 1,
                 output_activation: nn.Module = None,
                 prefiltering: nn.Module = None,
                 kernel: nn.Module = None):
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=temporal_window_size,
            out_neurons=out_neurons,
            prefiltering=prefiltering,
        )
        # core: identity. All learnable parameters live in the readout —
        # the per-neuron BatchNorm1d inside STRFReadout handles normalisation
        # at the model's output (one BN per cell, no params shared across N).
        self.core = nn.Identity()
        # readout: pluggable STRF kernel + per-neuron BN + output activation
        self.readout = STRFReadout(
            F=self.F, T=self.T, C_in=self.C_in, out_neurons=self.O,
            kernel=kernel,
            activation=output_activation if output_activation is not None else nn.Identity(),
        )
        # forward is inherited from NeuralModel — wav2spec → prefiltering → core → readout

    def STRF_weight(self, polarity: str = 'ON'):
        """
        Return the readout's STRF kernel as a ``(N, F, T)`` tensor.

        For models prefiltered with ``AdapTrans`` (``C_in == 2``),
        ``polarity`` selects the ON or OFF channel of the kernel. For
        single-channel inputs the parameter is ignored.
        """
        full = self.readout.STRF_weight()                           # (N, C_in, F, T)
        if isinstance(self.prefiltering, AdapTrans):
            if polarity in ('ON', 'On', 'on', 0):
                return full[:, 0]
            if polarity in ('OFF', 'Off', 'off', 1):
                return full[:, 1]
            raise ValueError(
                f"polarity must be 'ON' or 'OFF' for an AdapTrans-prefiltered "
                f"Linear model — got {polarity!r}"
            )
        return full[:, 0]


class LinearNonlinear(Linear):
    """
    Linear-Nonlinear (LN) STRF model — the Linear model followed by a
    pointwise output nonlinearity.

    Inherits everything from ``Linear`` and only changes the default
    output activation from ``nn.Identity`` to a per-neuron
    :class:`ParametricSoftplus`. Pass any ``nn.Module`` to
    ``output_activation`` to override.

    Parameters
    ----------
    output_activation : nn.Module, default ParametricSoftplus(out_neurons)
        Pointwise nonlinearity applied at the readout output. The default
        is unbounded above and non-negative — natural for spike-count
        regression on smoothed PSTHs that exceed 1. See
        ``deepSTRF.models.activations`` for other parametric variants
        (``ParametricSigmoid``, ``ParametricDoubleExponential``).

    See Also
    --------
    Linear : Same architecture without the output nonlinearity.
    """
    def __init__(self, n_frequency_bands: int = 34, temporal_window_size: int = 9,
                 out_neurons: int = 1,
                 output_activation: nn.Module = None,
                 prefiltering: nn.Module = None,
                 kernel: nn.Module = None):
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=temporal_window_size,
            out_neurons=out_neurons,
            output_activation=(output_activation if output_activation is not None
                               else ParametricSoftplus(out_neurons)),
            prefiltering=prefiltering,
            kernel=kernel,
        )



class NetworkReceptiveField(AudioEncodingModel):
    """
    Network Receptive Field (NRF) model — a two-layer feedforward STRF
    network.

    Architecture: a STRF kernel projects the input spectrogram into a
    hidden layer of ``H`` units; a 1×1 conv reads out the ``N`` output
    neurons from the hidden activations. With L1 regularization the
    paper finds typically 1–7 effective hidden units per neuron.

    Parameters
    ----------
    n_frequency_bands : int, default 34
        Number of input frequency bands ``F``.
    temporal_window_size : int, default 9
        STRF temporal extent ``T``.
    n_hidden : int, default 20
        Hidden layer width ``H``.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    output_activation : nn.Module, optional
        Pointwise nonlinearity at the readout output. Default
        ``nn.Sigmoid``.
    prefiltering : nn.Module, optional
        Optional spectrogram prefilter (``AdapTrans``, ``ICAdaptation``,
        any module exposing ``out_channels``). ``None`` (default) gives
        ``nn.Identity`` and ``C_in = 1``.
    kernel : nn.Module, optional
        Pluggable hidden-layer STRF kernel. ``None`` (default) gives a
        vanilla ``nn.Conv2d``; pass ``ParametricSTRF(...)`` for DCLS.

    References
    ----------
    Harper, Schoppe, Willmore, Cui, Schnupp & King (2016).
    "Network Receptive Field Modeling Reveals Extensive Integration and
    Multi-feature Selectivity in Auditory Cortical Neurons."
    PLOS Comp. Biol. 12(11): e1005113.
    https://doi.org/10.1371/journal.pcbi.1005113

    Notes
    -----
    Differences from the original paper:

    - We add a causal LayerNorm over input frequencies and over the
      hidden channel axis. The original assumes preprocessing-time
      input normalization and uses no internal norm.
    - The hidden activation is ``nn.Tanh`` (paper-faithful: scaled
      tanh with ρ₁ ≈ 1.7159, ρ₂ = 2/3 — we use the unscaled standard
      tanh, equivalent up to a learned rescaling absorbed into the
      readout).
    - Causal left-padding extends the model to arbitrary input lengths;
      the paper uses fixed-window slicing.
    - The hidden STRF kernel can be parameterized (DCLS); the paper
      uses a vanilla full kernel.
    """
    def __init__(self, n_frequency_bands: int = 34, temporal_window_size: int = 9,
                 n_hidden: int = 20,
                 out_neurons: int = 1,
                 output_activation: nn.Module = None,
                 prefiltering: nn.Module = None,
                 kernel: nn.Module = None):
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=temporal_window_size,
            out_neurons=out_neurons,
            prefiltering=prefiltering,
        )
        self.H = n_hidden

        # core: hidden STRF projection → channel norm → tanh.
        # The hidden STRF projection emits (B, H, 1, T); LinearReadout downstream
        # squeezes the singleton spatial axis automatically.
        self.core = nn.Sequential(
            layers.CausalSTRFConv(self.F, self.T, self.C_in, self.H, kernel=kernel),
            layers.CausalLayerNorm(self.H, dim=1),
            nn.Tanh(),
        )

        # readout: per-neuron 1×1 projection from H hidden units.
        self.readout = LinearReadout(
            in_features=self.H, out_neurons=self.O,
            activation=(output_activation if output_activation is not None
                        else ParametricSoftplus(self.O)),
        )
        # forward inherited from NeuralModel — wav2spec → prefiltering → core → readout

    def STRFs(self, hidden_idx: int = 0, polarity: str = 'ON'):
        """
        Return the hidden-layer STRF kernel for one hidden unit as ``(F, T)``.

        Parameters
        ----------
        hidden_idx : int, default 0
            Which of the ``H`` hidden units to return the STRF for.
        polarity : {'ON', 'OFF'}, default 'ON'
            Only relevant when the prefilter has ``C_in == 2`` (e.g.
            AdapTrans). Selects the ON or OFF channel of the kernel.
        """
        # core[0] is the CausalSTRFConv whose .STRF_weight() returns (H, C_in, F, T)
        full = self.core[0].STRF_weight()
        if isinstance(self.prefiltering, AdapTrans):
            if polarity in ('ON', 'On', 'on', 0):
                return full[hidden_idx, 0]
            if polarity in ('OFF', 'Off', 'off', 1):
                return full[hidden_idx, 1]
            raise ValueError(
                f"polarity must be 'ON' or 'OFF' for an AdapTrans-prefiltered NRF — got {polarity!r}"
            )
        return full[hidden_idx, 0]


class DNet(AudioEncodingModel):
    """
    Dynamic Network (DNet) — an NRF whose hidden and output units are
    stateful with learnable temporal decay.

    Architecture: STRF projection → channel-norm → sigmoid → learnable
    exponential decay (one time constant per hidden unit) → 1×1 readout
    → output activation. The exponential decay is causal: each unit's
    output at time ``t`` is a convolution of its instantaneous input
    with a learned one-sided exponential kernel.

    Parameters
    ----------
    n_frequency_bands : int, default 34
        Number of input frequency bands ``F``.
    temporal_window_size : int, default 9
        STRF temporal extent ``T``.
    n_hidden : int, default 20
        Hidden layer width ``H``.
    init_tau : float, default 2.0
        Initial time constant (in frames) for the hidden-unit
        exponential decay.
    decay_input : bool, default True
        If True, the exponential decay also weights its instantaneous
        input by ``1/(1+d²)`` (paper convention); if False, the
        instantaneous input passes through unscaled.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    output_activation : nn.Module, optional
        Pointwise nonlinearity at the readout output. Default
        ``nn.Identity`` (paper-faithful linear readout).
    prefiltering : nn.Module, optional
        Optional spectrogram prefilter.
    kernel : nn.Module, optional
        Pluggable hidden-layer STRF kernel.

    References
    ----------
    Rahman, Willmore, King & Harper (2019).
    "A dynamic network model of temporal receptive fields in primary
    auditory cortex." PLOS Comp. Biol. 15(5): e1006618.
    https://doi.org/10.1371/journal.pcbi.1006618

    Notes
    -----
    Differences from the original paper:

    - Causal LayerNorm replaces the missing internal normalization
      (paper assumes preprocessing-time input normalization).
    - Causal left-padding extends the model to arbitrary input lengths;
      the paper uses fixed-window slicing.
    - The hidden STRF kernel can be parameterized (DCLS); the paper
      uses a vanilla full kernel.
    """
    def __init__(self, n_frequency_bands: int = 34, temporal_window_size: int = 9,
                 n_hidden: int = 20, init_tau: float = 2.0, decay_input: bool = True,
                 out_neurons: int = 1,
                 output_activation: nn.Module = None,
                 prefiltering: nn.Module = None,
                 kernel: nn.Module = None):
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=temporal_window_size,
            out_neurons=out_neurons,
            prefiltering=prefiltering,
        )
        self.H = n_hidden
        decay_kernel = round(init_tau * 7)

        # core: hidden STRF projection → channel norm → sigmoid
        #       → per-hidden-unit causal exponential decay
        self.core = nn.Sequential(
            layers.CausalSTRFConv(self.F, self.T, self.C_in, self.H, kernel=kernel),
            layers.CausalLayerNorm(self.H, dim=1),
            nn.Sigmoid(),
            layers.LearnableExponentialDecay(self.H, kernel_size=decay_kernel,
                                             init_tau=init_tau, decay_input=decay_input),
        )

        # readout: per-neuron 1×1 projection from H decayed hidden units;
        # the hidden-side decay already provides temporal smoothing.
        self.readout = LinearReadout(
            in_features=self.H, out_neurons=self.O,
            activation=output_activation if output_activation is not None else nn.Identity(),
        )
        # forward inherited from NeuralModel — wav2spec → prefiltering → core → readout

    def STRFs(self, hidden_idx: int = 0, polarity: str = 'ON'):
        """
        Return the hidden-layer STRF kernel for one hidden unit as ``(F, T)``.

        Parameters
        ----------
        hidden_idx : int, default 0
            Which of the ``H`` hidden units to inspect.
        polarity : {'ON', 'OFF'}, default 'ON'
            Only relevant for AdapTrans-prefiltered models (``C_in == 2``).
        """
        # core[0] is the CausalSTRFConv whose .STRF_weight() returns (H, C_in, F, T)
        full = self.core[0].STRF_weight()
        if isinstance(self.prefiltering, AdapTrans):
            if polarity in ('ON', 'On', 'on', 0):
                return full[hidden_idx, 0]
            if polarity in ('OFF', 'Off', 'off', 1):
                return full[hidden_idx, 1]
            raise ValueError(
                f"polarity must be 'ON' or 'OFF' for an AdapTrans-prefiltered DNet — got {polarity!r}"
            )
        return full[hidden_idx, 0]


class ConvNet2D(AudioEncodingModel):
    """
    Convolutional STRF model with three sequential 2D convs and a 2-layer
    fully-connected readout — adapted from the '2D-CNN' of Pennington
    & David (2023).

    Architecture: three Conv2d → CausalLayerNorm → LeakyReLU blocks
    extract a stack of feature maps; the per-time-step features are
    flattened over the (channel × downsampled-frequency) axes and a
    2-layer FC reads out ``N`` output neurons.

    Parameters
    ----------
    n_frequency_bands : int, default 34
        Number of input frequency bands ``F``.
    kernel_size : tuple of int, default (3, 9)
        Conv2d kernel ``(K_F, K_T)`` shared across the three conv blocks.
    c_hidden : int, default 10
        Number of channels in each conv block.
    n_hidden : int, default 20
        Width of the FC hidden layer.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    output_activation : nn.Module, default ``ParametricSoftplus(out_neurons)``
        Pointwise nonlinearity at the output. The default is unbounded
        above and non-negative — natural for spike-count regression.
    prefiltering : dict or None
        Optional spectrogram prefilter spec.

    References
    ----------
    Pennington & David (2023). "A convolutional neural network provides
    a generalizable model of natural sound coding by neural populations
    in auditory cortex." PLOS Comp. Biol. 19(5): e1011110.
    https://doi.org/10.1371/journal.pcbi.1011110

    Notes
    -----
    Differences from the original paper:

    - Causal LayerNorm replaces the missing internal normalization
      (paper uses none).
    - Hidden activation is ``LeakyReLU(0.1)`` rather than ReLU. Empirical
      preference, very small architectural difference.
    - 2D convs over ``(F, T)`` rather than 1D convs over ``T`` (the
      paper applies 1D convolutions with implicit spectral pooling).
    - Frequency downsampling is implicit via valid-padding shrinkage:
      three convs each shrink ``F`` by ``K_F - 1``, giving
      ``F_down = F - 3*(K_F - 1)``.
    - Causal left-padding extends the model to arbitrary input lengths;
      the paper also uses explicit causal padding.
    - The output activation is configurable; the paper uses a 4-parameter
      double-exponential — see ``deepSTRF.models.activations.ParametricDoubleExponential``.
    """
    def __init__(self, n_frequency_bands: int = 34, kernel_size: tuple = (3, 9),
                 c_hidden: int = 10, n_hidden: int = 20,
                 out_neurons: int = 1,
                 output_activation: nn.Module = None,
                 prefiltering: nn.Module = None):
        temporal_window_size = 3 * (kernel_size[1] - 1)
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=temporal_window_size,
            out_neurons=out_neurons,
            prefiltering=prefiltering,
        )
        self.K = kernel_size
        self.C = c_hidden
        self.H = n_hidden

        # core: causal left-pad → 3× (Conv2d → LN → LeakyReLU)
        #       → flatten (C, F_down) into a single feature axis.
        # Three convs each shrink time by K_T-1 (and frequency by K_F-1); the
        # explicit left-pad of 3*(K_T-1) zeros restores the time length.
        F_down = self.F - 3 * (self.K[0] - 1)  # frequency dim after 3 convs
        self.core = nn.Sequential(
            nn.ZeroPad2d((3 * (self.K[1] - 1), 0, 0, 0)),
            nn.Conv2d(self.C_in, self.C, kernel_size=self.K, stride=1),
            layers.CausalLayerNorm(self.C, dim=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.C, self.C, kernel_size=self.K, stride=1),
            layers.CausalLayerNorm(self.C, dim=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.C, self.C, kernel_size=self.K, stride=1),
            layers.CausalLayerNorm(self.C, dim=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(start_dim=1, end_dim=2),  # (B, C, F_down, T) → (B, C*F_down, T)
        )

        # readout: per-timestep MLP (in → hidden → N) with output activation.
        self.readout = LinearReadout(
            in_features=self.C * F_down,
            out_neurons=self.O,
            hidden=self.H,
            activation=(output_activation if output_activation is not None
                        else ParametricSoftplus(self.O)),
        )
        # forward inherited from NeuralModel


class Transformer(AudioEncodingModel):
    """
    Attention-based STRF model — a Transformer encoder runs causal
    self-attention over a per-timestep token sequence extracted from the
    spectrogram.

    Architecture::

        input              (B, 1, F, L)
          ↓ prefilter      (B, C_in, F, L)
          ↓ time pad       (B, C_in, F, L + K_T - 1)         left-only causal
          ↓ patchify       (B, embedding_dim, F_p, L)        Conv2d, stride=(K_F, 1)
          ↓ flatten/permute (B, L, embedding_dim * F_p)      one token per timestep
          ↓ + sinusoidal positional encoding
          ↓ TransformerEncoder with causal (+optional window) mask
          ↓ readout        (B, N, 1, L)

    The patchifier is a strided ``nn.Conv2d`` with kernel ``(K_F, K_T)``
    and stride ``(K_F, 1)``. Frequency patches are non-overlapping
    (``F_p = F // K_F``); time stride is 1 so there is one token per
    timestep, and ``K_T > 1`` lets each token aggregate
    ``time_patch_size`` recent frames. A ``(K_T - 1)``-zero left pad
    along time keeps the patchifier strictly causal.

    The attention mask is constructed per forward pass at the actual
    sequence length, so the model generalizes to any input length ``L``.
    Setting ``context_window`` to a positive int restricts attention to
    the most recent ``context_window`` past frames (band-causal),
    recovering a Sahani-style fixed STRF context window when wanted —
    the model can be evaluated with or without the bound at inference
    without retraining.

    Parameters
    ----------
    n_frequency_bands : int, default 34
        Number of input frequency bands ``F``.
    freq_patch_size : int, optional
        Frequency-axis patch size for the patchifier. ``None`` (default)
        uses ``F`` itself — one token spans the full frequency axis at
        each timestep. Must divide ``F``.
    time_patch_size : int, default 1
        Temporal extent of each patch in frames. ``1`` gives one token =
        one timestep slice; larger values let each token aggregate
        across ``time_patch_size`` recent frames via the patchifier.
    context_window : int, optional
        If set, restrict attention to the most recent ``context_window``
        past frames (still causal — band-causal mask). ``None`` (default)
        gives unlimited causal context.
    embedding_dim : int, default 48
        Per-patch embedding dimension after the patchifier.
    n_heads : int, default 1
        Number of attention heads. Must divide ``embedding_dim * F_p``.
    n_layers : int, default 1
        Number of TransformerEncoderLayer blocks.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    output_activation : nn.Module, optional
        Pointwise nonlinearity at the readout. Default ``nn.Identity``.
    prefiltering : nn.Module, optional
        Optional spectrogram prefilter.

    References
    ----------
    Rançon, Bornschein, King, Schnupp, Willmore (2025). "Temporal
    recurrence as a general mechanism to explain neural responses in
    the auditory system." Comm. Bio. (preprint on BioRxiv).

    Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.

    Notes
    -----
    Sinusoidal positional encoding (Vaswani 2017) is used by default; it
    generalizes to arbitrary sequence lengths at inference. RoPE
    (Rotary Position Embedding, Su et al. 2021) is a planned alternative;
    it is omitted here because it requires a custom TransformerEncoderLayer
    (PyTorch's stock module hides Q and K).
    """

    def __init__(self, n_frequency_bands: int = 34,
                 freq_patch_size: int = None, time_patch_size: int = 1,
                 context_window: int = None,
                 embedding_dim: int = 48,
                 n_heads: int = 1, n_layers: int = 1,
                 out_neurons: int = 1,
                 output_activation: nn.Module = None,
                 prefiltering: nn.Module = None):
        # `temporal_window_size` on the base is used by STRF_gradmap to size
        # a null-stim probe; the receptive-field "window" in this model is
        # context_window if set else a sensible default.
        gradmap_T = context_window if context_window is not None else max(time_patch_size, 9)
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=gradmap_T,
            out_neurons=out_neurons,
            prefiltering=prefiltering,
        )

        # default freq_patch_size: cover the whole frequency axis (one token = one frame slice)
        if freq_patch_size is None:
            freq_patch_size = self.F
        if self.F % freq_patch_size != 0:
            raise ValueError(
                f"freq_patch_size={freq_patch_size} must divide F={self.F}"
            )
        self.K_F = freq_patch_size
        self.K_T = time_patch_size
        self.context_window = context_window
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.F_p = self.F // self.K_F
        self.token_dim = self.embedding_dim * self.F_p

        if self.token_dim % 2 != 0:
            raise ValueError(
                f"token_dim = embedding_dim * F_p = {self.token_dim} must be even "
                f"for sinusoidal positional encoding "
                f"(got embedding_dim={embedding_dim}, F_p={self.F_p})"
            )
        if self.token_dim % self.n_heads != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by n_heads={self.n_heads}"
            )

        # causal time-pad followed by frequency-strided / time-stride-1 patchifier
        self.time_pad = nn.ZeroPad2d((self.K_T - 1, 0, 0, 0))
        self.patchify = nn.Conv2d(
            self.C_in, self.embedding_dim,
            kernel_size=(self.K_F, self.K_T),
            stride=(self.K_F, 1),
        )

        # sinusoidal positional encoding — added in forward, length L is dynamic
        self.pos_encoding = layers.SinusoidalPositionalEncoding(self.token_dim)

        self.tsfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.token_dim, nhead=self.n_heads,
                dim_feedforward=4 * self.token_dim, dropout=0.1,
                batch_first=True,
            ),
            num_layers=self.n_layers,
        )

        # per-timestep readout: token_dim → N
        self.readout = LinearReadout(
            in_features=self.token_dim, out_neurons=self.O,
            activation=output_activation if output_activation is not None else nn.Identity(),
        )

    def forward(self, x):
        """
        Causal-attention forward.

        Overrides the base template because the attention mask must be
        rebuilt at the actual sequence length L of each input.
        """
        # x: (B, 1, F, L)
        B, _, _, L = x.shape
        y = self.prefiltering(x)                # (B, C_in, F, L)
        y = self.time_pad(y)                    # (B, C_in, F, L + K_T - 1)
        y = self.patchify(y)                    # (B, embedding_dim, F_p, L)
        y = y.flatten(start_dim=1, end_dim=2)   # (B, token_dim, L)
        y = y.transpose(1, 2)                   # (B, L, token_dim)
        y = self.pos_encoding(y)                # (B, L, token_dim) — sinusoidal
        # causal (and optionally windowed) attention mask, built at this L
        mask = layers.build_causal_window_mask(L, self.context_window, device=y.device)
        y = self.tsfm(y, mask=mask)             # (B, L, token_dim)
        y = y.transpose(1, 2)                   # (B, token_dim, L)
        return self.readout(y)                  # (B, N, 1, L)


class StateNet(AudioEncodingModel):
    """
    Fully stateful STRF model — relies entirely on temporal recurrence to
    extract information from stimulus sequences, with no explicit STRF
    delay window.

    Architecture: a stateless per-timestep spectral encoder maps each
    spectrogram column ``(C_in, F)`` to a hidden representation
    ``(C, F_down)``. The flattened hidden representation is fed
    timestep-by-timestep to a recurrent (or state-space) model that
    accumulates context implicitly through its hidden state. A linear
    readout projects the recurrent hidden state to ``N`` output neurons.

    Causality is inherent to the recurrent backbone (RNN/GRU/LSTM/LMU/
    Mamba/S4). The spectral encoder operates on a single timestep at a
    time so it does not couple frames temporally.

    Parameters
    ----------
    n_frequency_bands : int, default 34
        Number of input frequency bands ``F``.
    temporal_window_size : int, default 1
        Unused by StateNet (kept for ``AudioEncodingModel`` API
        compatibility); recurrence handles temporal context.
    kernel_size : int, default 7
        Frequency kernel size for the spectral encoder.
    stride : int, default 3
        Frequency stride for the spectral encoder.
    hidden_channels : int, default 7
        Channel count of the spectral encoder ``C``.
    connectivity : {'LC', 'FC', 'CONV'}, default 'LC'
        Spectral encoder connectivity. ``'LC'``: locally-connected 1D
        layer (frequency-position-specific weights). ``'FC'``: dense
        linear projection with reshape to ``(C, F_down)``. ``'CONV'``:
        weight-shared 1D convolution.
    rnn_type : {'GRU', 'LSTM', 'RNN', 'vanilla', 'LMU', 'Mamba', 'S4'}, default 'GRU'
        Recurrent / state-space backbone.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    output_activation : nn.Module, default ``ParametricSoftplus(out_neurons)``
        Pointwise nonlinearity at the output. The default is unbounded
        above and non-negative — natural for spike-count regression.
    prefiltering : dict or None
        Optional spectrogram prefilter spec.

    References
    ----------
    Rançon, Bornschein, King, Schnupp, Willmore (2025).
    "Temporal recurrence as a general mechanism to explain neural
    responses in the auditory system." Comm. Bio. (preprint on BioRxiv).

    Notes
    -----
    - The spectral encoder uses a CausalLayerNorm over the channel
      axis (``C``); the original implementation used BatchNorm1d
      which pools statistics over the (T*B, F_down) axis, making it
      non-causal.
    - The S4 backbone is imported lazily — its module emits CUDA-extension
      warnings on import that other backends would not see.
    """
    def __init__(self, n_frequency_bands: int = 34, temporal_window_size: int = 1,
                 kernel_size: int = 7, stride: int = 3, hidden_channels: int = 7,
                 connectivity: str = 'LC', rnn_type: str = 'GRU',
                 out_neurons: int = 1,
                 output_activation: nn.Module = None,
                 prefiltering: nn.Module = None):
        super().__init__(
            n_frequency_bands=n_frequency_bands,
            temporal_window_size=temporal_window_size,
            out_neurons=out_neurons,
            prefiltering=prefiltering,
        )
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels
        self.rnn_type = rnn_type

        # stateless per-timestep spectral encoder: (B, C_in, F) -> (B, C, F_down).
        # Three connectivity options share the same input/output shape contract.
        if connectivity == 'LC':
            self.encoder_layers = nn.Sequential(
                layers.LocallyConnected1d(input_size=self.F, in_channels=self.C_in,
                                          out_channels=self.C, kernel_size=self.K, stride=self.S),
                layers.CausalLayerNorm(self.C, dim=1),
                nn.Sigmoid(),
            )
        elif connectivity == 'FC':
            F_down = int((self.F - kernel_size) / self.S + 1)
            self.encoder_layers = nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),                       # (B, C_in, F) -> (B, C_in*F)
                nn.Linear(self.C_in * self.F, self.C * F_down),
                nn.Unflatten(dim=-1, unflattened_size=(self.C, F_down)),    # (B, C, F_down)
                layers.CausalLayerNorm(self.C, dim=1),
                nn.Sigmoid(),
            )
        elif connectivity == 'CONV':
            self.encoder_layers = nn.Sequential(
                nn.Conv1d(self.C_in, self.C, kernel_size=self.K, stride=self.S),
                layers.CausalLayerNorm(self.C, dim=1),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError(
                f"connectivity must be 'LC', 'FC', or 'CONV', got {connectivity!r}"
            )

        # F_down after the spectral encoder: used to size the RNN's input.
        self.L = (n_frequency_bands - self.K) // self.S + 1
        self.H = self.L * self.C

        # recurrent / state-space backbone
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif self.rnn_type in ('vanilla', 'RNN'):
            self.rnn = nn.RNN(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif self.rnn_type == 'LMU':
            self.rnn = LMU(input_size=self.H, hidden_size=self.H, memory_size=128,
                           theta=99, learn_a=False, learn_b=False)
        elif self.rnn_type == 'Mamba':
            self.rnn = MambaBlock(MambaConfig(d_model=self.H, n_layers=1))
        elif self.rnn_type == 'S4':
            from deepSTRF.models.dependencies.s4 import S4Block
            self.rnn = S4Block(d_model=self.H, transposed=False)
        else:
            raise NotImplementedError(
                f"unknown rnn_type {rnn_type!r}: choose 'GRU', 'LSTM', 'RNN', 'vanilla', "
                f"'LMU', 'Mamba', or 'S4'"
            )

        # per-neuron readout from the recurrent hidden state.
        self.readout = LinearReadout(
            in_features=self.H, out_neurons=self.O,
            activation=(output_activation if output_activation is not None
                        else ParametricSoftplus(self.O)),
        )

    def forward(self, x):
        """
        Per-timestep spectral encoder feeding a recurrent backbone.

        Overrides the base template because the encoder runs in a flattened
        (T*B, C_in, F) batch (so every timestep is independent in the
        spectral pass) and the RNN expects a batch-first ``(B, T, H)``
        layout — these reshapes don't decompose into the canonical core /
        readout slots.
        """
        # x: (B, 1, F, T)
        y = self.prefiltering(x)                                # (B, C_in, F, T)
        y = y.permute(3, 0, 1, 2)                               # (T, B, C_in, F)
        T_, B = y.shape[:2]
        y = y.flatten(0, 1)                                     # (T*B, C_in, F)
        y = self.encoder_layers(y)                              # (T*B, C, F_down)
        y = y.view(T_, B, self.C, self.L)                       # (T, B, C, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C*F_down=H)

        # RNN/SSM backbone — most modules return (output, hidden); Mamba is
        # the exception (returns a single tensor).
        if self.rnn_type == 'Mamba':
            y = self.rnn(y)
        else:
            y, _ = self.rnn(y)                                  # (B, T, H)

        y = y.transpose(-2, -1)                                 # (B, H, T) — readout convention
        return self.readout(y)                                  # (B, N, 1, T)
