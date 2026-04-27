import torch
import torch.nn as nn
import torch.nn.functional

from .audio_model import AudioEncodingModel

import deepSTRF.models.layers as layers
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
    holds the kernel of shape ``(N, C_in, F, T)`` and applies it causally
    via left-padding. The model's ``core`` is a single
    ``CausalLayerNorm`` over the frequency axis — see notes below for the
    rationale.

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
    A causal LayerNorm over the frequency axis is applied as the
    model's core (per-timestep input normalization). Empirically, this
    stabilizes training of small unparameterized STRF models
    substantially. Unlike BatchNorm, LayerNorm cannot be absorbed into
    the readout kernel at inference (it computes fresh statistics per
    sample), so the model is technically nonlinear in the strict sense
    — but the per-time-step normalization is mild and the learned
    kernel still serves as a directly interpretable STRF up to a
    per-sample input scaling.
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
        # core: causal per-timestep frequency normalization
        self.core = layers.CausalLayerNorm(self.F, dim=-2)
        # readout: pluggable STRF kernel + output activation
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
    output activation from ``nn.Identity`` to ``nn.Sigmoid``. Pass any
    ``nn.Module`` to ``output_activation`` to override.

    Parameters
    ----------
    output_activation : nn.Module, default nn.Sigmoid()
        Pointwise nonlinearity applied at the readout output. See
        ``deepSTRF.models.activations`` for parametric variants
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
            output_activation=output_activation if output_activation is not None else nn.Sigmoid(),
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

        # core: input freq norm → hidden STRF projection → channel norm → tanh
        # The hidden STRF projection emits (B, H, 1, T); LinearReadout downstream
        # squeezes the singleton spatial axis automatically.
        self.core = nn.Sequential(
            layers.CausalLayerNorm(self.F, dim=-2),
            layers.CausalSTRFConv(self.F, self.T, self.C_in, self.H, kernel=kernel),
            layers.CausalLayerNorm(self.H, dim=1),
            nn.Tanh(),
        )

        # readout: per-neuron 1×1 projection from H hidden units.
        self.readout = LinearReadout(
            in_features=self.H, out_neurons=self.O,
            activation=output_activation if output_activation is not None else nn.Sigmoid(),
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
        # core[1] is the CausalSTRFConv whose .STRF_weight() returns (H, C_in, F, T)
        full = self.core[1].STRF_weight()
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

        # core: input freq norm → hidden STRF projection → channel norm
        #       → sigmoid → per-hidden-unit causal exponential decay
        self.core = nn.Sequential(
            layers.CausalLayerNorm(self.F, dim=-2),
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
        # core[1] is the CausalSTRFConv whose .STRF_weight() returns (H, C_in, F, T)
        full = self.core[1].STRF_weight()
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
    output_activation : nn.Module, default nn.Sigmoid()
        Pointwise nonlinearity at the output.
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

        # core: input freq norm → causal left-pad → 3× (Conv2d → LN → LeakyReLU)
        #       → flatten (C, F_down) into a single feature axis.
        # Three convs each shrink time by K_T-1 (and frequency by K_F-1); the
        # explicit left-pad of 3*(K_T-1) zeros restores the time length.
        F_down = self.F - 3 * (self.K[0] - 1)  # frequency dim after 3 convs
        self.core = nn.Sequential(
            layers.CausalLayerNorm(self.F, dim=-2),
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
            activation=output_activation if output_activation is not None else nn.Sigmoid(),
        )
        # forward inherited from NeuralModel


class Transformer(AudioEncodingModel):
    """
    Attention-based STRF model — at every output timestep, a Transformer
    encoder attends over patches of the recent ``T``-frame spectrogram
    context.

    Architecture: a left-padded ``(F, T)`` context window slides over the
    spectrogram (one window per output timestep); the window is
    patchified into ``n_patches`` learnable embeddings via a strided
    Conv2d; a TransformerEncoder runs self-attention over the patches;
    a global mean over patches feeds a linear readout to ``N`` neurons.

    Causality is enforced architecturally by the ``unfold`` step: each
    output frame at time ``t`` sees only the window ``[t-T+1, t]``.
    The unfold turns each output frame into an *independent* batched
    item; the Conv2d patchifier and TransformerEncoder operate within
    each window and never across them. Output frame ``t`` is therefore
    a pure function of input ``[t-T+1, t]``. (The internal dropout
    layer is stochastic in train mode but does not introduce any
    cross-time dependence — controlled-seed forward passes confirm
    bitwise causality.)

    Parameters
    ----------
    n_frequency_bands : int, default 34
        Number of input frequency bands ``F``.
    temporal_window_size : int, default 1
        Context window length ``T`` (frames). Each output frame attends
        within this many past frames.
    token_size : tuple of int, default (34, 1)
        Patch dimensions ``(K_F, K_T)`` for the Conv2d patchifier.
    embedding_dim : int, default 48
        Per-patch embedding dimension before attention.
    n_heads : int, default 1
        Number of attention heads.
    n_layers : int, default 1
        Number of TransformerEncoderLayer blocks.
    out_neurons : int, default 1
        Number of output neurons ``N``.
    output_activation : nn.Module, default nn.Identity()
        Pointwise nonlinearity at the output. The paper uses a 4-parameter
        double-exponential — available as ``ParametricDoubleExponential``.
    prefiltering : dict or None
        Optional spectrogram prefilter spec.

    References
    ----------
    Rançon, Bornschein, King, Schnupp, Willmore (2025).
    "Temporal recurrence as a general mechanism to explain neural
    responses in the auditory system." Comm. Bio. (preprint on BioRxiv).

    Notes
    -----
    The current implementation explicitly fixes context length to ``T``
    via ``nn.Unfold``. A future refactor will replace this with an
    internal causal attention mask, freeing the architecture from a
    fixed context length and letting it generalize to arbitrary input
    durations (see ``TODO.md``).
    """

    def __init__(self, n_frequency_bands=34, temporal_window_size=1, token_size=(34, 1), embedding_dim=48, n_heads=1,
                 n_layers=1, out_neurons: int = 1, output_activation: nn.Module = None, prefiltering: nn.Module = None):
        super(Transformer, self).__init__(n_frequency_bands, temporal_window_size, out_neurons=out_neurons, prefiltering=prefiltering)
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        # patch dimensions and strides
        self.K_f, self.K_t = token_size

        # padding left only (causal inference)
        self.pad = nn.ZeroPad2d(((self.T - 1), 0, 0, 0))

        # im2col --> create context windows for attention/transformer layers
        self.unfold_context = nn.Unfold(kernel_size=(self.F, self.T), stride=(1, 1), padding=(0, 0))

        # Patchify with learnable embdeddings
        self.embedding_dim = embedding_dim
        self.conv_patches = nn.Conv2d(self.C_in, self.embedding_dim, kernel_size=(self.K_f, self.K_t), stride=(self.K_f, self.K_t))

        # determine sizes to define the readout layer
        prospective_input = torch.rand(1, self.C_in, self.F, self.T)
        # prospective_patches = self.unfold_patches(prospective_input)
        prospective_patches = self.conv_patches(prospective_input).flatten(-2, -1)
        _, patch_dim, n_patches = prospective_patches.shape
        self.H = patch_dim

        self.n_heads = n_heads
        self.n_layers = n_layers
        self.positional_encoding = torch.nn.Parameter(torch.rand(1, n_patches, patch_dim))
        self.tsfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=patch_dim, nhead=self.n_heads, dim_feedforward=32, dropout=0.1,
                                       batch_first=True),
            num_layers=self.n_layers,
        )

        self.readout = nn.Linear(patch_dim, self.O)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        B, C, F, L = x.shape
        y = self.prefiltering(x)                                    # (B, C=1|2, F, L)
        y = self.unfold_context(self.pad(y))  # (B, C*F*T, L)
        y = y.permute(0, 2, 1).flatten(0, 1)  # (B*L, C*F*T)

        y = y.reshape(-1, self.C_in, self.F, self.T)    # (B*L, C, F, T)
        y = self.conv_patches(y)                        # (B*L, C+, F-, T-)
        y = y.flatten(start_dim=-2, end_dim=-1)         # (B*L, patch_dim = C+, n_patches = F- * T-)
        y = y.permute(0, 2, 1)                          # (B*L, n_patches, patch_dim)

        y = self.tsfm(y + self.positional_encoding)  # (B*L, n_patches, patch_dim)

        # global average pooling
        y = y.mean(1, keepdim=True)  # (B*L, 1, patch_dim)

        y = y.flatten(start_dim=1, end_dim=2)   # (B*L, n_patches * patch_dim)
        y = y.unflatten(dim=0, sizes=(B, L))    # B, L, n_patches * patch_dim)
        y = self.readout(y)                     # (B, L, N)
        y = self.output_activation(y)           # (B, L, N)
        y = y.permute(0, 2, 1)                  # (B, N, L)  TODO: --> (B, N, 1, L)
        return y


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
    output_activation : nn.Module, default nn.Sigmoid()
        Pointwise nonlinearity at the output.
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
    def __init__(self, n_frequency_bands=34, temporal_window_size=1, kernel_size: int = 7, stride: int = 3,
                 hidden_channels: int = 7, connectivity: str = 'LC', rnn_type: str = 'GRU', out_neurons: int = 1,
                 output_activation: nn.Module = None, prefiltering: nn.Module = None):
        super(StateNet, self).__init__(n_frequency_bands, temporal_window_size, out_neurons=out_neurons, prefiltering=prefiltering)
        self.output_activation = output_activation if output_activation is not None else nn.Sigmoid()

        # general
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels
        self.rnn_type = rnn_type

        # stateless spectral encoder
        if connectivity == 'LC':
            self.encoder_layers = nn.Sequential(
                layers.LocallyConnected1d(input_size=self.F, in_channels=self.C_in, out_channels=self.C, kernel_size=self.K, stride=self.S),
                layers.CausalLayerNorm(self.C, dim=1),
                nn.Sigmoid()
            )
        elif connectivity == 'FC':
            F_down = int((self.F - kernel_size) / self.S + 1)
            self.encoder_layers = nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),                               # (B, C_in, F) --> (B, C_in * F)
                nn.Linear(self.C_in * self.F, self.C * F_down),                     # (B, C_in * F) --> (B, C * F_down)
                nn.Unflatten(dim=-1, unflattened_size=(self.C, F_down)),            # (B, C, F_down)
                layers.CausalLayerNorm(self.C, dim=1),
                nn.Sigmoid()
            )
        elif connectivity == 'CONV':
            self.encoder_layers = nn.Sequential(
                nn.Conv1d(self.C_in, self.C, kernel_size=self.K, stride=self.S),    # (B, C, F)
                layers.CausalLayerNorm(self.C, dim=1),
                nn.Sigmoid()
            )

        # get output shape and embedding space dim for GRUs
        self.L = (n_frequency_bands - self.K) // self.S + 1
        self.H = self.L * self.C

        # RNNs for temporal processing
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif (self.rnn_type == 'vanilla') or (self.rnn_type == 'RNN'):
            self.rnn = nn.RNN(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif self.rnn_type == 'LMU':
            self.rnn = LMU(input_size=self.H, hidden_size=self.H, memory_size=128, theta=99, learn_a=False, learn_b=False)  # ok perfs but slow ! try memory_size=1024 and theta=50
        elif self.rnn_type == "Mamba":
            self.rnn = MambaBlock(MambaConfig(d_model=self.H, n_layers=1))
        elif self.rnn_type == "S4":
            from deepSTRF.models.dependencies.s4 import S4Block
            self.rnn = S4Block(d_model=self.H, transposed=False)
        else:
            raise NotImplementedError(f"received unknown rnn_type '{rnn_type}': please choose between: "
                                      f"'GRU' (default), 'LSTM', 'RNN', 'vanilla', 'LMU', 'S4', 'Mamba'")

        # readout from RNNs' hidden state
        self.fc = nn.Linear(self.H, self.O)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering(x)                                    # (B, 1|2, F, T)
        y = y.permute(3, 0, 1, 2)                                   # (T, B, 2, F)

        # pass through spectral encoder efficiently
        y_shape = [y.shape[0], y.shape[1]]
        y = y.flatten(0, 1)
        y = self.encoder_layers(y)
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)                                         # (T, B, C_hidd, F_down)

        # prepare for RNN
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)      # (B, T, C_hidd*F_down)

        # RNN
        if self.rnn_type != "Mamba":
            y, _ = self.rnn(y)      # (B, T, H)
        else:
            y = self.rnn(y)         # for mamba

        y = self.fc(y)              # (B, T, N)
        y = self.output_activation(y)  # (B, T, N)


        y = y.permute(0, 2, 1)      # (B, N, T)  TODO: --> (B, N, 1, T)
        return y
