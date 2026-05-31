import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


#    #########################
#       POSITIONAL ENCODINGS
#    #########################

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017,
    "Attention Is All You Need"). Computed on the fly per forward pass,
    so the same module generalizes to arbitrary sequence lengths.

    Parameters
    ----------
    d_model : int
        Embedding dimension. Must be even (the encoding alternates
        ``sin`` and ``cos`` along the channel axis).

    Notes
    -----
    Adds the encoding to the input rather than returning it separately.
    Input shape ``(B, L, d_model)``, output shape ``(B, L, d_model)``.

    Per-dimension frequencies follow the standard
    ``1 / 10000^(2i / d_model)`` schedule and are stored as a non-trained
    buffer so they follow ``.to(device)``.
    """
    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        i = torch.arange(0, d_model, 2, dtype=torch.float)
        self.register_buffer(
            'div_term', torch.exp(-i * (math.log(10000.0) / d_model))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(
                f"input has d_model={D}, expected {self.d_model}"
            )
        pos = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(1)  # (L, 1)
        angles = pos * self.div_term.to(x.dtype)                            # (L, d_model//2)
        pe = torch.empty(L, D, dtype=x.dtype, device=x.device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return x + pe.unsqueeze(0)


def build_causal_window_mask(L: int, window: int = None,
                             device=None) -> torch.Tensor:
    """Build a ``(L, L)`` attention mask for causal (optionally windowed) self-attention.

    Position ``i`` attends to position ``j`` iff ``j <= i`` (causal) and,
    when ``window`` is set, ``i - j < window`` (otherwise the past is
    unlimited).

    Parameters
    ----------
    L : int
        Sequence length.
    window : int, optional
        Maximum look-back distance. ``None`` (default) allows attending to
        the entire past.
    device : torch.device, optional
        Device for the returned mask.

    Returns
    -------
    torch.Tensor
        A ``(L, L)`` bool tensor where ``True`` means "mask out / forbid
        attention" — matching ``nn.TransformerEncoderLayer`` and
        ``F.scaled_dot_product_attention``.
    """
    # forbid future: True above the diagonal
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device),
                      diagonal=1)
    if window is not None and window > 0:
        # forbid the too-far past: True at distance >= window below the diagonal
        mask = mask | torch.tril(torch.ones(L, L, dtype=torch.bool, device=device),
                                 diagonal=-window)
    return mask


#    #########################
#       NORMALIZATION
#    #########################

class CausalLayerNorm(nn.Module):
    """
    LayerNorm applied to a non-trailing axis of its input — equivalently,
    LayerNorm computed independently at every position of every other axis.

    Strictly causal: never pools statistics across time. Drop-in replacement
    for ``nn.BatchNorm{1,2,3}d`` in models that need to stay causal.

    Parameters
    ----------
    normalized_shape : int
        Size of the axis being normalized.
    dim : int, default 1
        Index of the axis to normalize. ``dim=1`` (default) targets the
        channel axis of an ``(B, C, ...)`` tensor; use ``dim=-2`` to
        normalize the frequency axis of an ``(B, C, F, T)`` audio
        spectrogram (the axis just before time).
    eps : float, default 1e-5
        Numerical stability term forwarded to ``nn.LayerNorm``.
    elementwise_affine : bool, default True
        Whether to learn per-element scale and shift.
    """
    def __init__(self, normalized_shape, dim: int = 1, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(normalized_shape, eps=eps,
                               elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.ln(x.movedim(self.dim, -1)).movedim(-1, self.dim)


#    #########################
#       ACTIVATION FUNCTIONS
#    #########################

class LearnableExponentialDecay(nn.Module):
    """Per-band learnable exponential-decay low-pass filter.

    Convolves each frequency band of a ``(B, 1, F, T)`` spectrogram with a
    causal exponential kernel whose time constant is learned per band, and
    returns a low-pass version of the same shape. The decay parameterization
    follows Rahman et al. (DNet) and Fang et al. (PLIF).

    Parameters
    ----------
    input_size : int
        Number of frequency bands ``F`` (one learnable time constant each).
    kernel_size : int
        Temporal extent ``K`` of the decay kernel in frames.
    init_tau : float, default 2.0
        Mean initial time constant (frames) for the decay parameters.
    decay_input : bool, default True
        If True, scale the kernel so the filtered input keeps unit DC gain.

    Notes
    -----
    Currently single input / output channel only, and processes a 2-D
    ``(B, 1, F, T)`` input as a stack of 1-D temporal convolutions.
    """
    def __init__(self, input_size: int, kernel_size: int, init_tau: float = 2., decay_input: bool = True):
        super(LearnableExponentialDecay, self).__init__()

        # general attributes
        self.input_size = input_size
        self.K = kernel_size
        self.decay_input = decay_input

        # initialization as in Rahman et al.
        init_d = torch.ones(input_size).exponential_(lambd=(1/math.sqrt(init_tau - 1.)))

        # learnable parameters (one per feature, so C_out in total)
        self.d = Parameter(init_d)

    def build_kernel(self, device='cpu'):
        """Build the per-band decay kernel of shape ``(input_size, 1, kernel_size)``.

        The kernel is convolved with the last (temporal) axis of the input.

        Parameters
        ----------
        device : torch.device or str, default 'cpu'
            Device on which to build the kernel.

        Returns
        -------
        torch.Tensor
            Kernel of shape ``(input_size, 1, kernel_size)``.
        """
        kernel = torch.ones(self.input_size, 1, self.K).to(device)
        kernel = kernel * (1 - 1 / (1 + self.d ** 2)).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.K)
        kernel = kernel ** torch.arange(0, self.K).flip(0).to(device)
        kernel = kernel / (1 + self.d ** 2).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.K) if self.decay_input else kernel
        return kernel.to(device)

    def forward(self, x):
        """Low-pass each frequency band of a 1-channel spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram of shape ``(B, 1, F, T)`` (B=batch, 1 channel,
            F frequency bands, T timesteps).

        Returns
        -------
        torch.Tensor
            Low-pass-filtered spectrogram of the same shape.
        """
        # build exponential kernel
        kernel = self.build_kernel(x.device)

        # convolve input spectrogram with the kernel
        x = x.squeeze(2)                                                        # (B, 1, F, T)  --> (B, F, T)
        x = nn.functional.pad(x, pad=(self.K - 1, 0), mode='replicate')         # (B, F, T) --> (B, F, T+K-1)   # TODO: padding à gauche!
        x = nn.functional.conv1d(x, kernel, stride=1, groups=self.input_size)   # (B, F, T+K-1) --> (B, F, T)
        x = x.unsqueeze(2)                                                      # (B, F, T) --> (B, 1, F, T)

        return x

    def tau(self):
        with torch.no_grad():
            tau = 1 + self.d ** 2
        return tau


#    ##################
#       WEIGHT LAYERS
#    ##################

class CausalSTRFConv(nn.Module):
    """
    A causal Spectro-Temporal Receptive Field convolution: ``T-1`` zeros
    are prepended along the time axis, then a 2D STRF kernel of shape
    ``(C_out, C_in, F, T)`` is applied with valid padding. Output time
    length matches input time length.

    The actual STRF kernel module is pluggable via the ``kernel`` kwarg.
    The default is a plain ``nn.Conv2d``; passing ``ParametricSTRF`` or a
    separable-kernel ``nn.Sequential`` swaps in alternative
    parameterizations without changing the model that holds this layer.

    Parameters
    ----------
    F : int
        Spectrogram frequency bins (the height of the kernel).
    T : int
        Temporal extent of the STRF in frames (the width of the kernel).
    C_in : int
        Input channel count (typically 1, or 2 with AdapTrans prefiltering).
    C_out : int
        Output channel count. Hidden width when used inside a core,
        ``N`` when used inside an STRF readout.
    kernel : nn.Module, optional
        Pre-built kernel module. Must produce
        ``(B, C_out, 1, T_in)`` from ``(B, C_in, F, T_in + T - 1)``.
        ``None`` (default) instantiates a vanilla
        ``nn.Conv2d(C_in, C_out, kernel_size=(F, T))``.
    bias : bool, default True
        Used only when ``kernel is None``.
    """
    def __init__(self, F: int, T: int, C_in: int, C_out: int,
                 kernel: nn.Module = None, bias: bool = True):
        super().__init__()
        self.F = F
        self.T = T
        self.C_in = C_in
        self.C_out = C_out
        self.pad = nn.ZeroPad2d((T - 1, 0, 0, 0))
        if kernel is None:
            self.kernel = nn.Conv2d(C_in, C_out, kernel_size=(F, T), bias=bias)
        else:
            self.kernel = kernel

    def forward(self, x):
        return self.kernel(self.pad(x))

    def STRF_weight(self):
        """
        Return the effective STRF kernel as a ``(C_out, C_in, F, T)`` tensor,
        detached and on CPU.

        Works across kernel types: vanilla ``nn.Conv2d``, ``ParametricSTRF``
        (DCLS), and the frequency-time separable ``nn.Sequential`` variant.
        """
        if hasattr(self.kernel, 'build_kernel'):
            return self.kernel.build_kernel().detach().cpu()
        if isinstance(self.kernel, nn.Conv2d):
            return self.kernel.weight.data.detach().cpu()
        if isinstance(self.kernel, nn.Sequential) and len(self.kernel) == 2:
            # Separable: nn.Sequential(Conv2d(C_in, C_out, (F, 1)),
            #                          Conv2d(C_out, C_out, (1, T), groups=C_out))
            #   Effective kernel = outer product of the two factors.
            wf = self.kernel[0].weight.data       # (C_out, C_in, F, 1)
            wt = self.kernel[1].weight.data       # (C_out, 1,    1, T)
            return (wf * wt).detach().cpu()       # (C_out, C_in, F, T)
        raise NotImplementedError(
            f"STRF_weight() not implemented for kernel of type {type(self.kernel).__name__}")


class ParametricSTRF(nn.Module):
    """
    Spectro-Temporal Receptive Field kernel parameterized as a sum of
    learnable 2D Gaussians on the ``(F, T)`` grid.

    A direct PyTorch reimplementation of the DCLS Gaussian-mixture
    parameterization (Khalfaoui-Hassani et al. 2023, ICLR), free from
    the upstream library's silent asymmetric-kernel bug. Each of the
    ``num_gaussians`` Gaussians has:

      - a 2D position ``(f, t)`` in the kernel grid coordinates
        ``[0, F-1] × [0, T-1]``,
      - per-axis standard deviations ``(sigma_f, sigma_t)``,
      - per-(C_out, C_in) weight.

    The effective ``(C_out, C_in, F, T)`` kernel is the weighted sum of
    Gaussians, normalized so each Gaussian has unit mass on the grid
    (DCLS convention).

    Parameters
    ----------
    F : int
        Frequency bins of the kernel.
    T : int
        Temporal extent of the kernel in frames.
    C_in, C_out : int
        Input / output channel counts.
    num_gaussians : int, default 1
        Number of Gaussians per ``(C_out, C_in)`` slot.
    bias : bool, default True
        Whether to add a per-output-channel bias (conv2d convention).

    References
    ----------
    Khalfaoui-Hassani, Pellegrini & Masquelier (2023).
    "Dilated Convolution with Learnable Spacings." ICLR.

    Notes
    -----
    The upstream `DCLS` library's `ConstructKernel2d` silently
    mishandles asymmetric kernels: for ``dilated_kernel_size=(F, T)``
    with ``F != T``, its position-offset step ``+lim//2`` adds the
    F-half-width to the T-axis position parameter and vice versa,
    concentrating Gaussians near the centre of one axis and outside
    the grid on the other. This is the cause of the observed "Gaussians
    don't populate the entire STRF window" behavior on auditory STRF
    shapes like ``(34, 9)``. The deepSTRF reimplementation parametrizes
    positions in *absolute* grid coordinates ``[0, F-1] × [0, T-1]``,
    avoids the offset entirely, and removes the optional DCLS
    dependency.
    """
    def __init__(self, F: int, T: int, C_in: int, C_out: int,
                 num_gaussians: int = 1, bias: bool = True):
        super().__init__()
        self.F = F
        self.T = T
        self.C_in = C_in
        self.C_out = C_out
        self.G = num_gaussians

        # P is stored as (axis, C_out, C_in, G); axis 0 = F-coord, axis 1 = T-coord.
        # Coordinates are absolute positions on the kernel grid.
        self.P = nn.Parameter(torch.empty(2, C_out, C_in, num_gaussians))
        self.SIG = nn.Parameter(torch.empty(2, C_out, C_in, num_gaussians))
        self.weight = nn.Parameter(torch.empty(C_out, C_in, num_gaussians))

        # init: positions uniformly distributed across the full grid;
        # sigmas constant; weights kaiming-uniform.
        nn.init.uniform_(self.P[0], 0.0, float(F - 1))
        nn.init.uniform_(self.P[1], 0.0, float(T - 1))
        nn.init.constant_(self.SIG, 1.0)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # bias term — one per output channel (conv2d convention)
        if bias:
            self.bias = nn.Parameter(torch.zeros(C_out))
            nn.init.uniform_(self.bias, -1.0, 1.0)
        else:
            self.bias = None

    def build_kernel(self, device=None):
        """Build the effective ``(C_out, C_in, F, T)`` kernel from the K Gaussians."""
        device = device if device is not None else self.P.device

        # grid coordinates, broadcasted to (F, T, 1, 1, 1)
        f_grid = torch.arange(self.F, device=device).view(self.F, 1, 1, 1, 1).float()
        t_grid = torch.arange(self.T, device=device).view(1, self.T, 1, 1, 1).float()

        # sigmas with floor (DCLS convention) so they stay positive even at init=0
        sig_f = self.SIG[0].abs() + 0.27   # (C_out, C_in, G)
        sig_t = self.SIG[1].abs() + 0.27

        # normalized distances to each Gaussian centre
        df = (f_grid - self.P[0]) / sig_f  # (F, T, C_out, C_in, G)
        dt = (t_grid - self.P[1]) / sig_t

        # 2D Gaussian, normalized to unit mass on the grid (DCLS convention)
        gauss = torch.exp(-0.5 * (df ** 2 + dt ** 2))
        gauss = gauss / (gauss.sum(dim=(0, 1), keepdim=True) + 1e-7)

        # weighted sum over Gaussians: (F, T, C_out, C_in, G) * (C_out, C_in, G)
        kernel = (gauss * self.weight).sum(dim=-1)            # (F, T, C_out, C_in)
        return kernel.permute(2, 3, 0, 1)                     # (C_out, C_in, F, T)

    def forward(self, x):
        # x: (B, C_in, F, T). No internal padding — caller handles temporal
        # padding (typically via an outer ZeroPad2d for left-only causal pad).
        kernel = self.build_kernel(x.device)
        return torch.nn.functional.conv2d(x, kernel, self.bias, stride=(1, 1))


class SeparableSTRF(nn.Module):
    """Frequency-time separable Spectro-Temporal Receptive Field (2D) kernel.

    The effective ``(C_out, C_in, F, T)`` kernel is the rank-1 outer
    product ``w_F(f) · w_T(t)`` of two per-``(C_out, C_in)`` factors.
    Drastically reduces parameter count compared to a vanilla
    ``nn.Conv2d`` STRF (``C_out·C_in·(F + T)`` vs ``C_out·C_in·F·T``)
    while preserving the conv2d call signature so it drops in as a
    ``kernel=`` arg on any STRFReadout-using model.

    Parameters
    ----------
    F : int
        Frequency bins of the kernel.
    T : int
        Temporal extent of the kernel in frames.
    C_in, C_out : int
        Input / output channel counts.
    bias : bool, default True
        Whether to add a per-output-channel bias (conv2d convention).
    """
    def __init__(self, F: int, T: int, C_in, C_out, bias: bool = True):
        super(SeparableSTRF, self).__init__()

        self.F = F
        self.T = T
        self.C_in = C_in
        self.C_out = C_out

        # Per-(C_out, C_in) frequency and temporal factors. Shapes are chosen
        # so the rank-1 outer product weight_f * weight_t broadcasts directly
        # to a (C_out, C_in, F, T) conv2d-compatible kernel.
        self.weight_f = torch.nn.Parameter(torch.empty(self.C_out, self.C_in, F, 1))
        self.weight_t = torch.nn.Parameter(torch.empty(self.C_out, self.C_in, 1, T))

        # initialization
        torch.nn.init.kaiming_uniform_(self.weight_f)
        torch.nn.init.kaiming_uniform_(self.weight_t)

        # bias term — one per output channel (conv2d convention)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.C_out))
            torch.nn.init.uniform_(self.bias, -1., 1.)
        else:
            self.bias = None

    def build_kernel(self, device='cpu'):
        # (C_out, C_in, F, 1) * (C_out, C_in, 1, T) → (C_out, C_in, F, T)
        kernel = self.weight_f * self.weight_t
        return kernel.to(device)

    def forward(self, x):
        # x: (B, C_in, F, T). No internal padding — caller handles temporal
        # padding. See ParametricSTRF.forward for rationale.
        strf_kernel = self.build_kernel(x.device)
        out = torch.nn.functional.conv2d(x, strf_kernel, self.bias, stride=(1, 1))
        return out


class LocallyConnected1d(nn.Module):
    """Locally connected (LC) layer for 1-D tensors — a trade-off between Linear and Conv1d.

    Like a convolution, but the kernel weights are *not* shared across
    positions: every output position has its own filter.

    Notes
    -----
    ``nn.Unfold`` only accepts images, so 1-D inputs are first unsqueezed to
    2-D, the unfold / conv operations are performed, and the result is
    squeezed back to 1-D.
    """
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(LocallyConnected1d, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = True

        self.S_in = input_size
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size

        prospective_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        prospective_input = torch.rand(1, in_channels, input_size)
        prospective_output = prospective_conv(prospective_input)
        self.S_out = prospective_output.shape[-1]

        self.unfold = torch.nn.Unfold((kernel_size, 1), dilation, padding, stride)
        prospective_unfold = self.unfold(prospective_input.unsqueeze(-1))
        self.L = prospective_unfold.shape[-1]

        self.weights = Parameter(torch.rand(self.C_out, self.C_in * self.K, self.L))
        self.biases = Parameter(torch.rand(self.C_out, self.L)) if bias else None

        self.fold = torch.nn.Fold(output_size=(self.S_out, 1), kernel_size=(1, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, C_in, S_in)       --> (B, C_in, S_in, 1)
        patches = self.unfold(x)  # (B, C_in, H_in, W_in) --> (B, M, L)
        patches = patches.unsqueeze(1).repeat(1, self.C_out, 1, 1)  # (B, M, L)             --> (B, C_out, M, L)

        if self.bias:
            y = torch.sum(patches * self.weights, dim=2) + self.biases  # (B, C_out, M, L)      --> (B, C_out, L)
        else:
            y = torch.sum(patches * self.weights, dim=2)

        return y  # in 1D, fold is the identity since L = S_out — torch.equal(y, self.fold(y).squeeze(-1)) holds

    def __str__(self):
        s = f'LocallyConnected1d(input_size={(self.S_in,)}, in_channels={self.C_in}, out_channels={self.C_out}, ' \
            f'kernel_size={self.K}, stride={self.stride}'
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s