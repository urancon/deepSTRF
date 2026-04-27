import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from DCLS.construct.modules import Dcls1d, ConstructKernel2d


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
    """
    TODO: description --> Rahman et al. (DNet) + Fang et al. (PLIF)

    TODO:
     - multiple input channel       | Now: C_in=1 only
     - multiple output channel      | Now: C_out=1 only
     - make 1d and 2d versions      | Now: 1d processing with expected 2d input

    Takes a (B, F, T) tensor as input, and returns a low-pass version of the same shape

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
        """
        Creates a parametrized kernel to be convolved with the last (temporal) dimension of the input tensor
        This output kernel has for shape: (input_size, 1, kernel_size)

        """
        kernel = torch.ones(self.input_size, 1, self.K).to(device)
        kernel = kernel * (1 - 1 / (1 + self.d ** 2)).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.K)
        kernel = kernel ** torch.arange(0, self.K).flip(0).to(device)
        kernel = kernel / (1 + self.d ** 2).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.K) if self.decay_input else kernel
        return kernel.to(device)

    def forward(self, x):
        """
        Convolves each frequency band of input 1-channel spectrogram with filters and standardize the output.

        :param spectro_in: shape is (B, C, F, T) with B=Batch, C=Channels=1 (raw spectrogram), F=#Frequency_bands, T=#Timesteps
        :return: a tensor of the same shape
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
    SPECTRO-Temporal Receptive Field (2D) kernel.

    The kernel is parameterized as a set of gaussians:
      - at a given (x,y) position,
      - with a given variance (sigma_x, sigma_y) along both dimensions
      - with a given weight value
        ==> 5 degrees of freedom per gaussian.

    Drastically reduces the number of learnable parameters.

    See Khalfaoui-Hassani et al. (2023), "Dilated convolutions with learnable spacings (DCLS)", ICLR,
    """
    def __init__(self, F: int, T: int, C_in, C_out, num_gaussians: int = 1, bias: bool = True):
        super(ParametricSTRF, self).__init__()

        self.F = F
        self.T = T
        self.C_in = C_in
        self.C_out = C_out
        self.G = num_gaussians

        # DCLS = parametrized STRF kernel and its parameters
        self.DCK = ConstructKernel2d(in_channels=self.C_in, out_channels=self.C_out, groups=1, kernel_count=self.G, dilated_kernel_size=(F, T), version='gauss')
        self.P = torch.nn.Parameter(torch.rand(2, self.C_out, self.C_in, self.G))       # positions
        self.SIG = torch.nn.Parameter(torch.rand(2, self.C_out, self.C_in, self.G))     # sigmas
        self.weight = torch.nn.Parameter(torch.rand(self.C_out, self.C_in, self.G))     # values

        # initialize parameters (recommended by Ismail)
        torch.nn.init.uniform_(self.P.select(0, 0), -F / 2, F / 2)
        torch.nn.init.uniform_(self.P.select(0, 1), -T / 2, T / 2)
        torch.nn.init.constant_(self.SIG, 0.23)
        torch.nn.init.kaiming_uniform_(self.weight)

        # bias term — one per output channel (conv2d convention)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.C_out))
            torch.nn.init.uniform_(self.bias, -1., 1.)
        else:
            self.bias = None

    def build_kernel(self, device='cpu'):
        # create a (C_out, C_in, Kf, Kt) kernel
        kernel = self.DCK(self.weight, self.P, self.SIG)
        return kernel.to(device)

    def forward(self, x):
        # x: (B, C_in, F, T). No internal padding — caller handles temporal
        # padding (typically via an outer ZeroPad2d for left-only causal pad).
        # This matches nn.Conv2d's no-pad default and avoids double-padding
        # when an outer model (e.g. Linear, DNet) already pads.
        strf_kernel = self.build_kernel(x.device)
        out = torch.nn.functional.conv2d(x, strf_kernel, self.bias, stride=(1, 1))
        return out


class SeparableSTRF(nn.Module):
    """
    SPECTRO-Temporal Receptive Field (2D) kernel.

    Frequency-time separable.

    Drastically reduces the number of learnable parameters.
    """
    def __init__(self, F: int, T: int, C_in, C_out, bias: bool = True):
        super(SeparableSTRF, self).__init__()

        self.F = F
        self.T = T
        self.C_in = C_in
        self.C_out = C_out

        # parameters
        self.weight_f = torch.nn.Parameter(torch.rand(self.C_in, self.C_out, F, 1))
        self.weight_t = torch.nn.Parameter(torch.rand(self.C_in, self.C_out, 1, T))

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
        # create a (C_out, C_in, Kf, Kt) kernel
        kernel = self.weight_f.unsqueeze(-1) * self.weight_t.unsqueeze(-2)
        return kernel.to(device)

    def forward(self, x):
        # x: (B, C_in, F, T). No internal padding — caller handles temporal
        # padding. See ParametricSTRF.forward for rationale.
        strf_kernel = self.build_kernel(x.device)
        out = torch.nn.functional.conv2d(x, strf_kernel, self.bias, stride=(1, 1))
        return out


class LocallyConnected1d(nn.Module):
    """
    Implementation of Locally Connected (LC) for 1D tensors
    A trade-off between Linear and Conv1d

    Note:
        nn.Unfold is only compatible with images, so for 1d inputs, it is necessary to first unsqueeze them to 2d,
        perform the unfold/conv operations, and finally squeeze them back from 2d to 1d

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