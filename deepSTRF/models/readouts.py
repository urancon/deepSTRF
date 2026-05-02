"""
Readout modules: per-neuron projections that take a feature representation
emitted by a model's ``core`` and produce predictions of shape
``(B, N, R=1, T)``.

Two flavours are shipped:

- :class:`STRFReadout` — a learnable STRF kernel applied causally via
  ``CausalSTRFConv``. Used when the readout itself is the model's main
  learnable apparatus (Linear / LinearNonlinear) or when an explicit
  STRF interpretation of the per-neuron weights is wanted.

- :class:`LinearReadout` — a per-timestep linear projection
  ``in_features -> N``, with an optional 1-hidden-layer MLP. Used by
  models whose ``core`` already produces a flat per-timestep feature
  vector (ConvNet2D / Transformer / StateNet).

See ``docs/_source/md/model_paradigm.md`` §7 for the full readout
contract.
"""
import torch
import torch.nn as nn

from deepSTRF.models.layers import CausalSTRFConv


class STRFReadout(nn.Module):
    """
    Per-neuron readout backed by a causal Spectro-Temporal Receptive
    Field kernel.

    Wraps a ``CausalSTRFConv`` of shape ``(N, C_in, F, T)`` and applies an
    output activation. The frequency axis is collapsed by the conv from
    ``F → 1``, so the readout naturally emits the canonical
    ``(B, N, R=1, T)`` rank.

    Parameters
    ----------
    F : int
        Frequency bins of the input spectrogram.
    T : int
        STRF temporal extent.
    C_in : int
        Input channel count after prefiltering.
    out_neurons : int
        Number of output neurons ``N``.
    kernel : nn.Module, optional
        Pluggable kernel module. ``None`` (default) instantiates a vanilla
        ``nn.Conv2d``; pass ``ParametricSTRF`` (DCLS) or a separable
        ``nn.Sequential`` to swap parameterizations.
    activation : nn.Module, optional
        Pointwise output nonlinearity. Defaults to ``nn.Identity``.
    bias : bool, default True
        Used only when ``kernel is None``.
    """
    def __init__(self, F: int, T: int, C_in: int, out_neurons: int,
                 kernel: nn.Module = None, activation: nn.Module = None,
                 bias: bool = True):
        super().__init__()
        self.strf = CausalSTRFConv(F, T, C_in, out_neurons, kernel=kernel, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        # x: (B, C_in, F, T)
        out = self.strf(x)                                  # (B, N, 1, T)
        # Apply activation on (B, T, N) so per-neuron parametric activations
        # (ParametricSoftplus / ParametricSigmoid / ParametricDoubleExponential)
        # broadcast correctly with N as the last axis. Shape-invariant
        # activations (Identity, nn.Sigmoid, nn.ReLU, ...) are unaffected.
        out = out.squeeze(-2)                                # (B, N, T)
        out = out.transpose(-1, -2)                          # (B, T, N)
        out = self.activation(out)                           # (B, T, N)
        out = out.transpose(-1, -2).unsqueeze(-2)            # (B, N, 1, T)
        return out

    def STRF_weight(self, polarity: str = None):
        """
        Return the underlying STRF kernel as ``(N, C_in, F, T)``.

        Parameters
        ----------
        polarity : {'ON', 'OFF', None}, optional
            If the model's prefilter produces ``C_in == 2`` channels (e.g.
            AdapTrans's ON/OFF), select one. ``None`` (default) returns
            the full ``(N, C_in, F, T)`` tensor; ``'ON'`` slices channel
            0; ``'OFF'`` slices channel 1.
        """
        kernel = self.strf.STRF_weight()
        if polarity is None:
            return kernel
        if polarity in ('ON', 'On', 'on', 0):
            return kernel[:, 0]
        if polarity in ('OFF', 'Off', 'off', 1):
            return kernel[:, 1]
        raise ValueError(f"polarity must be 'ON', 'OFF', or None — got {polarity!r}")


class LinearReadout(nn.Module):
    """
    Per-neuron readout: a per-timestep linear projection from a flat
    feature vector to ``N`` output neurons.

    Accepts either a 3D input ``(B, in_features, T)`` or a 4D input
    ``(B, in_features, 1, T)`` (the singleton spatial axis emitted by an
    STRF-style conv that collapsed ``F → 1``); both shapes route through
    the same projection. The output is the canonical ``(B, N, 1, T)``.

    Parameters
    ----------
    in_features : int
        Per-timestep feature dimension produced by the model's ``core``.
    out_neurons : int
        Number of output neurons ``N``.
    hidden : int, optional
        If given, inserts a 1-hidden-layer MLP
        ``in_features → hidden → N`` with a LeakyReLU(0.1) between.
        ``None`` (default) gives a single linear projection.
    activation : nn.Module, optional
        Pointwise output nonlinearity applied to the per-timestep output
        before the rank is unsqueezed to ``(B, N, 1, T)``. Defaults to
        ``nn.Identity``.
    bias : bool, default True
    """
    def __init__(self, in_features: int, out_neurons: int,
                 hidden: int = None, activation: nn.Module = None,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.N = out_neurons
        if hidden is None:
            self.fc = nn.Linear(in_features, out_neurons, bias=bias)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_features, hidden, bias=bias),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden, out_neurons, bias=bias),
            )
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        if x.dim() == 4:
            assert x.shape[-2] == 1, (
                f"4D input to LinearReadout must have a singleton dim -2 "
                f"(emitted by an F→1 conv); got shape {tuple(x.shape)}"
            )
            x = x.squeeze(-2)                       # (B, C, T)
        elif x.dim() != 3:
            raise ValueError(
                f"LinearReadout expects 3D (B, C, T) or 4D (B, C, 1, T) "
                f"input; got {x.dim()}D shape {tuple(x.shape)}"
            )
        # x: (B, C, T)
        y = x.transpose(-2, -1)                     # (B, T, C)
        y = self.fc(y)                              # (B, T, N)
        y = self.activation(y)                      # (B, T, N) — broadcasts over N
        y = y.transpose(-2, -1)                     # (B, N, T)
        return y.unsqueeze(-2)                      # (B, N, 1, T)
