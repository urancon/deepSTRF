"""Output activations for deepSTRF readouts.

Two parametric activations from the auditory-fitting literature, each with
opt-out non-negativity reparameterisation that pairs naturally with
``poisson_loss(log_input=False)`` (see ``metrics_paradigm.md`` §6.2).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricSigmoid(nn.Module):
    """4-parameter parametric sigmoid (Willmore et al. 2016).

    Per-neuron output:

    .. math::

        f(x) = b \\cdot \\sigma((x - c) / d) + a       \\quad\\text{(bias=True)}

    where ``b`` is the dynamic range, ``a`` the baseline (minimum firing
    rate), ``c`` the input inflection point, and ``d`` the reciprocal of
    the gain.

    Parameters
    ----------
    num_features : int
        ``N``: number of independent per-neuron parameter sets.
    bias : bool, default True
        Whether to include the additive baseline ``a``.
    non_negative_output : bool, default True
        When True, ``b`` (and ``a``, if ``bias=True``) are stored as raw
        parameters and softplus-mapped to the strictly-positive half-line
        at every forward pass. This guarantees a non-negative output
        curve, suitable for spike-count targets and ``poisson_loss``. When
        False, parameters are direct (signed-output mode) — useful for
        LFP / EEG / centred PSTH targets where outputs may legitimately
        be negative.

    Notes
    -----
    The shipped behaviour replaces an earlier closure-based implementation
    that built ``forward`` inside ``__init__``. The current version uses a
    standard ``forward()`` method and exposes ``b`` and ``a`` via
    ``@property`` so that ``softplus`` is re-applied on the live parameter
    values at every step (ensures ``state_dict`` round-trips and
    parameter-replacement work correctly).

    References
    ----------
    Willmore, B. D. B., Schoppe, O., King, A. J., Schnupp, J. W. H. &
    Harper, N. S. (2016). "Incorporating midbrain adaptation to mean sound
    level improves models of auditory cortical processing." *Journal of
    Neuroscience*, 36(2), 280–289.
    """

    def __init__(
        self,
        num_features: int,
        bias: bool = True,
        non_negative_output: bool = True,
    ):
        super().__init__()
        self.N = num_features
        self.bias = bias
        self.non_negative_output = non_negative_output

        # Inflection point and gain are unconstrained.
        self.c = nn.Parameter(torch.empty(self.N))
        self.d = nn.Parameter(torch.empty(self.N))
        nn.init.uniform_(self.c, -0.5, 0.5)
        nn.init.uniform_(self.d, 0.5, 1.5)

        # Amplitude (and optionally baseline) gate non-negativity. Their raw
        # storage is `_raw_b` / `_raw_a`; the public attributes `b` / `a` are
        # @property views that apply softplus when non_negative_output=True.
        self._raw_b = nn.Parameter(torch.empty(self.N))
        if non_negative_output:
            # softplus(_raw_b) ~ uniform(0.5, 1.5) at init
            nn.init.uniform_(self._raw_b, -0.43, 1.40)
        else:
            nn.init.uniform_(self._raw_b, 0.5, 1.5)

        if bias:
            self._raw_a = nn.Parameter(torch.empty(self.N))
            if non_negative_output:
                # softplus(_raw_a) ~ uniform(0.2, 1.0) at init
                nn.init.uniform_(self._raw_a, -1.43, 0.43)
            else:
                nn.init.uniform_(self._raw_a, 0.0, 1.0)

    @property
    def b(self) -> torch.Tensor:
        """Dynamic-range parameter, post-reparameterisation."""
        return F.softplus(self._raw_b) if self.non_negative_output else self._raw_b

    @property
    def a(self) -> Optional[torch.Tensor]:
        """Baseline parameter, post-reparameterisation. ``None`` if ``bias=False``."""
        if not self.bias:
            return None
        return F.softplus(self._raw_a) if self.non_negative_output else self._raw_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.b * torch.sigmoid((x - self.c) / self.d)
        if self.bias:
            out = out + self.a
        return out

    def extra_repr(self) -> str:
        return (
            f"N={self.N}, bias={self.bias}, "
            f"non_negative_output={self.non_negative_output}"
        )


class ParametricDoubleExponential(nn.Module):
    """4-parameter parametric double-exponential (Thorson et al. 2015).

    Per-neuron output:

    .. math::

        f(x) = a \\cdot \\exp(-\\exp(k \\cdot x - s)) + b
        \\quad\\text{(bias=True)}

    where ``a`` is the saturated firing rate, ``b`` the baseline, ``s``
    the firing threshold, and ``k`` the gain.

    Parameters
    ----------
    num_features : int
        ``N``: number of independent per-neuron parameter sets.
    bias : bool, default True
        Whether to include the additive baseline ``b``.
    non_negative_output : bool, default True
        When True, ``a`` (and ``b``, if ``bias=True``) are stored as raw
        parameters and softplus-mapped to the strictly-positive half-line
        at every forward pass. The inner ``exp(-exp(k·x − s))`` term is
        always in ``(0, 1]``, so this fully guarantees ``f(x) ≥ 0``. When
        False, parameters are direct (signed-output mode).

    References
    ----------
    Thorson, I. L., Liénard, J. & David, S. V. (2015). "The essential
    complexity of auditory receptive fields." *PLOS Computational
    Biology*, 11(3), e1004228.
    """

    def __init__(
        self,
        num_features: int,
        bias: bool = True,
        non_negative_output: bool = True,
    ):
        super().__init__()
        self.N = num_features
        self.bias = bias
        self.non_negative_output = non_negative_output

        # Threshold and gain are unconstrained.
        self.k = nn.Parameter(torch.empty(self.N))
        self.s = nn.Parameter(torch.empty(self.N))
        nn.init.uniform_(self.k, -0.5, 0.5)
        nn.init.uniform_(self.s, 0.5, 1.5)

        # Saturated rate (and optionally baseline) gate non-negativity.
        self._raw_a = nn.Parameter(torch.empty(self.N))
        if non_negative_output:
            nn.init.uniform_(self._raw_a, -0.43, 1.40)
        else:
            nn.init.uniform_(self._raw_a, 0.5, 1.5)

        if bias:
            self._raw_b = nn.Parameter(torch.empty(self.N))
            if non_negative_output:
                nn.init.uniform_(self._raw_b, -1.43, 0.43)
            else:
                nn.init.uniform_(self._raw_b, 0.0, 1.0)

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self._raw_a) if self.non_negative_output else self._raw_a

    @property
    def b(self) -> Optional[torch.Tensor]:
        if not self.bias:
            return None
        return F.softplus(self._raw_b) if self.non_negative_output else self._raw_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.a * torch.exp(-torch.exp(self.k * x - self.s))
        if self.bias:
            out = out + self.b
        return out

    def extra_repr(self) -> str:
        return (
            f"N={self.N}, bias={self.bias}, "
            f"non_negative_output={self.non_negative_output}"
        )
