"""Output activations for deepSTRF readouts.

Three parametric activations, each with opt-out non-negativity
reparameterisation that pairs naturally with
``poisson_loss(log_input=False)`` (see ``metrics_paradigm.md`` §6.2):

- :class:`ParametricSigmoid` — saturating, bounded above (Willmore 2016).
- :class:`ParametricDoubleExponential` — saturating, bounded above
  (Thorson 2015).
- :class:`ParametricSoftplus` — non-saturating, unbounded above. Natural
  default for spike-count regression where the response can take any
  non-negative magnitude.

All three follow the same input-shape contract: parameters are
``(N,)``-shaped per-neuron tensors that broadcast against the last axis
of the input. Use these inside readouts that emit
``(..., N)``-last-axis tensors (e.g. :class:`LinearReadout`).
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


class ParametricSoftplus(nn.Module):
    r"""Per-neuron Softplus with learnable sharpness β and additive baseline.

    Output:

    .. math::

        f(x) = \frac{1}{\beta} \log\!\bigl(1 + \exp(\beta x)\bigr) + b

    where ``β > 0`` is the per-neuron sharpness (β → ∞ approaches ReLU,
    β → 0 approaches a soft linear with slope ½) and ``b`` is the per-neuron
    additive baseline. Both are always learnable per-neuron.

    Unlike :class:`ParametricSigmoid` and
    :class:`ParametricDoubleExponential`, the underlying curve is
    **unbounded above** — natural for spike-count regression on smoothed
    PSTHs that can take any non-negative magnitude (e.g. NS1: peaks
    ~3 spikes/bin after Hsu/Borst/Theunissen 21 ms Hanning smoothing).

    Parameters
    ----------
    num_features : int
        ``N``: number of independent per-neuron parameter sets.
    non_negative_output : bool, default True
        When True, ``b`` is stored as a raw parameter and softplus-mapped
        to the strictly-positive half-line at every forward pass.
        Combined with the always-non-negative softplus core, this
        guarantees ``f(x) ≥ 0`` for every ``x``. When False, ``b`` is
        unconstrained — pair with ``poisson_loss(log_input=True)`` for
        signed-output targets (LFP / EEG / centred PSTH).

    Notes
    -----
    ``β`` is **always** non-negativity-reparameterised
    (``β = softplus(_raw_beta)``) regardless of ``non_negative_output``,
    because a non-positive sharpness would flip the curve and is never
    physically meaningful.

    The implementation expects an input whose **last axis is N** (matching
    the :class:`ParametricSigmoid` / :class:`ParametricDoubleExponential`
    contract); that is what :class:`LinearReadout` and
    :class:`STRFReadout` emit at the activation step.
    """

    def __init__(
        self,
        num_features: int,
        non_negative_output: bool = True,
    ):
        super().__init__()
        self.N = num_features
        self.non_negative_output = non_negative_output

        # β > 0 always (structural softplus on _raw_beta). Init: softplus
        # spans ~5 to ~6 — sharp / near-ReLU at start so f(0) = log(2)/β is
        # small (~0.12-0.14). A milder init around β≈1 makes f(0) ≈ 0.7,
        # which is way above typical spike-count target means (~0.1-0.3) and
        # causes a "mean-collapse" failure mode where training reduces loss
        # by suppressing prediction magnitude rather than learning structure
        # (verified empirically on NS1 + StateNet, 2026-05-02).
        self._raw_beta = nn.Parameter(torch.empty(self.N))
        nn.init.uniform_(self._raw_beta, 5.0, 6.0)

        # Per-neuron baseline. With non_negative_output=True, softplus-reparam
        # so b ≥ 0; init very near zero (softplus → ~0.005 to ~0.05) so the
        # activation does not impose a hard positive floor at start. The
        # gradient on _raw_b is sigmoid(_raw_b), small (~0.01-0.05) but
        # non-vanishing — the optimizer still moves it.
        self._raw_b = nn.Parameter(torch.empty(self.N))
        if non_negative_output:
            nn.init.uniform_(self._raw_b, -5.0, -3.0)
        else:
            nn.init.uniform_(self._raw_b, -0.1, 0.1)

    @property
    def beta(self) -> torch.Tensor:
        """Per-neuron sharpness, always > 0."""
        return F.softplus(self._raw_beta)

    @property
    def b(self) -> torch.Tensor:
        """Per-neuron additive baseline, post-reparameterisation."""
        return F.softplus(self._raw_b) if self.non_negative_output else self._raw_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.beta
        return F.softplus(beta * x) / beta + self.b

    def extra_repr(self) -> str:
        return (
            f"N={self.N}, non_negative_output={self.non_negative_output}"
        )
