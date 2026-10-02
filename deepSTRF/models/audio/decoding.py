"""Decoding models: neural activity → audio spectrogram.

Mirror image of :class:`~deepSTRF.models.audio.audio_model.AudioEncodingModel`.
A decoder reads the dataset's ``responses`` and predicts its ``stims``:

    input   responses  (B, N, R, T)  — repeats are averaged (NaN-aware), or (B, N, T)
    output  spectrogram (B, 1, F, T) — same layout as an audio dataset's stims

Decoders carry ``direction = "decoding"``; :class:`~deepSTRF.training.Fitter`
reads it to swap input and target.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepSTRF.models.neural_model import NeuralModel


def collapse_repeats(responses: torch.Tensor) -> torch.Tensor:
    """``(B, N, R, T)`` → ``(B, N, T)``: NaN-aware mean over repeats, NaN → 0.

    Missing (NaN) neurons / time bins become 0 — "no spikes observed" — so they
    contribute nothing to a linear read-out. ``(B, N, T)`` passes through
    (with the same NaN → 0 replacement).
    """
    if responses.dim() == 4:
        all_nan = responses.isnan().all(dim=2)
        x = torch.nan_to_num(responses, nan=0.0).sum(2)
        n = (~responses.isnan()).sum(2).clamp(min=1)
        x = torch.where(all_nan, torch.zeros_like(x), x / n)
        return x
    if responses.dim() == 3:
        return torch.nan_to_num(responses, nan=0.0)
    raise ValueError(f"expected responses (B, N, R, T) or (B, N, T), got {tuple(responses.shape)}")


def valid_time_mask(responses: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` bool: time bins where at least one neuron has data.

    ``neural_collate`` NaN-pads responses past each item's length, so this
    recovers the per-item length inside a padded batch.
    """
    if responses.dim() == 4:
        return (~responses.isnan()).any(dim=2).any(dim=1)
    return (~responses.isnan()).any(dim=1)


class AudioDecodingModel(NeuralModel):
    """Base class for decoders of audio from neural activity.

    Forward signature: responses ``(B, N, R, T)`` (or ``(B, N, T)``) →
    spectrogram ``(B, 1, F, T)``. Subclasses implement :meth:`decode` on the
    repeat-averaged ``(B, N, T)`` input; :meth:`forward` does the collapse.

    ``self.O`` (the base class's output width) is ``F``: the decoder's outputs
    are frequency bands, so per-band metrics play the role of per-neuron ones.

    Parameters
    ----------
    n_neurons : int
        Input width ``N`` (neurons / channels).
    n_frequency_bands : int
        Output width ``F`` (spectrogram bands).
    """

    direction = "decoding"

    def __init__(self, n_neurons: int, n_frequency_bands: int, *args, **kwargs):
        super().__init__(out_neurons=n_frequency_bands, *args, **kwargs)
        self.N_in = n_neurons
        self.F = n_frequency_bands

    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        return self.decode(collapse_repeats(responses))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, N, T)`` activity to a ``(B, 1, F, T)`` spectrogram."""
        raise NotImplementedError

    def validate(self):
        super().validate()
        assert isinstance(self.N_in, int) and self.N_in > 0, \
            f"self.N_in must be a positive int (got {self.N_in!r})"
        assert isinstance(self.F, int) and self.F > 0, \
            f"self.F must be a positive int (got {self.F!r})"


class WienerFilter(AudioDecodingModel):
    """Lagged linear decoder (Wiener filter / linear reconstruction filter).

    Each output band is a linear function of every neuron's activity over a
    window of ``n_lags`` bins::

        y_f(t) = b_f + sum_n sum_{j=0}^{n_lags-1} W[f, n, j] * x_n(t - lag_offset - j)

    With ``lag_offset >= 0`` (default 0) the decoder is causal: the output at
    ``t`` only uses activity up to ``t`` — the natural direction for *motor*
    areas, whose activity precedes the behaviour. For *sensory* areas, whose
    responses lag the stimulus, use a negative ``lag_offset`` (e.g.
    ``lag_offset = -(n_lags - 1)`` reads ``x(t) … x(t + n_lags - 1)``, the
    classic stimulus-reconstruction window).

    Fit in closed form with ridge regression (:meth:`fit`), the standard way to
    estimate a Wiener filter, or train it by gradient descent like any other
    model (it is a ``Conv1d``).

    Parameters
    ----------
    n_neurons, n_frequency_bands : int
        Input width ``N`` and output width ``F``.
    n_lags : int, default 10
        Window length in bins.
    lag_offset : int, default 0
        Lag of the most recent bin read (0 = current bin; negative = future).

    References
    ----------
    Wiener filters (ridge regression with history) are the standard linear
    baseline for decoding behaviour from neural populations, e.g. the FALCON
    benchmark baselines: Karpowicz et al. (2024), "Few-shot Algorithms for
    Consistent Neural Decoding (FALCON) Benchmark", *NeurIPS Datasets and
    Benchmarks Track*.

    Linear reconstruction of a sound spectrogram from neural activity:
    Pasley et al. (2012), "Reconstructing Speech from Human Auditory Cortex",
    *PLoS Biology* 10(1): e1001251.
    """

    def __init__(self, n_neurons: int, n_frequency_bands: int, n_lags: int = 10,
                 lag_offset: int = 0):
        super().__init__(n_neurons, n_frequency_bands)
        if n_lags < 1:
            raise ValueError(f"n_lags must be >= 1, got {n_lags}")
        self.n_lags = int(n_lags)
        self.lag_offset = int(lag_offset)
        self.readout = nn.Conv1d(n_neurons, n_frequency_bands, kernel_size=self.n_lags)
        self.validate()

    # padding such that output t reads input [t - lag_offset - (L-1), t - lag_offset]
    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, T)`` → ``(B, N, T, L)`` lag windows, kernel-index order."""
        L, off, T = self.n_lags, self.lag_offset, x.shape[-1]
        pad_l, pad_r = max(0, L - 1 + off), max(0, -off)
        i0 = pad_l - (L - 1) - off
        xp = F.pad(x, (pad_l, pad_r))
        return xp.unfold(-1, L, 1)[..., i0:i0 + T, :]

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        w = self._windows(x)                                     # (B, N, T, L)
        y = torch.einsum("bntl,fnl->bft", w, self.readout.weight) + self.readout.bias[None, :, None]
        return y.unsqueeze(1)                                    # (B, 1, F, T)

    @torch.no_grad()
    def fit(self, loader: Iterable[Mapping[str, torch.Tensor]], alpha: float = 1.0) -> "WienerFilter":
        """Closed-form ridge fit on every batch of ``loader``.

        ``loader`` yields ``neural_collate`` batches (dicts with ``'responses'``
        and ``'stims'``); padded time bins (all-NaN responses) are skipped.
        Solves ``(X'X + alpha I) W = X'Y`` on mean-centred lagged features, then
        sets the bias so the fit passes through the means. Returns ``self``.
        """
        xtx = xty = None
        sx = sy = 0.0
        n = 0
        for batch in loader:
            resp, stim = batch["responses"], batch["stims"]
            if stim.dim() != 4:
                raise ValueError("WienerFilter.fit needs spectrogram targets (B, 1, F, T)")
            valid = valid_time_mask(resp)                        # (B, T)
            x = collapse_repeats(resp).double()
            w = self._windows(x)                                 # (B, N, T, L)
            B, N, T, L = w.shape
            X = w.permute(0, 2, 1, 3).reshape(B * T, N * L)[valid.reshape(-1)]
            Y = stim[:, 0, :, :T].permute(0, 2, 1).reshape(B * T, -1).double()[valid.reshape(-1)]
            if xtx is None:
                xtx = torch.zeros(N * L, N * L, dtype=torch.float64)
                xty = torch.zeros(N * L, Y.shape[1], dtype=torch.float64)
            xtx += X.T @ X
            xty += X.T @ Y
            sx = sx + X.sum(0)
            sy = sy + Y.sum(0)
            n += X.shape[0]
        if n == 0:
            raise ValueError("WienerFilter.fit: loader yielded no valid time bins")
        mx, my = sx / n, sy / n
        # centred normal equations: X_c'X_c = X'X - n mx mx'
        cxx = xtx - n * torch.outer(mx, mx)
        cxy = xty - n * torch.outer(mx, my)
        eye = torch.eye(cxx.shape[0], dtype=torch.float64)
        W = torch.linalg.solve(cxx + alpha * eye, cxy)           # (N*L, F)
        b = my - mx @ W
        self.readout.weight.copy_(W.T.reshape(self.F, self.N_in, self.n_lags).float())
        self.readout.bias.copy_(b.float())
        return self


class MeanTargetBaseline(AudioDecodingModel):
    """Control decoder that ignores the neural input.

    Predicts the time-aligned mean of the training targets (the "template"):
    ``y(t) = mean_i target_i(t)``. For stereotyped behaviour such as zebra-finch
    song, where every trial is the same motif aligned on onset, this is a strong
    baseline; a decoder only earns credit for what it predicts *beyond* it.

    Fit with :meth:`fit` on a loader of training batches. At prediction time
    the template is cropped (or edge-padded) to the input length.
    """

    def __init__(self, n_neurons: int, n_frequency_bands: int):
        super().__init__(n_neurons, n_frequency_bands)
        self.readout = nn.Identity()
        self.register_buffer("template", torch.zeros(n_frequency_bands, 1))
        self.validate()

    @torch.no_grad()
    def fit(self, loader: Iterable[Mapping[str, torch.Tensor]]) -> "MeanTargetBaseline":
        """Average the targets of every batch, per time bin (padded bins skipped)."""
        tot = cnt = None
        for batch in loader:
            resp, stim = batch["responses"], batch["stims"]
            T = resp.shape[-1]
            valid = valid_time_mask(resp).double()               # (B, T)
            s = stim[:, 0, :, :T].double() * valid[:, None, :]
            if tot is None:
                tot, cnt = s.sum(0), valid.sum(0)
            else:
                if s.shape[-1] > tot.shape[-1]:
                    pad = s.shape[-1] - tot.shape[-1]
                    tot, cnt = F.pad(tot, (0, pad)), F.pad(cnt, (0, pad))
                tot[:, :s.shape[-1]] += s.sum(0)
                cnt[:s.shape[-1]] += valid.sum(0)
        if tot is None:
            raise ValueError("MeanTargetBaseline.fit: empty loader")
        self.template = (tot / cnt.clamp(min=1)).float()
        return self

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        tpl = self.template
        if tpl.shape[-1] < T:
            tpl = F.pad(tpl[None], (0, T - tpl.shape[-1]), mode="replicate")[0]
        return tpl[None, None, :, :T].expand(B, 1, -1, -1).to(x.dtype)
