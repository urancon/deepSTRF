"""Auditory-nerve model (``Dataset_CC.npy``) helpers for Gosselin 2025.

The release gives 691 model fibres with no metadata. Established from the
data: fibres are ordered by characteristic frequency, and within each inner
hair cell the fibres are ordered by increasing spontaneous rate, so a new
hair cell starts wherever the spontaneous rate stops increasing (51 groups;
the paper's methods say 40). CFs are only known approximately: the methods
give 5-50 kHz, so groups are labelled log-spaced over that range.
"""
from __future__ import annotations

import numpy as np

CF_RANGE_KHZ = (5.0, 50.0)
PRE_ONSET_MS = 200


def ihc_groups_from_spont(spont: np.ndarray) -> np.ndarray:
    """Hair-cell index of each fibre: a new group wherever the spontaneous
    rate does not increase."""
    spont = np.asarray(spont, dtype=np.float64)
    new = np.r_[True, spont[1:] <= spont[:-1]]
    return np.cumsum(new) - 1


def approx_cf_khz(n_groups: int) -> np.ndarray:
    """Approximate CF of each hair cell, log-spaced over the model's range."""
    lo, hi = CF_RANGE_KHZ
    return np.geomspace(lo, hi, n_groups)


def ihc_means(cc: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """(n_groups, T, S) mean rate of each hair cell's fibres, from (fibres, T, S)."""
    G = int(groups.max()) + 1
    return np.stack([cc[groups == g].mean(axis=0) for g in range(G)]).astype(np.float32)


def overlap_weights(n_in: int, bin_ms: float, n_out: int) -> np.ndarray:
    """(n_out, n_in) weights averaging 1 ms samples into ``bin_ms`` bins by
    their overlap (bins need not be whole milliseconds)."""
    W = np.zeros((n_out, n_in))
    j = np.arange(n_in, dtype=np.float64)
    for k in range(n_out):
        b, e = k * bin_ms, (k + 1) * bin_ms
        W[k] = np.clip(np.minimum(j + 1, e) - np.maximum(j, b), 0.0, None)
    return W / W.sum(axis=1, keepdims=True)


def rebin_cochlear(x: np.ndarray, dt_ms: float, n_out: int) -> np.ndarray:
    """(C, n_out) from (C, 900) 1 ms rates. Bins past the model's +700 ms
    end take each channel's pre-onset (spontaneous) mean: the sounds are over
    by then, so that is what the model would output."""
    x = np.asarray(x, dtype=np.float64)
    n_need = int(np.ceil(n_out * dt_ms))
    if n_need > x.shape[1]:
        spont = x[:, :PRE_ONSET_MS].mean(axis=1, keepdims=True)
        x = np.concatenate([x, np.repeat(spont, n_need - x.shape[1], axis=1)], axis=1)
    x = x[:, :n_need]
    return (x @ overlap_weights(n_need, dt_ms, n_out).T).astype(np.float32)
