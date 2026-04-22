"""Shared spike-file loaders for the CRCNS-AA* datasets.

The CRCNS AA1, AA2, (and AA4 to a lesser extent) datasets all store
single-unit spike trains as plain-text ``spikeX`` files — one file per
(neuron, stimulus) pair, one row per repeat, space-separated spike times
in ms relative to stimulus onset (negative = pre-onset spontaneous activity).

These helpers are the only format-specific bits of those loaders; moving
them here avoids duplicating the same ~60 lines across each dataset class.

Private module (leading underscore) — not re-exported by ``deepSTRF.datasets.audio``
since these helpers are only meaningful against the CRCNS AA spike-file format.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def time_binning(spike_times, dt_ms: float = 1.0) -> torch.Tensor:
    """Bin a list of post-stimulus-onset spike times (in ms) into a spike-count tensor.

    Parameters
    ----------
    spike_times : Sequence[float]
        Positive floats (ms) relative to stimulus onset. Pre-onset spikes
        must be filtered out upstream.
    dt_ms : float, default 1.0
        Bin width in ms.

    Returns
    -------
    torch.Tensor
        1-D spike-count tensor of shape ``(T,)`` where
        ``T = floor(max(spike_times) / dt_ms) + 1``. Returns a ``(1,)``
        zero tensor when ``spike_times`` is empty.
    """
    if len(spike_times) == 0:
        return torch.zeros(1)

    bin_indices = [int(t // dt_ms) for t in spike_times]
    T = max(bin_indices) + 1
    counts = [0] * T
    for b in bin_indices:
        counts[b] += 1
    return torch.tensor(counts, dtype=torch.float32)


def load_spike_file(path: str, dt_ms: float = 1.0) -> torch.Tensor:
    """Parse a ``spikeX`` text file into a ``(R, T)`` spike-count tensor.

    One row per repeat, space-separated spike times in ms (relative to stim
    onset). Pre-onset spikes (negative times) are dropped. Rows of different
    lengths are right-padded with zeros so all repeats share the same ``T``.

    Parameters
    ----------
    path : str
        Path to the ``spikeX`` text file.
    dt_ms : float, default 1.0
        Time-bin width in ms.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(R, T)``.
    """
    with open(path, "r") as f:
        responses = []
        for line in f:
            # strip trailing newline marker + empty trailing token
            tokens = line.split(" ")[:-1]
            spike_times = [float(t) for t in tokens]
            spike_times = [t for t in spike_times if t >= 0]  # post-onset only
            responses.append(time_binning(spike_times, dt_ms=dt_ms))

    T_max = max(r.shape[-1] for r in responses)
    for i, r in enumerate(responses):
        if r.shape[-1] < T_max:
            responses[i] = F.pad(r, (0, T_max - r.shape[-1]), mode="constant", value=0.0)

    return torch.stack(responses, dim=0)  # (R, T)
