"""Native (NEMS-free) parser for the Wingert 2026 NEMS-recording archives.

The Wingert 2026 release packages each recording site as a single tarball
containing a NEMS-format dump. The format is well-defined and small
enough to read directly:

  Per-site recording (``recordings/<SITE>_<hash>.tgz``):
    <SITE>/<SITE>.meta.json         siteid, fs, channel count
    <SITE>/<SITE>.resp.h5           PointProcess: one (n_spikes,) float64
                                    array per cell, values are spike times
                                    in seconds.
    <SITE>/<SITE>.resp.json         chans (cell ids), fs=100, segments=[[0, T_total]]
    <SITE>/<SITE>.resp.epoch.csv    (epoch_name, start_s, end_s) — STIM_00*=test,
                                    STIM_seq####=estimation
    <SITE>/<SITE>.stim.h5           TiledSignal: one (F=32, T=2200) dataset
                                    per unique stim, keyed STIM_<wavname>
    <SITE>/<SITE>.stim.json         signal metadata
    <SITE>/<SITE>.stim.epoch.csv    same epoch table as resp

Compared to the NAT4 release (parsed by ``_nat4_native.py``):
 - stim is per-stimulus inside ``stim.h5`` (TiledSignal dict), not a
   continuous CSV — much simpler;
 - resp is spike times at fs=100 (NAT4 per-site is fs=1000);
 - no separate population recording.

We intentionally keep this parser independent of ``_nat4_native.py`` —
the two storage layouts are different enough that a shared module would
confuse more than it shares. The spike-time → bin floor convention is
deliberately identical (``np.floor(spikes_s * fs).astype(int64)``,
``np.add.at`` for collisions), matching NEMS' ``PointProcess.rasterize``.

Validated against the published cell counts (2128 A1 / 746 PEG) and the
6-test-stim invariant (test gtgrams are bit-identical across sites).
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


# Cell ids in Wingert follow ``<site>-<probe-channel>-<unit-on-channel>``,
# e.g. ``CLT027c-009-1`` (site ``CLT027c``, probe channel 9, unit 1) or
# ``PRN018b-039-1``. The site prefix is ``<3-letter animal>NNN<letter>``.
_CELL_ID_RE = re.compile(r"^([A-Za-z]{3})\d+[a-z]?-(\d+)-(\d+)$")


def parse_wingert_cell_id(cell_id: str) -> dict:
    """Best-effort decomposition of a Wingert cell id.

    Returns
    -------
    dict
        Keys ``site``, ``animal``, ``electrode``, ``unit_in_electrode``.
        Any field whose source is missing or unparseable is ``None``.
    """
    out = {"site": None, "animal": None, "electrode": None, "unit_in_electrode": None}
    if not isinstance(cell_id, str) or "-" not in cell_id:
        return out
    parts = cell_id.split("-")
    out["site"] = parts[0]
    m = _CELL_ID_RE.match(cell_id)
    if m is None:
        return out
    out["animal"] = m.group(1)
    out["electrode"] = int(m.group(2))
    out["unit_in_electrode"] = int(m.group(3))
    return out


@dataclass
class SiteRecording:
    """Parsed contents of one Wingert per-site .tgz.

    Attributes
    ----------
    site_id : str
        Canonical site identifier derived from the cell-id prefix (NOT
        from ``meta.json[siteid]``, which can disagree — e.g. PRN018b's
        meta says ``'PRN018a'``).
    fs : int
        Bin rate (Hz). Always 100 for Wingert 2026.
    cell_ids : list of str
        Length-N list of unit identifiers in their order in ``resp.h5``.
    spike_times : dict
        ``{cell_id: np.ndarray}`` mapping each cell to its (n_spikes,)
        float64 array of spike times in seconds.
    stims : dict
        ``{stim_name: np.ndarray}`` mapping each ``STIM_<wavname>`` key
        to its (F, T) float64 gammatone-gram. Stims not presented at
        this site are absent.
    epochs : pandas.DataFrame
        The ``resp.epoch.csv`` table, columns ``['name', 'start', 'end']``
        with times in seconds.
    """

    site_id: str
    fs: int
    cell_ids: List[str]
    spike_times: Dict[str, np.ndarray]
    stims: Dict[str, np.ndarray]
    epochs: pd.DataFrame


def _resolve_site_dir(directory: str) -> str:
    """Return the directory actually containing the NEMS files.

    A NEMS recording dir holds ``*.meta.json`` directly; some extractions
    wrap everything in an extra subdir (the .tgz files in this release
    DO have a top-level ``<SITE>/`` wrapper). Auto-descend through a
    single subdir if needed.
    """
    if any(f.endswith('.meta.json') for f in os.listdir(directory)):
        return directory
    subs = [d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))]
    if len(subs) == 1:
        inner = os.path.join(directory, subs[0])
        if any(f.endswith('.meta.json') for f in os.listdir(inner)):
            return inner
    raise FileNotFoundError(
        f"No NEMS recording files (*.meta.json) found in {directory}/."
    )


def _load_signal_json(directory: str, signal_name: str) -> dict:
    """Read ``<site>.<signal>.json`` (e.g. resp, stim)."""
    cands = [f for f in os.listdir(directory)
             if f.endswith(f'.{signal_name}.json')]
    assert len(cands) == 1, (
        f"expected exactly one *.{signal_name}.json in {directory}, got {cands}"
    )
    with open(os.path.join(directory, cands[0])) as f:
        return json.load(f)


def _load_epoch_csv(directory: str, signal_name: str = 'resp') -> pd.DataFrame:
    """Read ``<site>.<signal>.epoch.csv``: ['name', 'start', 'end'] in seconds."""
    cands = [f for f in os.listdir(directory)
             if f.endswith(f'.{signal_name}.epoch.csv')]
    assert len(cands) == 1, (
        f"expected exactly one *.{signal_name}.epoch.csv in {directory}, got {cands}"
    )
    return pd.read_csv(os.path.join(directory, cands[0]))


def epoch_names_matching(epochs: pd.DataFrame, regex: str) -> List[str]:
    """Return the unique epoch names matching ``regex``, in encounter order.

    Drop-in equivalent of ``nems0.epoch.epoch_names_matching``.
    """
    pat = re.compile(regex)
    seen, names = set(), []
    for n in epochs['name']:
        if pat.match(n) and n not in seen:
            seen.add(n)
            names.append(n)
    return names


def rasterize_spike_times(spikes_s: np.ndarray, T: int, fs: int) -> np.ndarray:
    """Bin a 1-D array of spike times (in seconds) to a length-T count vector.

    Uses the NEMS reference convention: floor (not round) of ``t * fs``,
    drop out-of-range, ``np.add.at`` for the rare same-bin collisions.

    Parameters
    ----------
    spikes_s : np.ndarray
        (n_spikes,) float array of spike times in seconds.
    T : int
        Number of output time bins.
    fs : int
        Bin rate (Hz).

    Returns
    -------
    np.ndarray
        Shape (T,), dtype float32, integer-valued spike counts.
    """
    out = np.zeros(T, dtype=np.float32)
    if spikes_s.size == 0:
        return out
    idx = np.floor(spikes_s * fs).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < T)]
    np.add.at(out, idx, 1.0)
    return out


def _load_site_dir(site_dir: str) -> SiteRecording:
    """Parse an already-extracted Wingert site directory."""
    resp_meta = _load_signal_json(site_dir, 'resp')
    cell_ids = list(resp_meta['chans'])
    fs = int(resp_meta['fs'])

    # Canonical site id from the cell-id prefix (NOT meta.json[siteid], which
    # can disagree — verified for PRN018b which has meta.siteid='PRN018a').
    site_id = cell_ids[0].split('-', 1)[0] if cell_ids else _load_signal_json(
        site_dir, 'meta').get('siteid', '?')

    # spike times: one float64 dataset per cell in resp.h5
    h5_cands = [f for f in os.listdir(site_dir) if f.endswith('.resp.h5')]
    assert len(h5_cands) == 1, f"expected one *.resp.h5, got {h5_cands}"
    spike_times: Dict[str, np.ndarray] = {}
    with h5py.File(os.path.join(site_dir, h5_cands[0]), 'r') as h5:
        for cell_id in cell_ids:
            spike_times[cell_id] = h5[cell_id][...]

    # stim spectrograms: one (F, T) dataset per unique STIM_<wavname> in stim.h5
    stim_h5_cands = [f for f in os.listdir(site_dir) if f.endswith('.stim.h5')]
    assert len(stim_h5_cands) == 1, f"expected one *.stim.h5, got {stim_h5_cands}"
    stims: Dict[str, np.ndarray] = {}
    with h5py.File(os.path.join(site_dir, stim_h5_cands[0]), 'r') as h5:
        for name in h5.keys():
            stims[name] = h5[name][...]

    epochs = _load_epoch_csv(site_dir, 'resp')

    return SiteRecording(
        site_id=site_id,
        fs=fs,
        cell_ids=cell_ids,
        spike_times=spike_times,
        stims=stims,
        epochs=epochs,
    )


def load_site_recording(path: str) -> SiteRecording:
    """Parse one Wingert per-site recording from a .tgz or extracted directory.

    Parameters
    ----------
    path : str
        Either a ``.tgz`` / ``.tar.gz`` archive, or an already-extracted
        site directory containing the NEMS files (or a parent of one).

    Returns
    -------
    SiteRecording
        Dataclass with ``site_id``, ``fs``, ``cell_ids``, ``spike_times``
        (dict cell_id → array of spike-times-in-seconds), ``stims`` (dict
        stim_name → (F, T) array), and ``epochs`` (DataFrame).
    """
    if path.endswith(('.tgz', '.tar.gz')) and os.path.isfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(path, 'r:*') as tf:
                tf.extractall(tmp)
            site_dir = _resolve_site_dir(tmp)
            return _load_site_dir(site_dir)
    site_dir = _resolve_site_dir(path)
    return _load_site_dir(site_dir)
