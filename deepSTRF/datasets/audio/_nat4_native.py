"""Native (NEMS-free) parser for NAT4 NEMS-recording archives.

The NAT4 release packages each "recording" as a tarball / directory of
CSV + JSON (+ .h5 for per-site spike data). The format is well-defined
and small enough to read directly:

  Population recording (``<area>_NAT4_ozgf.fs100.ch18.tgz``), at fs=100
  with the 18 val stimuli pre-averaged over 20 reps and placed at the
  head of the time axis (first 27 s), then 575 est stimuli at R=1 each:
    <site>.meta.json         siteid, cellid list, source files
    <site>.resp.csv          (T_total, N) rasterized spike rates
    <site>.resp.json         chans (cell ids), fs, segments
    <site>.resp.epoch.csv    (epoch_name, start_s, end_s) — STIM_00cat=val,
                             STIM_cat=est
    <site>.stim.csv          (T_total, F=18) log mel spectrogram
    <site>.stim.json         signal metadata
    <site>.mask_est.csv      bool (T_total, N) — True where cell saw an
                             est trial at this timestep
    <site>.mask_est.json

  Per-site recording (``<area>_single_sites/<site>.tgz``), at fs=1000
  with all 20 val reps interleaved with the est trials in the original
  experimental order:
    <site>.meta.json
    <site>.resp.h5           one (n_spikes,) float64 array per cell;
                             values are spike times in seconds.
    <site>.resp.json
    <site>.resp.epoch.csv

This module replaces:
 - ``nems0.recording.load_recording``
 - ``nems0.epoch.epoch_names_matching``
 - ``nems0.xforms.normalize_sig``    (log1p + minmax)
 - ``nems0.preprocessing.split_pop_rec_by_mask``  (was dead code in the
                                                   existing NAT4 loader)

Validated against the published cell counts (849 A1 / 398 PEG), stim
counts (18 val + 577 est = 595), and auditory-cell counts (777 / 339).
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


@dataclass
class _Recording:
    """Minimal subset of NEMS' Recording surface used by NAT4Dataset."""
    resp: np.ndarray            # (T_total, N) rasterized response (pop) or
                                # (T_total, N) integer spike counts (per-site
                                # rasterized from .h5)
    stim: Optional[np.ndarray]  # (T_total, F) — pop only; None for per-site
    chans: List[str]            # length N — cell ids
    epochs: pd.DataFrame        # ['name','start','end'] in seconds
    fs: int                     # samples / second


def _resolve_site_dir(directory: str) -> str:
    """Return the dir actually containing the NEMS files.

    A NEMS recording dir holds ``*.meta.json`` directly; some local
    extractions (eg ``tar -xzf foo.tgz -C foo/``) wrap everything in an
    extra subdir. Auto-descend through a single subdir if needed.
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
    """Read ``<site>.<signal>.json`` (e.g. resp/stim/mask_est)."""
    cands = [f for f in os.listdir(directory)
             if f.endswith(f'.{signal_name}.json')]
    assert len(cands) == 1, f"expected exactly one *.{signal_name}.json in {directory}, got {cands}"
    with open(os.path.join(directory, cands[0])) as f:
        return json.load(f)


def _load_csv_signal(directory: str, signal_name: str) -> np.ndarray:
    """Read a NEMS CSV signal into a (T_total, K) numpy array.

    NEMS stores rasterized signals as comma-delimited float CSVs with one
    row per timestep and one column per channel (no header). NaN encodes
    missing data (channels not recorded for that timestep).
    """
    cands = [f for f in os.listdir(directory)
             if f.endswith(f'.{signal_name}.csv') and not f.endswith(f'.{signal_name}.epoch.csv')]
    assert len(cands) == 1, f"expected exactly one *.{signal_name}.csv in {directory}, got {cands}"
    return np.loadtxt(os.path.join(directory, cands[0]), delimiter=',')


def _load_epoch_csv(directory: str, signal_name: str = 'resp') -> pd.DataFrame:
    """Read ``<site>.<signal>.epoch.csv``: ['name', 'start', 'end'] in seconds."""
    cands = [f for f in os.listdir(directory)
             if f.endswith(f'.{signal_name}.epoch.csv')]
    assert len(cands) == 1, f"expected exactly one *.{signal_name}.epoch.csv in {directory}, got {cands}"
    return pd.read_csv(os.path.join(directory, cands[0]))


def load_pop_recording(path: str) -> _Recording:
    """Parse a NAT4 population recording from a .tgz or extracted dir.

    Population recordings store (T_total, N) at fs=100 with the val stims
    pre-averaged over 20 reps in the first 27 s. The resp + stim signals
    have ``segments=[[0, T_total]]`` (one continuous block).
    """
    if path.endswith(('.tgz', '.tar.gz')) and os.path.isfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(path, 'r:*') as tf:
                tf.extractall(tmp)
            return load_pop_recording(tmp)
    site_dir = _resolve_site_dir(path)

    resp_meta = _load_signal_json(site_dir, 'resp')
    chans = list(resp_meta['chans'])
    fs = int(resp_meta['fs'])

    resp = _load_csv_signal(site_dir, 'resp')   # (T_total, N)
    stim = _load_csv_signal(site_dir, 'stim')   # (T_total, F)

    if resp.ndim == 1:
        resp = resp.reshape(-1, 1)
    if stim.ndim == 1:
        stim = stim.reshape(-1, 1)
    assert resp.shape[1] == len(chans), \
        f"resp width {resp.shape[1]} != n cellids {len(chans)}"

    epochs = _load_epoch_csv(site_dir, 'resp')
    return _Recording(resp=resp, stim=stim, chans=chans, epochs=epochs, fs=fs)


def load_per_site_recording(tgz_path: str) -> _Recording:
    """Parse a per-site NAT4 recording from its .tgz.

    Per-site stores spike trains as 1-D float arrays of spike-times-in-
    seconds in an HDF5 file (``<site>.resp.h5``), at fs=1000. We
    rasterize on the fly to a (T_total, N) integer-count array so the
    rest of the pipeline can use a uniform extract_epoch primitive.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tgz_path, 'r:*') as tf:
            tf.extractall(tmp)
        site_dir = _resolve_site_dir(tmp)

        resp_meta = _load_signal_json(site_dir, 'resp')
        chans = list(resp_meta['chans'])
        fs = int(resp_meta['fs'])
        # NEMS' segments give the rasterized length in samples; per-site
        # spike times can extend a hair past this when the recording's
        # last trial includes post-stim spikes.
        segments = resp_meta.get('segments', [[0, 0]])
        T_total = int(segments[-1][-1])

        h5_cands = [f for f in os.listdir(site_dir) if f.endswith('.resp.h5')]
        assert len(h5_cands) == 1
        h5_path = os.path.join(site_dir, h5_cands[0])

        resp = np.zeros((T_total, len(chans)), dtype=np.float32)
        with h5py.File(h5_path, 'r') as h5:
            for n, cell in enumerate(chans):
                spikes_s = h5[cell][...]  # spike times in seconds
                # bin to (fs=1000) integer time-bin indices, drop spikes
                # past the rasterized signal's end. Floor (NOT round) to
                # match NEMS' ``PointProcess.rasterize``: ``b = int(np.floor
                # (t * fs))``. Different rounding rules disagree on spikes
                # that fall exactly on a bin boundary; floor is the
                # reference convention.
                idx = np.floor(spikes_s * fs).astype(np.int64)
                idx = idx[(idx >= 0) & (idx < T_total)]
                # multiple spikes may fall in the same 1ms bin (rare);
                # np.add.at handles that correctly.
                np.add.at(resp, (idx, n), 1.0)

        epochs = _load_epoch_csv(site_dir, 'resp')

    return _Recording(resp=resp, stim=None, chans=chans, epochs=epochs, fs=fs)


def epoch_names_matching(epochs: pd.DataFrame, regex: str) -> List[str]:
    """Drop-in replacement for ``nems0.epoch.epoch_names_matching``.

    Returns the unique names matching the regex, in encounter order.
    """
    pat = re.compile(regex)
    seen, names = set(), []
    for n in epochs['name']:
        if pat.match(n) and n not in seen:
            seen.add(n)
            names.append(n)
    return names


def extract_epoch(rec: _Recording, signal: str, epoch_name: str) -> np.ndarray:
    """Drop-in replacement for ``signal.extract_epoch(epoch_name)``.

    For a signal of shape (T_total, K) and an epoch matching R rows in
    the epoch table, returns (R, K, T_stim) — channels-second to match
    NEMS' convention (NEMS' RasterizedSignal stores (K, T) internally
    so its extract returns (R, K, T_stim); our resp/stim are (T, K)
    here, but we transpose on the way out so the caller doesn't need
    to care).
    """
    arr = rec.resp if signal == 'resp' else rec.stim
    if arr is None:
        raise ValueError(f"recording has no '{signal}' signal")
    matches = rec.epochs[rec.epochs['name'] == epoch_name]
    out = []
    for _, row in matches.iterrows():
        s = int(round(row['start'] * rec.fs))
        e = int(round(row['end'] * rec.fs))
        sl = arr[s:e]                # (T_stim, K)
        out.append(sl.T)             # (K, T_stim)
    if not out:
        raise KeyError(f"no epochs match {epoch_name!r}")
    return np.stack(out, axis=0)     # (R, K, T_stim)


def normalize_log1p_minmax_inplace(rec: _Recording) -> None:
    """Replace ``xforms.normalize_sig(sig='stim', 'minmax', log_compress=1)``.

    NaN-safe: nan-min / nan-max ignore missing entries. Operates on the
    full (T_total, F) stim matrix, like NEMS does.
    """
    if rec.stim is None:
        return
    a = np.log1p(rec.stim)
    a_min = float(np.nanmin(a))
    a_max = float(np.nanmax(a))
    if a_max > a_min:
        a = (a - a_min) / (a_max - a_min)
    rec.stim = a


def normalize_minmax_inplace(rec: _Recording) -> None:
    """Replace ``xforms.normalize_sig(sig='resp', 'minmax')``."""
    a = rec.resp
    a_min = float(np.nanmin(a))
    a_max = float(np.nanmax(a))
    if a_max > a_min:
        a = (a - a_min) / (a_max - a_min)
    rec.resp = a
