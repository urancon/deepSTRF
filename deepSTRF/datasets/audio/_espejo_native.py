"""Native (NEMS-free) parser for Espejo (Lopez-Espejo et al. 2019) archives.

Each Espejo recording is a ``<site>.tgz`` with the standard NEMS layout:

    <site>.meta.json         rasterfs (always 100), stimfmt ('ozgf' for NAT,
                             'envelope' for VMN), cellid list, batch
    <site>.resp.h5           HDF5 dict cellid -> 1D float spike-times (s)
                             — NEMS PointProcess
    <site>.resp.json         chans (cellids), fs, segments [[0, T_total_samples]]
    <site>.resp.epoch.csv    epoch table (start, end, name) in seconds.
                             STIM_<name> rows = one row per occurrence.
    <site>.stim.h5           HDF5 dict STIM_<name> -> (F, T_stim) float
                             cochleagram — NEMS TiledSignal: unique stims
                             stored once, replayed at each occurrence.
    <site>.stim.json         (TiledSignal metadata; not actually needed)
    <site>.stim.epoch.csv    same as resp.epoch.csv

Same family as NAT4's per-site recordings (resp is PointProcess) but with
the stim stored as TiledSignal rather than NAT4's pop-recording
RasterizedSignal — so the per-stim spectrograms come straight out of
``stim.h5`` already pre-rasterized as ``(F, T_stim)`` tensors.

This module replaces:
 - ``nems0.recording.load_recording``
 - ``signal.rasterize()`` / ``signal.extract_epoch(epoch_name)``
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd


@dataclass
class EspejoSite:
    """Parsed contents of one Espejo NEMS-recording archive."""
    site_id: str                                 # 'AMT003c', 'btn144a', ...
    fs: int                                      # 100 Hz for Espejo
    stim_format: str                             # 'ozgf' (NAT) or 'envelope' (VMN)
    cellids: List[str]                           # length N_site
    spike_times: Dict[str, np.ndarray]           # cellid -> (n_spikes,) spike times in s
    stim_cochleagrams: Dict[str, np.ndarray]     # STIM_name -> (F, T_stim) cochleagram
    epochs: pd.DataFrame                         # ['start', 'end', 'name']
    duration_s: float                            # total recording length in s


def _resolve_site_dir(directory: str) -> str:
    """Return the directory actually holding ``*.meta.json``.

    Some local extractions wrap the NEMS files in an extra ``<site>/``
    subdirectory; auto-descend one level when that's the case.
    """
    if any(f.endswith(".meta.json") for f in os.listdir(directory)):
        return directory
    subs = [d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))]
    if len(subs) == 1:
        inner = os.path.join(directory, subs[0])
        if any(f.endswith(".meta.json") for f in os.listdir(inner)):
            return inner
    raise FileNotFoundError(
        f"No NEMS recording files (*.meta.json) found in {directory}."
    )


def _load_one_json(directory: str, suffix: str) -> dict:
    cands = [f for f in os.listdir(directory) if f.endswith(suffix)]
    assert len(cands) == 1, (
        f"expected exactly one *{suffix} in {directory}, got {cands}"
    )
    with open(os.path.join(directory, cands[0])) as f:
        return json.load(f)


def _load_epoch_csv(directory: str, signal_name: str = "resp") -> pd.DataFrame:
    cands = [f for f in os.listdir(directory)
             if f.endswith(f".{signal_name}.epoch.csv")]
    assert len(cands) == 1, (
        f"expected exactly one *.{signal_name}.epoch.csv in {directory}, got {cands}"
    )
    return pd.read_csv(os.path.join(directory, cands[0]))


def _load_h5_dict(directory: str, suffix: str) -> Dict[str, np.ndarray]:
    cands = [f for f in os.listdir(directory) if f.endswith(suffix)]
    assert len(cands) == 1, (
        f"expected exactly one *{suffix} in {directory}, got {cands}"
    )
    out: Dict[str, np.ndarray] = {}
    with h5py.File(os.path.join(directory, cands[0]), "r") as h5:
        for key in h5.keys():
            out[key] = h5[key][...]
    return out


def load_espejo_site(path: str) -> EspejoSite:
    """Parse one Espejo NEMS archive (``.tgz``) or pre-extracted directory.

    Accepts either ``<site>.tgz`` or a directory holding the unpacked
    ``<site>.{meta,resp,stim}.{json,h5,epoch.csv}`` files.
    """
    if path.endswith((".tgz", ".tar.gz")) and os.path.isfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(path, "r:*") as tf:
                tf.extractall(tmp)
            return load_espejo_site(tmp)

    site_dir = _resolve_site_dir(path)

    meta = _load_one_json(site_dir, ".meta.json")
    resp_meta = _load_one_json(site_dir, ".resp.json")

    fs = int(resp_meta["fs"])
    cellids = list(resp_meta["chans"])
    site_id = str(resp_meta.get("recording", ""))
    stim_format = str(meta.get("stimfmt", ""))

    segments = resp_meta.get("segments", [[0, 0]])
    duration_s = float(segments[-1][-1]) / fs

    spike_times = _load_h5_dict(site_dir, ".resp.h5")
    stim_cochleagrams = _load_h5_dict(site_dir, ".stim.h5")
    epochs = _load_epoch_csv(site_dir, "resp")

    return EspejoSite(
        site_id=site_id,
        fs=fs,
        stim_format=stim_format,
        cellids=cellids,
        spike_times=spike_times,
        stim_cochleagrams=stim_cochleagrams,
        epochs=epochs,
        duration_s=duration_s,
    )


def rasterize_pointprocess(
    spike_times_s: np.ndarray,
    start_s: float,
    end_s: float,
    bin_s: float,
) -> np.ndarray:
    """Bin spike times (seconds) into a ``(T,)`` float count array.

    Mirrors NEMS' ``PointProcess.rasterize`` for one epoch occurrence:
    ``b = int(floor((t - start_s) / bin_s))``. Spikes outside
    ``[start_s, start_s + T·bin_s)`` are silently dropped. ``T`` is
    derived from ``round((end_s - start_s) / bin_s)`` so callers that
    pass an exact integer number of bins get the expected size.
    """
    T = int(round((end_s - start_s) / bin_s))
    out = np.zeros((T,), dtype=np.float32)
    if spike_times_s.size == 0:
        return out
    rel = spike_times_s - start_s
    idx = np.floor(rel / bin_s).astype(np.int64)
    keep = (idx >= 0) & (idx < T)
    if keep.any():
        np.add.at(out, idx[keep], 1.0)
    return out


def extract_epoch_rasters(
    site: EspejoSite,
    cellid: str,
    stim_name: str,
    bin_s: float,
    T: int,
) -> np.ndarray:
    """Extract per-rep rasters for one cell × one stim epoch as ``(R, T)``.

    ``R`` is the number of occurrences of ``stim_name`` in
    ``site.epochs``. Each row is the binned spike count over the epoch
    window, right-padded with zeros if the epoch is shorter than ``T``
    and truncated if longer (rare; aligns the raster to the stim
    spectrogram's frame count).
    """
    spikes = site.spike_times.get(cellid)
    if spikes is None:
        raise KeyError(f"cellid {cellid!r} not in site {site.site_id!r}")
    rows = site.epochs[site.epochs["name"] == stim_name]
    if rows.empty:
        return np.zeros((0, T), dtype=np.float32)
    out = np.zeros((len(rows), T), dtype=np.float32)
    for r, (_, row) in enumerate(rows.iterrows()):
        start = float(row["start"])
        end = float(row["end"])
        raster = rasterize_pointprocess(spikes, start, end, bin_s)
        if raster.size >= T:
            out[r] = raster[:T]
        else:
            out[r, :raster.size] = raster
    return out


def stim_occurrence_counts(epochs: pd.DataFrame) -> Dict[str, int]:
    """Count occurrences of each ``STIM_*`` epoch name in the epoch table."""
    names = epochs.loc[epochs["name"].str.startswith("STIM_"), "name"]
    return names.value_counts().to_dict()
