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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd


@dataclass
class EspejoSite:
    """Parsed contents of one Espejo NEMS-recording archive.

    The response signal is stored in one of two NEMS conventions:

    - ``PointProcess`` (the majority): spike times in ``<site>.resp.h5``
      keyed by cellid. ``signal_type='pointprocess'``, ``spike_times``
      populated, ``resp_rasterized`` is None.

    - ``RasterizedSignal``: continuous ``(T_total, N)`` matrix in
      ``<site>.resp.csv``. ``signal_type='rasterized'``,
      ``resp_rasterized`` populated, ``spike_times`` is None. Three VMN
      sites (chn002h, eno023c, eno028f) use this form.

    ``stim_cochleagrams`` is always a dict mapping ``STIM_<name>`` to a
    ``(F, T_stim)`` array — assembled at load time regardless of whether
    the source was ``stim.h5`` (TiledSignal) or ``stim.csv`` +
    ``stim.epoch.csv`` (RasterizedSignal, sliced at the first
    occurrence of each STIM_ epoch).
    """
    site_id: str                                            # 'AMT003c', 'btn144a', ...
    fs: int                                                 # 100 Hz for Espejo
    stim_format: str                                        # 'ozgf' (NAT) or 'envelope' (VMN)
    cellids: List[str]                                      # length N_site
    signal_type: str                                        # 'pointprocess' or 'rasterized'
    spike_times: Optional[Dict[str, np.ndarray]] = None     # PointProcess case
    resp_rasterized: Optional[np.ndarray] = None            # RasterizedSignal case: (T_total, N)
    stim_cochleagrams: Dict[str, np.ndarray] = field(default_factory=dict)
    epochs: pd.DataFrame = field(default_factory=pd.DataFrame)
    duration_s: float = 0.0


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


def _load_csv_signal(directory: str, signal_name: str) -> np.ndarray:
    """Read ``<site>.<signal>.csv`` (RasterizedSignal): a ``(T_total, K)``
    float matrix, no header, NaN-encoded missing entries.
    """
    cands = [f for f in os.listdir(directory)
             if f.endswith(f".{signal_name}.csv")
             and not f.endswith(f".{signal_name}.epoch.csv")]
    assert len(cands) == 1, (
        f"expected exactly one *.{signal_name}.csv in {directory}, got {cands}"
    )
    arr = np.loadtxt(os.path.join(directory, cands[0]), delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _stim_cochleagrams_from_csv(
    stim_matrix: np.ndarray,
    epochs: pd.DataFrame,
    fs: int,
) -> Dict[str, np.ndarray]:
    """Slice a continuous ``(T_total, F)`` stim matrix into a per-STIM dict.

    Used for the RasterizedSignal stim form. The stim signal is tiled —
    each ``STIM_<name>`` epoch in the table refers to the same underlying
    spectrogram — so we take the first occurrence of each name and
    transpose into ``(F, T_stim)`` to match the HDF5 TiledSignal layout.
    """
    out: Dict[str, np.ndarray] = {}
    for _, row in epochs.iterrows():
        name = row["name"]
        if not isinstance(name, str) or not name.startswith("STIM_"):
            continue
        if name in out:
            continue
        s = int(round(float(row["start"]) * fs))
        e = int(round(float(row["end"]) * fs))
        out[name] = stim_matrix[s:e].T.copy()  # (F, T_stim)
    return out


def load_espejo_site(path: str) -> EspejoSite:
    """Parse one Espejo NEMS archive (``.tgz``) or pre-extracted directory.

    Accepts either ``<site>.tgz`` or a directory holding the unpacked
    ``<site>.{meta,resp,stim}.{json,h5,csv,epoch.csv}`` files. Detects
    whether resp + stim are PointProcess/TiledSignal (HDF5) or
    RasterizedSignal (CSV) — three VMN sites use the latter.
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

    epochs = _load_epoch_csv(site_dir, "resp")

    files_here = os.listdir(site_dir)
    has_resp_h5 = any(f.endswith(".resp.h5") for f in files_here)
    has_resp_csv = any(f.endswith(".resp.csv") and not f.endswith(".resp.epoch.csv")
                       for f in files_here)
    has_stim_h5 = any(f.endswith(".stim.h5") for f in files_here)
    has_stim_csv = any(f.endswith(".stim.csv") and not f.endswith(".stim.epoch.csv")
                       for f in files_here)

    # --- response ---
    if has_resp_h5:
        signal_type = "pointprocess"
        spike_times = _load_h5_dict(site_dir, ".resp.h5")
        resp_rasterized = None
    elif has_resp_csv:
        signal_type = "rasterized"
        spike_times = None
        resp_rasterized = _load_csv_signal(site_dir, "resp")
        assert resp_rasterized.shape[1] == len(cellids), (
            f"site {site_id}: resp.csv has {resp_rasterized.shape[1]} channels "
            f"but resp.json lists {len(cellids)} cellids"
        )
    else:
        raise FileNotFoundError(
            f"site {site_id}: neither .resp.h5 nor .resp.csv found in {site_dir}"
        )

    # --- stim cochleagrams (always returned as a dict regardless of source) ---
    if has_stim_h5:
        stim_cochleagrams = _load_h5_dict(site_dir, ".stim.h5")
    elif has_stim_csv:
        stim_matrix = _load_csv_signal(site_dir, "stim")
        stim_cochleagrams = _stim_cochleagrams_from_csv(stim_matrix, epochs, fs)
    else:
        raise FileNotFoundError(
            f"site {site_id}: neither .stim.h5 nor .stim.csv found in {site_dir}"
        )

    return EspejoSite(
        site_id=site_id,
        fs=fs,
        stim_format=stim_format,
        cellids=cellids,
        signal_type=signal_type,
        spike_times=spike_times,
        resp_rasterized=resp_rasterized,
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

    Works for both signal types — for PointProcess sites the rasterizer
    floors spike times to bins; for RasterizedSignal sites the matrix
    is already binned at fs=100 Hz and we just slice it.
    """
    rows = site.epochs[site.epochs["name"] == stim_name]
    if rows.empty:
        return np.zeros((0, T), dtype=np.float32)

    if site.signal_type == "pointprocess":
        spikes = site.spike_times.get(cellid)  # type: ignore[union-attr]
        if spikes is None:
            raise KeyError(f"cellid {cellid!r} not in site {site.site_id!r}")
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

    # rasterized signal: slice the (T_total, N) matrix at each epoch
    if cellid not in site.cellids:
        raise KeyError(f"cellid {cellid!r} not in site {site.site_id!r}")
    n = site.cellids.index(cellid)
    mat = site.resp_rasterized  # type: ignore[assignment]
    assert mat is not None
    out = np.zeros((len(rows), T), dtype=np.float32)
    samples_per_bin = bin_s * site.fs
    assert abs(samples_per_bin - round(samples_per_bin)) < 1e-9, (
        f"dt_ms={bin_s*1000} ms is not an integer multiple of the "
        f"rasterized signal's 1/fs={1000/site.fs} ms"
    )
    samples_per_bin = int(round(samples_per_bin))
    for r, (_, row) in enumerate(rows.iterrows()):
        s = int(round(float(row["start"]) * site.fs))
        e = int(round(float(row["end"]) * site.fs))
        slab = mat[s:e, n]
        if samples_per_bin != 1:
            # block-sum to coarser bins
            cut = (slab.shape[0] // samples_per_bin) * samples_per_bin
            slab = slab[:cut].reshape(-1, samples_per_bin).sum(axis=1)
        if slab.size >= T:
            out[r] = slab[:T]
        else:
            out[r, :slab.size] = slab
    return out


def stim_occurrence_counts(epochs: pd.DataFrame) -> Dict[str, int]:
    """Count occurrences of each ``STIM_*`` epoch name in the epoch table."""
    names = epochs.loc[epochs["name"].str.startswith("STIM_"), "name"]
    return names.value_counts().to_dict()
