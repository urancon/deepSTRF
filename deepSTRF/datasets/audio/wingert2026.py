"""Wingert 2026 — natural-sound responses from ferret auditory cortex.

Reference
---------
Wingert JC, Parida S, Norman-Haignere SV, David SV (2026).
"Convolutional neural network models describe the encoding subspace of local
circuits in auditory cortex." *Nature Neuroscience*.
https://doi.org/10.1038/s41593-026-02216-0

Data: Zenodo record 18331549 (open access). Single-unit Kilosort-sorted
spikes from primary (A1) and non-primary (PEG) ferret auditory cortex,
plus less-curated AC and HC subsets, recorded with high-density silicon
probes (64-ch FHC) and Neuropixels during passive presentation of
natural-sound sequences.

This module's loader is NEMS-free — see ``_wingert_native.py``.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._wingert_native import (
    parse_wingert_cell_id,
)
from deepSTRF.utils.data_download import (
    default_cache_dir,
    unzip,
    zenodo_download,
)


# Public Zenodo record. https://doi.org/10.5281/zenodo.18331549
WINGERT_ZENODO_RECORD = 18331549

# Areas as labelled in cell_list.csv. The paper headlines A1 / PEG but
# the released csv also tags AC (217 cells) and HC (37 cells), plus 131
# cells with no area label.
_VALID_AREAS = ("A1", "PEG", "AC", "HC")


def download_wingert2026(dest: Optional[str] = None) -> str:
    """Download the Wingert 2026 release from Zenodo into ``dest``.

    Fetches ``recordings.zip`` (per-site .tgz archives, the only file the
    loader actually needs) and ``cell_list.csv`` (per-cell metadata).
    Does NOT fetch the much larger ``wav.zip`` (raw waveforms, not used
    by the spectrogram-only loader) or ``models.zip`` (published CNN /
    LN / subspace fits, not used by deepSTRF).

    Idempotent — skips files / dirs that already exist.

    Parameters
    ----------
    dest : str, optional
        Defaults to ``default_cache_dir('Wingert2026')`` (overridable via
        ``$DEEPSTRF_DATA_DIR``).

    Returns
    -------
    str
        The destination directory.
    """
    dest_path = str(default_cache_dir("Wingert2026") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)

    csv_path = os.path.join(dest_path, "cell_list.csv")
    if not os.path.exists(csv_path):
        zenodo_download(WINGERT_ZENODO_RECORD, "cell_list.csv", csv_path)

    recordings_dir = os.path.join(dest_path, "recordings")
    if not (os.path.isdir(recordings_dir)
            and sum(1 for f in os.listdir(recordings_dir) if f.endswith(".tgz")) >= 60):
        zip_path = os.path.join(dest_path, "recordings.zip")
        if not os.path.exists(zip_path):
            zenodo_download(WINGERT_ZENODO_RECORD, "recordings.zip", zip_path)
        unzip(zip_path, dest_path)

    return dest_path


def _coerce_to_list(value: Union[None, str, Iterable[str]]) -> Optional[List[str]]:
    """Accept ``None``, a single str, or an iterable of str — return list or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


class Wingert2026Dataset(AudioNeuralDataset):
    """PyTorch dataset for Wingert et al. 2026 (Nat Neurosci).

    A high-density ferret auditory-cortex recording library: 2 128 A1 +
    746 PEG + 217 AC + 37 HC single units across 67 recording sites (68
    cell_list ``siteid`` groups, since SLJ032a's two-probe recording
    contributes two siteids — A-probe ``'SLJ032a'`` and B-probe
    ``'SLJ032a-B'``). Stimuli are 20–22 s sequences of crossfaded natural
    sound segments (Audioset Core 3 Complete + Pro Sound Effects), each
    site presents ~100 estimation stims (single-rep) and 1–6 test stims
    (R ranging from 5 to 30 across sites).

    The release ships gammatone-gram spectrograms ("cochleagrams")
    precomputed at fs = 100 Hz (10 ms bins), F = 32 log-spaced bands from
    200 Hz to 20 kHz, log-compressed by the David-lab gtgram pipeline.
    Two stim-duration cohorts coexist in the released data:

    - 47 sites at ``T = 2000`` bins (20 s, no silence flanks);
    - 21 sites at ``T = 2200`` bins (22 s = 1 s pre + 20 s sound + 1 s
      post).

    The deepSTRF data paradigm supports ragged T natively — the per-stim
    tensor keeps its own time length and collate zero-pads on the right.

    The loader reads the published archive directly with native CSV /
    JSON / HDF5 parsers — no ``nems0`` dependency. Data are open access
    at https://doi.org/10.5281/zenodo.18331549 and auto-fetched by
    ``Wingert2026Dataset(download=True)``.

    Notes
    -----
    Follows the standard deepSTRF data paradigm (see
    ``docs/_source/md/data_paradigm.md``). Wingert-specific metadata:

    - ``stim_meta`` dicts hold ``name`` (e.g. ``'STIM_seq0032.wav'``),
      ``subset`` (``'est'`` for ``STIM_seq*``, ``'val'`` for ``STIM_00*``),
      and ``site`` (the cell_list-canonical site id this stim was presented
      at). The same source wav can appear under multiple ``(name, site)``
      pairs because each session re-rasterizes its own copy and the two
      duration cohorts produce different-shape tensors.
    - ``nrn_meta`` dicts hold ``cell_id``, ``site`` (from
      ``cell_list.csv``, authoritative), ``area``, ``layer``, ``depth``,
      ``narrow``, ``celltype``, ``sw``, ``goodpred``, and the parsed
      ``animal`` / ``electrode`` / ``unit_in_electrode`` components.

    The published cell counts hold whenever the cohort uses the standard
    A1 + PEG filter; AC and HC are exposed but documented as less-curated.

    References
    ----------
    Wingert et al. (2026). "Convolutional neural network models describe
    the encoding subspace of local circuits in auditory cortex."
    *Nature Neuroscience*. https://doi.org/10.1038/s41593-026-02216-0
    """

    def __init__(self,
                 path: Optional[str] = None,
                 area: Union[None, str, Iterable[str]] = None,
                 site: Union[None, str, Iterable[str]] = None,
                 dt_ms: float = 10.0,
                 subset: str = "all",
                 smooth: bool = False,
                 download: bool = False,
                 _enumerate_only: bool = False):
        """
        Parameters
        ----------
        path : str, optional
            Path to the unpacked dataset root (the directory containing
            ``recordings/`` and ``cell_list.csv``). Defaults to
            ``default_cache_dir('Wingert2026')``.
        area : str or iterable of str, optional
            Restrict to one or more cortical areas: any of
            ``'A1'``, ``'PEG'``, ``'AC'``, ``'HC'``. ``None`` (default)
            loads every area-labelled cell; cells with ``area=NaN`` in
            ``cell_list.csv`` (131 cells, presumably sort-failed) are
            always excluded.
        site : str or iterable of str, optional
            Restrict to one or more cell_list ``siteid`` values (e.g.
            ``'CLT027c'``, ``'SLJ032a-B'``, ``'PRN018a'``). ``None``
            (default) loads every site that survives the ``area`` filter.
        dt_ms : float, default 10.0
            Time-bin width in ms. Currently must equal 10.0 — the
            published gammatone-gram is precomputed at fs = 100 and a
            future down-binning helper is out of v1 scope.
        subset : {'all', 'est', 'val'}, default 'all'
            ``'est'`` keeps only the single-rep ``STIM_seq*`` estimation
            stims; ``'val'`` keeps only the high-rep ``STIM_00*`` test
            stims. The bidirectional select rule applies — cells whose
            site did not present any retained stim are masked out of
            ``__getitem__`` automatically.
        smooth : bool, default False
            If True, smooth PSTHs with a 21 ms Hanning window via
            ``self.smooth_responses(window_ms=21.0)``.
        download : bool, default False
            If True, fetch ``recordings.zip`` + ``cell_list.csv`` from
            Zenodo (record ``18331549``) if missing. The 8 GB
            ``wav.zip`` is NOT fetched (the loader uses the precomputed
            gtgrams in ``stim.h5``).
        _enumerate_only : bool, default False
            Internal flag for tests: populate ``nrn_meta`` and
            ``N_neurons`` only, skip the (~1 minute) per-site .tgz read
            pass. Subclasses of this loader should not rely on it.
        """
        # ---- input validation ----
        areas = _coerce_to_list(area)
        sites = _coerce_to_list(site)
        if areas is not None:
            for a in areas:
                assert a in _VALID_AREAS, (
                    f"unknown area {a!r}; valid: {_VALID_AREAS} (or None for all)"
                )
        assert subset in ("all", "est", "val"), \
            f"subset must be 'all', 'est', or 'val' (got {subset!r})"
        assert dt_ms == 10.0, (
            f"Wingert 2026 gammatone-grams are precomputed at dt=10 ms; "
            f"got dt_ms={dt_ms}. Re-binning is out of v1 scope."
        )

        # ---- resolve dataset root ----
        if download:
            path = download_wingert2026(path)
        elif path is None:
            path = str(default_cache_dir("Wingert2026"))
        cell_list_path = os.path.join(path, "cell_list.csv")
        recordings_dir = os.path.join(path, "recordings")
        assert os.path.exists(cell_list_path), (
            f"cell_list.csv not found under {path!r}. Pass download=True or "
            f"point `path=` at the unzipped Zenodo record."
        )
        assert os.path.isdir(recordings_dir), (
            f"recordings/ subdirectory not found under {path!r}. Pass "
            f"download=True or point `path=` at the unzipped Zenodo record."
        )

        super().__init__(path, dt_ms)
        self.species = "ferret"
        self.F = 32
        self.subset = subset

        # ---- enumerate cells from cell_list.csv (the canonical curated list) ----
        df = pd.read_csv(cell_list_path)
        # Drop cells with no area label (131 cells, sort-failed).
        df = df[df["area"].isin(_VALID_AREAS)].reset_index(drop=True)
        if areas is not None:
            df = df[df["area"].isin(areas)].reset_index(drop=True)
        if sites is not None:
            df = df[df["siteid"].isin(sites)].reset_index(drop=True)
            # Catch typos: every requested site must exist in cell_list.
            missing = set(sites) - set(df["siteid"])
            assert not missing, (
                f"site(s) not in cell_list.csv (after area filter): {sorted(missing)}"
            )
        if len(df) == 0:
            raise ValueError(
                f"No cells match the filter area={area!r}, site={site!r}. "
                f"Try a different combination."
            )

        self._cell_list = df  # retained for the Phase 3 loader
        self.nrn_meta = [
            _make_nrn_meta(row) for _, row in df.iterrows()
        ]
        self.N_neurons = len(self.nrn_meta)

        if _enumerate_only:
            # Phase-2 path: skip the heavy .tgz read. self.stims / .stim_meta /
            # .responses remain empty — only the neuron-side surface is
            # populated. self.validate() would fail (S == 0); callers know.
            return

        # ---- Phase 3: stim + response loading + normalisation + subset filter ----
        raise NotImplementedError(
            "Wingert2026Dataset full load is not yet implemented (Phase 3). "
            "Use _enumerate_only=True for now."
        )


def _make_nrn_meta(row: pd.Series) -> dict:
    """Build the per-neuron metadata dict from one row of ``cell_list.csv``.

    Pulls only the fields the public deepSTRF API exposes; published
    CNN / LN / subspace prediction-correlation columns are intentionally
    omitted.
    """
    cell_id = str(row["cellid"])
    parsed = parse_wingert_cell_id(cell_id)
    return {
        "cell_id": cell_id,
        "site": str(row["siteid"]),
        "area": str(row["area"]),
        # 'layer' is a string in the source csv (e.g. '56', '1-3'); keep as str.
        "layer": str(row["layer"]) if not pd.isna(row["layer"]) else None,
        "depth": float(row["depth"]) if not pd.isna(row["depth"]) else None,
        "narrow": (bool(row["narrow"]) if not pd.isna(row["narrow"]) else None),
        "celltype": (str(row["celltype"]) if not pd.isna(row["celltype"]) else None),
        "sw": float(row["sw"]) if not pd.isna(row["sw"]) else None,
        "goodpred": bool(row["goodpred"]),
        "animal": parsed["animal"],
        "electrode": parsed["electrode"],
        "unit_in_electrode": parsed["unit_in_electrode"],
    }
