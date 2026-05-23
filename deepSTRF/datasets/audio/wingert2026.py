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

import json
import os
import tarfile
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._wingert_native import (
    load_site_recording,
    parse_wingert_cell_id,
    rasterize_spike_times,
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
                 include_unlabeled: bool = False,
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
        include_unlabeled : bool, default False
            If True, also include the 131 cells in ``cell_list.csv``
            that lack an area label (and therefore also lack
            ``layer`` / ``depth`` / ``narrow`` / ``celltype``). These
            come from three otherwise-unrepresented PRN sessions
            (PRN010b, PRN011b, PRN020b) and have ``area=None``,
            ``layer=None``, ``depth=None``, etc. in ``nrn_meta``.
            ``goodpred`` is still populated. The default ``False``
            matches the paper's analysis cohort.
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
        if not include_unlabeled:
            # Default: drop the 131 cells with area=NaN.
            df = df[df["area"].isin(_VALID_AREAS)].reset_index(drop=True)
        if areas is not None:
            # An explicit ``area=`` filter implies labelled cohort only.
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

        # ---- map session_id → .tgz path ----
        session_to_tgz = _build_session_to_tgz_map(recordings_dir)

        # Group target cells by session for the load loop. Each session is
        # opened exactly once even when it serves multiple cell_list siteids
        # (e.g. SLJ032a's two-probe recording feeds 'SLJ032a' and 'SLJ032a-B').
        cells_by_session: Dict[str, List[int]] = {}
        for n_idx, meta in enumerate(self.nrn_meta):
            cells_by_session.setdefault(meta["session"], []).append(n_idx)

        missing_sessions = [s for s in cells_by_session if s not in session_to_tgz]
        if missing_sessions:
            raise FileNotFoundError(
                f"No .tgz found in {recordings_dir!r} for sessions: {missing_sessions}. "
                f"Re-run with download=True or check the data path."
            )

        # ---- shared sentinel: one tensor object, referenced everywhere a
        # (stim, cell) pair is missing. Without this trick the (S, N)
        # response grid balloons from ~80 MB (pointer cost) to multiple
        # GB (per-slot fresh torch.full call). See plan §H risk #1. ----
        NAN = torch.full((1, 1), float("nan"))

        self.stims = []
        self.stim_meta = []
        self.responses = []

        # Deterministic session order so concat'd / persisted instances are
        # bit-stable across runs.
        for session in tqdm(sorted(cells_by_session.keys()),
                            desc="Wingert2026 sites"):
            tgz_path = session_to_tgz[session]
            rec = load_site_recording(tgz_path)
            session_cell_idx = {
                self.nrn_meta[n]["cell_id"]: n for n in cells_by_session[session]
            }

            # Cells the .tgz contributes but the filter dropped (e.g. probe-A
            # cells when site='SLJ032a-B'): silently ignored, the rasterizer
            # never visits their spike trains.
            in_session = [c for c in rec.cell_ids if c in session_cell_idx]

            for stim_name in sorted(rec.stims.keys()):
                spec = rec.stims[stim_name]                  # (F, T_s)
                F_s, T_s = spec.shape
                assert F_s == self.F, (
                    f"unexpected F={F_s} for stim {stim_name!r} in session "
                    f"{session!r}; expected F={self.F}"
                )
                s_idx = len(self.stims)
                self.stims.append(torch.from_numpy(spec).unsqueeze(0).float())
                self.stim_meta.append({
                    "name": stim_name,
                    "subset": "val" if stim_name.startswith("STIM_00") else "est",
                    "session": session,
                })

                # Default response row: NaN sentinel everywhere.
                row: List[torch.Tensor] = [NAN] * self.N_neurons

                # Epoch rows giving R presentation windows for this stim.
                epoch_rows = rec.epochs[rec.epochs["name"] == stim_name]
                R = len(epoch_rows)
                if R == 0 or not in_session:
                    self.responses.append(row)
                    continue

                # Rasterize R repeats × T_s per cell.
                ep_starts = epoch_rows["start"].to_numpy()
                ep_ends = epoch_rows["end"].to_numpy()
                for cell_id in in_session:
                    spikes_s = rec.spike_times[cell_id]
                    reps = np.zeros((R, T_s), dtype=np.float32)
                    for r_idx in range(R):
                        s, e = ep_starts[r_idx], ep_ends[r_idx]
                        # Spikes in this presentation window, expressed
                        # relative to the window start.
                        in_win = (spikes_s >= s) & (spikes_s < e)
                        rel = spikes_s[in_win] - s
                        reps[r_idx] = rasterize_spike_times(rel, T_s, rec.fs)
                    row[session_cell_idx[cell_id]] = torch.from_numpy(reps)
                self.responses.append(row)

            del rec  # free per-site spike-time / stim memory ASAP

        # ---- per-instance global minmax on stim and on resp ----
        _normalize_minmax_inplace(self.stims, self.responses)

        # ---- subset filter (drop est / val after the global load) ----
        if subset != "all":
            keep = [i for i, m in enumerate(self.stim_meta) if m["subset"] == subset]
            self.stims = [self.stims[i] for i in keep]
            self.stim_meta = [self.stim_meta[i] for i in keep]
            self.responses = [self.responses[i] for i in keep]

        if smooth:
            self.smooth_responses(window_ms=21.0)

        self.validate()


# ---------- module-level helpers ----------

def _build_session_to_tgz_map(recordings_dir: str) -> Dict[str, str]:
    """Scan ``recordings/`` once and return ``{session_id: tgz_path}``.

    The session id is the first dash-separated segment of any cell id
    inside the .tgz's ``resp.json`` — i.e. the recording-session label
    that's invariant under the 3-/4-segment cell-id schism (SLJ032a-A-...
    and SLJ032a-B-... both belong to session ``'SLJ032a'``).

    Handles two release-side quirks:

    - Three PRN .tgz files have a basename that doesn't match the cells
      they contain (e.g. ``PRN015b_*.tgz`` holds ``PRN015a-*`` cells).
      Mapping by cell id rather than filename resolves this.
    - ``PRN018a_*.tgz`` and ``PRN018b_*.tgz`` contain identical data
      (same cells, same stims, same spike times). We keep the .tgz
      whose basename matches the session id (``PRN018a``) and drop the
      duplicate.
    """
    sessions: Dict[str, List[str]] = {}
    for fname in sorted(os.listdir(recordings_dir)):
        if not fname.endswith(".tgz"):
            continue
        tgz_path = os.path.join(recordings_dir, fname)
        # Peek at resp.json without unpacking the whole archive.
        with tarfile.open(tgz_path, "r:*") as tf:
            resp_json_member = next(
                (m for m in tf.getmembers() if m.name.endswith(".resp.json")), None,
            )
            if resp_json_member is None:
                continue
            with tf.extractfile(resp_json_member) as f:
                resp_meta = json.load(f)
        cellids = resp_meta.get("chans") or []
        if not cellids:
            continue
        session = cellids[0].split("-", 1)[0]
        sessions.setdefault(session, []).append(tgz_path)

    out: Dict[str, str] = {}
    duplicates: List[str] = []
    for session, tgzs in sessions.items():
        if len(tgzs) == 1:
            out[session] = tgzs[0]
            continue
        # Prefer the .tgz whose filename starts with the session id; if
        # several still tie, take the alphabetically first.
        preferred = sorted(
            t for t in tgzs
            if os.path.basename(t).split("_", 1)[0] == session
        )
        if preferred:
            chosen = preferred[0]
            duplicates.extend(t for t in tgzs if t != chosen)
        else:
            tgzs_sorted = sorted(tgzs)
            chosen = tgzs_sorted[0]
            duplicates.extend(tgzs_sorted[1:])
        out[session] = chosen

    if duplicates:
        warnings.warn(
            "Wingert2026: dropped {} duplicate .tgz file(s) (same session "
            "id as a kept archive): {}".format(
                len(duplicates), [os.path.basename(t) for t in duplicates]
            ),
            stacklevel=2,
        )
    return out


def _normalize_minmax_inplace(stims: List[torch.Tensor],
                              responses: List[List[torch.Tensor]]) -> None:
    """Per-instance global minmax to [0, 1] on stims and responses, in place.

    Stim normalisation: one (min, max) pair across the concatenation of
    every (1, F, T) stim tensor. Stims are NaN-free by data-paradigm
    invariant, so plain min / max suffice.

    Response normalisation: NaN-safe min / max over the (1, 1)-sentinel-
    or-real response tensors. Sentinels are skipped (their NaN values do
    not affect the global range), and the sentinel tensor itself is
    untouched. Real tensors are rescaled in place.

    No log compression is applied — the Wingert gtgrams are already
    log-compressed by the David-lab preprocessing pipeline.
    """
    # --- stims ---
    s_min = min(float(s.min()) for s in stims)
    s_max = max(float(s.max()) for s in stims)
    if s_max > s_min:
        for s in stims:
            s.sub_(s_min).div_(s_max - s_min)

    # --- responses ---
    real: List[torch.Tensor] = []
    for row in responses:
        for t in row:
            # The (1, 1) NaN sentinel is identified by shape AND by being a
            # shared reference (object identity is not checked here because
            # the load loop is the only place that assigns the sentinel and
            # we trust it). Real tensors have shape (R, T_s) with T_s >= 2.
            if t.numel() > 1:
                real.append(t)
    if not real:
        return
    r_min = float(min(t.min() for t in real))
    r_max = float(max(t.max() for t in real))
    if r_max > r_min:
        for t in real:
            t.sub_(r_min).div_(r_max - r_min)


def _make_nrn_meta(row: pd.Series) -> dict:
    """Build the per-neuron metadata dict from one row of ``cell_list.csv``.

    Pulls only the fields the public deepSTRF API exposes; published
    CNN / LN / subspace prediction-correlation columns are intentionally
    omitted. NaN-valued fields become ``None`` (Python's standard
    missing-data sentinel) — relevant for the 131 unlabeled cells when
    ``include_unlabeled=True`` is in play.
    """
    cell_id = str(row["cellid"])
    parsed = parse_wingert_cell_id(cell_id)
    return {
        "cell_id": cell_id,
        "site": str(row["siteid"]),
        "session": cell_id.split("-", 1)[0],
        "area": str(row["area"]) if not pd.isna(row["area"]) else None,
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
