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
import torchaudio
from tqdm.auto import tqdm

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._wingert_native import (
    load_site_recording,
    parse_wingert_cell_id,
    rasterize_spike_times,
)
from deepSTRF.utils.audio_io import load_wav
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


def download_wingert2026(dest: Optional[str] = None, wav: bool = False) -> str:
    """Download the Wingert 2026 release from Zenodo into ``dest``.

    Fetches ``recordings.zip`` (~4.35 GB of per-site .tgz archives, the
    only large file the spectrogram loader needs) and ``cell_list.csv``
    (~5.4 MB of per-cell metadata). Does NOT fetch ``models.zip``
    (published CNN / LN / subspace fits, not used by deepSTRF).

    Idempotent — skips files / dirs that already exist.

    Parameters
    ----------
    dest : str, optional
        Defaults to ``default_cache_dir('Wingert2026')`` (overridable via
        ``$DEEPSTRF_DATA_DIR``).
    wav : bool, default False
        If True, also fetch and unpack ``wav.zip`` (~3.7 GB of source
        waveforms, 44.1 kHz) into ``<dest>/wav/`` for the raw-waveform
        branch (``Wingert2026Dataset(return_waveform=True)``). The
        spectrogram-mode loader does not need it.

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

    if wav:
        wav_dir = os.path.join(dest_path, "wav")
        if not os.path.isdir(wav_dir):
            wav_zip = os.path.join(dest_path, "wav.zip")
            if not os.path.exists(wav_zip):
                zenodo_download(WINGERT_ZENODO_RECORD, "wav.zip", wav_zip)
            unzip(wav_zip, dest_path)

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
    200 Hz to 20 kHz. The values in ``stim.h5`` are the **raw (linear)**
    gammatone-gram; the loader reproduces the paper's preprocessing on
    top of them — log compression ``log(10·x + 1)`` then per-band minmax
    to ``[0, 1]`` (see ``log_compress`` argument). Responses are
    per-neuron minmax-normalised. This matches
    ``aud_subspace_fit_demo.ipynb`` (NEMS ``log_compress`` +
    ``normalize('minmax')``) to float32 precision.
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
                 log_compress: bool = True,
                 log_offset: float = -1.0,
                 download: bool = False,
                 include_unlabeled: bool = False,
                 return_waveform: bool = False,
                 audio_fs: int = 44100,
                 prestim_ms: float = 1000.0,
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
        log_compress : bool, default True
            If True, apply the David-lab log compression
            ``log((x + d) / d)`` with ``d = 10**log_offset`` to the raw
            (linear) gammatone-gram before normalisation, reproducing the
            ``nems.preprocessing.normalization.log_compress`` step in the
            paper's pipeline. Set False to feed the raw linear gtgram.
        log_offset : float, default -1.0
            Offset exponent for ``log_compress`` (``d = 10**log_offset``).
            The paper uses ``-1`` (i.e. ``d = 0.1``, so the transform is
            ``log(10·x + 1)``). Ignored when ``log_compress=False``.
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
        return_waveform : bool, default False
            If True, each stimulus is the raw mono waveform ``(1, T_audio)``
            at ``audio_fs`` Hz instead of the precomputed gammatone-gram. The
            source ``seq*.wav`` files (44.1 kHz) are read from ``<path>/wav/``
            and inset at the recording's ``prestim_ms`` pre-silence offset
            inside the trial window, then grid-locked to
            ``T_audio = T_neural * hop`` (``hop = audio_fs * dt_ms / 1000``).
            Feed it through a model's ``wav2spec`` slot (e.g.
            ``CausalGammatone`` to reproduce the native front-end). Pass
            ``download=True`` to also fetch ``wav.zip`` from Zenodo.
        audio_fs : int, default 44100
            Audio sample rate for ``return_waveform=True``. The default
            44.1 kHz is the native rate of the source wavs and gives an exact
            integer ``hop = 441`` at ``dt_ms = 10`` (no resampling). Choose any
            rate making ``audio_fs * dt_ms / 1000`` an integer. Ignored unless
            ``return_waveform=True``.
        prestim_ms : float, default 1000.0
            Pre-stimulus silence (ms) before the sound onset in the trial
            window, used only in ``return_waveform=True`` to inset the wav so
            it aligns with the gammatone-gram frames (= response bins). The
            default 1000 ms (= 100 bins at dt=10 ms) was recovered empirically
            and is constant across all sites (the gtgram's leading silence is
            not in the epoch table). Ignored unless ``return_waveform=True``.
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
            path = download_wingert2026(path, wav=return_waveform)
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
        self.hearing_range_hz = (200.0, 40000.0)   # ferret (informational)

        # Raw-waveform input mode (opt-in). The native stim is the precomputed
        # gammatone-gram; here we instead hand out the source waveform and let a
        # model's wav2spec slot build the spectrogram (strictly causally). The
        # gtgram embeds the 17.79 s sound after a fixed 1 s pre-silence; that
        # offset is not in the epoch table, so we inset the wav at prestim_ms
        # (empirically constant across sites) and grid-lock to T_neural * hop.
        self.return_waveform = bool(return_waveform)
        self.audio_fs = int(audio_fs) if return_waveform else None
        if self.return_waveform:
            self._wav_dir = os.path.join(path, "wav")
            if not os.path.isdir(self._wav_dir):
                raise FileNotFoundError(
                    f"Wingert waveform mode needs the source wavs at "
                    f"{self._wav_dir!r}. Pass download=True to fetch wav.zip from "
                    f"Zenodo, or unpack it manually."
                )
            self._pre_samples = int(round(prestim_ms / 1000.0 * self.audio_fs))
            # STIM_<seqfile> -> on-disk filename (case-insensitive, robust to
            # any stray case mismatch between epoch names and files on disk).
            self._wav_index = {fn.lower(): fn for fn in os.listdir(self._wav_dir)
                               if fn.lower().endswith(".wav")}

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
                if self.return_waveform:
                    self.stims.append(self._load_stim_waveform(stim_name, T_s))
                else:
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
                #
                # Convention matches NEMS0's PointProcess.rasterize ->
                # extract_epoch pipeline exactly: each spike's absolute
                # bin is ``floor(t * fs)``, and the in-epoch bin is
                # ``floor(t * fs) - round(epoch_start * fs)``. Computing
                # ``floor((t - epoch_start) * fs)`` would also be sensible
                # but disagrees with NEMS0 by ±1 bin at epoch boundaries
                # whenever ``epoch_start * fs`` is not an integer (which
                # is the common case in this release -- epoch starts come
                # from BAPHY trial-onset timestamps, not bin-aligned). The
                # absolute-floor convention preserves bit-equivalence with
                # the published David-lab pipeline.
                ep_starts = epoch_rows["start"].to_numpy()
                ep_ends = epoch_rows["end"].to_numpy()
                for cell_id in in_session:
                    spikes_s = rec.spike_times[cell_id]
                    reps = np.zeros((R, T_s), dtype=np.float32)
                    for r_idx in range(R):
                        s, e = ep_starts[r_idx], ep_ends[r_idx]
                        start_bin = int(round(s * rec.fs))
                        in_win = (spikes_s >= s) & (spikes_s < e)
                        abs_bin = np.floor(spikes_s[in_win] * rec.fs).astype(np.int64)
                        rel_bin = abs_bin - start_bin
                        rel_bin = rel_bin[(rel_bin >= 0) & (rel_bin < T_s)]
                        if rel_bin.size:
                            np.add.at(reps[r_idx], rel_bin, 1.0)
                    row[session_cell_idx[cell_id]] = torch.from_numpy(reps)
                self.responses.append(row)

            del rec  # free per-site spike-time / stim memory ASAP

        # ---- preprocessing: log-compress + per-channel minmax ----
        # Reproduces the paper's pipeline (see aud_subspace_fit_demo.ipynb):
        #   stim: rasterize -> log_compress -> normalize('minmax')
        #   resp: rasterize -> normalize('minmax')
        # where NEMS' 'minmax' is PER-CHANNEL (per-band for stim, per-neuron
        # for resp), not global.
        _preprocess_inplace(
            self.stims, self.responses,
            log_compress=log_compress, log_offset=log_offset,
            normalize_stims=not self.return_waveform,
        )

        # ---- subset filter (drop est / val after the global load) ----
        if subset != "all":
            keep = [i for i, m in enumerate(self.stim_meta) if m["subset"] == subset]
            self.stims = [self.stims[i] for i in keep]
            self.stim_meta = [self.stim_meta[i] for i in keep]
            self.responses = [self.responses[i] for i in keep]

        if smooth:
            self.smooth_responses(window_ms=21.0)

        self.validate()

    def _load_stim_waveform(self, stim_name: str, T_neural: int) -> torch.Tensor:
        """Reconstruct a stim's ``(1, T_audio)`` waveform from its source .wav.

        The epoch name is ``STIM_<seqfile>`` (e.g. ``STIM_seq0032.wav`` →
        ``seq0032.wav``, ``STIM_00seq1.wav`` → ``00seq1.wav``). The source wav
        holds only the ~17.79 s sound; the gammatone-gram embeds it after a
        fixed pre-silence (``self._pre_samples``), so we zero-pad to that
        offset and crop / pad to exactly ``T_neural * hop`` samples (grid lock
        C1) so audio sample ``j`` maps to response bin ``j // hop``.
        """
        fname = stim_name[len("STIM_"):] if stim_name.startswith("STIM_") else stim_name
        resolved = self._wav_index.get(fname.lower())
        if resolved is None:
            raise FileNotFoundError(
                f"Wingert waveform mode: no source wav for epoch {stim_name!r} in "
                f"{self._wav_dir!r}. Pass download=True to fetch wav.zip from Zenodo."
            )
        w, sr = load_wav(os.path.join(self._wav_dir, resolved))   # (C, T)
        if w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)                              # mono
        if sr != self.audio_fs:
            w = torchaudio.functional.resample(w, sr, self.audio_fs)
        T_audio = T_neural * self.hop
        full = torch.zeros(1, T_audio)
        seg = w[0, : max(0, T_audio - self._pre_samples)]
        full[0, self._pre_samples: self._pre_samples + seg.shape[0]] = seg
        return full.contiguous().float()


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


def _log_compress(x: torch.Tensor, offset: float) -> torch.Tensor:
    """Port of ``nems.preprocessing.normalization.log_compress``.

    Returns ``log((x + d) / d)`` with ``d = 10**offset``. The paper uses
    ``offset = -1`` → ``d = 0.1`` → ``log(10·x + 1)``. NEMS softens
    extreme offsets (``|offset| > 2``) by a factor of 50; we replicate
    that branch for exactness though the default never triggers it.
    """
    inflect = 2.0
    adj = offset
    if offset > inflect:
        adj = inflect + (offset - inflect) / 50.0
    elif offset < -inflect:
        adj = -inflect + (offset + inflect) / 50.0
    d = 10.0 ** adj
    return torch.log((x + d) / d)


def _preprocess_inplace(stims: List[torch.Tensor],
                        responses: List[List[torch.Tensor]],
                        *,
                        log_compress: bool = True,
                        log_offset: float = -1.0,
                        normalize_stims: bool = True) -> None:
    """Reproduce the paper's stim/resp preprocessing in place.

    Mirrors ``aud_subspace_fit_demo.ipynb`` exactly:

    - **stim** — optional ``log_compress`` of the raw (linear) gtgram,
      then **per-band** minmax to ``[0, 1]``. The per-band min/max is
      taken across the concatenation of every stim (all of est+val),
      matching NEMS' ``RasterizedSignal.normalize('minmax')`` which
      computes statistics per channel over the full time axis. NEMS also
      forces post-norm values ``< 1e-6`` to exactly ``0`` ("quiet" stim →
      true zero); we replicate that.
    - **resp** — **per-neuron** minmax to ``[0, 1]``, statistics taken
      across all repeats and all stims for that neuron. The ``(1, 1)``
      NaN sentinels (shared object) are skipped and left untouched.

    Per-channel (not global) is the deliberate NEMS choice — the global
    branch is commented out in ``nems0.signal._normalize_data``. For the
    response, per-neuron vs global rescaling is invariant under
    correlation-based metrics (cc / cc_norm), but per-neuron balances the
    per-cell contribution to an MSE training loss.
    """
    if not stims:
        return

    # ---- STIM: log compression + per-band minmax (gtgram mode only) ----
    # Skipped in raw-waveform mode: the stims are (1, T_audio) waveforms, not
    # (1, F, T) gtgrams, and any spectral normalisation belongs in the model's
    # wav2spec front-end. Responses are still normalised below, identically to
    # gtgram mode, so the two modes stay response-for-response equivalent.
    if normalize_stims:
        F = stims[0].shape[1]
        if log_compress:
            for s in stims:
                s.copy_(_log_compress(s, log_offset))
        band_min = torch.full((F,), float("inf"))
        band_max = torch.full((F,), float("-inf"))
        for s in stims:
            sq = s[0]                                  # (F, T)
            band_min = torch.minimum(band_min, sq.amin(dim=1))
            band_max = torch.maximum(band_max, sq.amax(dim=1))
        band_rng = band_max - band_min
        band_rng[band_rng == 0] = 1.0                  # avoid divide-by-zero
        for s in stims:
            s.sub_(band_min.view(1, F, 1)).div_(band_rng.view(1, F, 1))
            s[s < 1e-6] = 0.0                          # NEMS "quiet → zero"

    # ---- RESP: per-neuron minmax across all reps + stims ----
    N = len(responses[0]) if responses else 0
    n_min = [float("inf")] * N
    n_max = [float("-inf")] * N
    for row in responses:
        for n, t in enumerate(row):
            if t.numel() > 1:                      # skip (1,1) NaN sentinels
                n_min[n] = min(n_min[n], float(t.min()))
                n_max[n] = max(n_max[n], float(t.max()))
    for row in responses:
        for n, t in enumerate(row):
            if t.numel() > 1 and n_max[n] > n_min[n]:
                t.sub_(n_min[n]).div_(n_max[n] - n_min[n])
                t[t < 1e-6] = 0.0                  # mirror NEMS clamp (no-op when min=0)


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
