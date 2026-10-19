"""CRCNS-AC1 — intracellular Vm in rat A1 + MGB (Wehr 2002-2003 / Asari 2005-2007).

Reference
---------
Asari H., Wehr M., Machens C. & Zador A. (2009). "Auditory cortex and
thalamic neuronal responses to various natural and synthetic sounds."
CRCNS.org. http://dx.doi.org/10.6080/K0KW5CXR

Two published-paper companion datasets:

- **Wehr** subset — used in Machens, Wehr & Zador (2004), "Linearity of
  Cortical Receptive Fields Measured with Natural Sounds," *J. Neurosci.*
  24(5): 1089-1100. ~25 whole-cell recordings in anaesthetised rat A1,
  Vm sampled at 4 kHz.
- **Asari** subset — used in Asari & Zador (2009), "Long-Lasting Context
  Dependence Constrains Neural Encoding Models in Rodent Auditory
  Cortex," *J. Neurophysiol.* 102(5): 2638-2656. ~160 recordings in rat
  A1 + MGB (whole-cell + cell-attached), Vm sampled at 10 kHz; stimuli
  are spliced sequences of natural-sound segments.

This loader is **Python-only** — no MATLAB runtime needed. The CRCNS
archive ships raw recording ``.mat`` files + stimulus waveforms; we
parse them directly via ``scipy.io.loadmat`` and:

- compute a Hamming-windowed log-spectrogram at exactly the target
  temporal resolution (``dt_ms``) via a Goertzel STFT at log-spaced
  frequencies, faithful to ``wehr/Tools/logspectrogram.m`` — see
  :mod:`deepSTRF.datasets.audio._logspectrogram`;
- detrend each Vm repeat with a MedGauss baseline subtraction and gate
  out repeats that fail dynamic-range or derivative-MAD tests (drift +
  motion artifacts are common in these recordings) — see
  :mod:`deepSTRF.datasets.audio._crcns_ac1_native`.

CRCNS is auth-walled (free account). ``download=True`` fetches the
three archives via :func:`deepSTRF.utils.data_download.crcns_download`
using ``$CRCNS_USERNAME`` / ``$CRCNS_PASSWORD``.
"""
from __future__ import annotations

import json
import os
import random
import re
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._logspectrogram import logspectrogram, n_bands_for
from deepSTRF.datasets.audio._crcns_ac1_native import (
    CellRecord,
    _load_asari_stim_by_relpath,
    apply_ramp,
    RepeatGating,
    bin_response,
    detect_spikes_psth,
    ensure_extracted,
    iterate_asari_cells,
    iterate_wehr_cells,
    prepare_repeats,
)
from deepSTRF.utils.data_download import (
    crcns_download,
    default_cache_dir,
)


# ---------------------------------------------------------------------------
# Reproducibility constants (Rançon 2024 / 2025)
# ---------------------------------------------------------------------------

# 21 of the 25 Wehr cells used in the Rançon papers — drops the unresponsive
# indices 1, 2, 4, 8 reported in Machens et al. 2004.
WEHR_VALID_NEURONS: Tuple[int, ...] = (
    0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
)

# Per-neuron (train, val, test) stim-count splits used in Rançon 2024/2025.
# Index i is the per-cell ``_wehr_cell_idx`` in ``nrn_meta``; entries for cells
# not in WEHR_VALID_NEURONS are kept for completeness but are not meant to be
# used.
WEHR_NEURONS_SPLIT_NATURAL: Tuple[Tuple[int, int, int], ...] = (
    (7, 2, 2),     # 0
    (5, 1, 1),     # 1   unresponsive
    (3, 1, 1),     # 2   unresponsive
    (7, 1, 1),     # 3
    (1, 1, 1),     # 4   not enough data
    (6, 1, 1),     # 5
    (3, 1, 1),     # 6
    (1, 1, 1),     # 7
    (4, 1, 1),     # 8   unresponsive
    (7, 1, 2),     # 9
    (11, 2, 3),    # 10
    (25, 4, 7),    # 11
    (7, 1, 2),     # 12
    (25, 4, 7),    # 13
    (7, 1, 1),     # 14
    (14, 3, 5),    # 15
    (25, 4, 7),    # 16
    (17, 3, 5),    # 17
    (5, 1, 1),     # 18
    (11, 2, 3),    # 19
    (5, 1, 1),     # 20
    (12, 2, 4),    # 21
    (7, 2, 2),     # 22
    (44, 6, 13),   # 23
    (4, 1, 1),     # 24
)


# Wehr stimulus files (``'<category>/<idx>'``) that share audio. The first ~10 s of
# 13 of the 18 fragment files reappear inside 14 streams, and fragments 12 / 13 are
# themselves built from fragment 8 and a screaming piha recording; streams share no
# audio with each other. Found by matching the log-spectrograms of all 51 files frame
# by frame and confirmed on the waveforms (r >= 0.99 over the shared stretch). The
# 2002 cells heard only fragments and the 2003 cells only streams, so the overlap
# links two cohorts through any model they share. Every other file is its own group.
WEHR_SHARED_AUDIO: Tuple[Tuple[str, ...], ...] = (
    ("fragments/8", "fragments/12", "fragments/13",
     "category2/5", "category2/10", "category2/15"),
    ("fragments/7", "fragments/9", "category2/18"),
    ("fragments/1", "category1/10"),
    ("fragments/2", "category2/2"),
    ("fragments/3", "category1/11"),
    ("fragments/4", "category1/1"),
    ("fragments/5", "category2/7"),
    ("fragments/6", "category2/17"),
    ("fragments/14", "category2/3"),
)

# Asari sessions left out by default. 20071009-ha1-003 (MGB, cell-attached): its
# spikes are ~13 mV, far above the 2.5-sigma threshold (~1 mV), so each spike's
# rebound and small bumps also cross it; half its inter-spike intervals stay under
# 2 ms even with the 1-ms refractory period. Pass exclude_sessions=() to keep it.
ASARI_EXCLUDED_SESSIONS: Tuple[str, ...] = ("20071009-ha1-003",)

# All Wehr natural-sound stimulus files: (category, number of files).
_WEHR_CORPUS: Tuple[Tuple[str, int], ...] = (
    ("fragments", 18), ("category1", 11), ("category2", 18), ("category3", 4),
)


_ASARI_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "_crcns_ac1_asari_fragments.json")
_ASARI_TABLE: Optional[Dict] = None


def _asari_table() -> Dict:
    global _ASARI_TABLE
    if _ASARI_TABLE is None:
        with open(_ASARI_TABLE_PATH) as f:
            _ASARI_TABLE = json.load(f)
    return _ASARI_TABLE


def asari_fragment_sources() -> Dict[str, Tuple[str, str]]:
    """Asari segment file (``'class6/3l.mat'``) -> (source id, source name).

    The source is the CD track the fragment was cut from (``'<CD>/<track>'``),
    parsed from the file's description. The base file and its short (``s``) and
    long (``l``) cuts share a source. Lists the 49 files used by the
    natural-sound sequences.
    """
    return {f: tuple(v) for f, v in _asari_table()["sources"].items()}


def wehr_overlap_group(category: str, idx: int) -> str:
    """The ``overlap_group`` of a Wehr stimulus file: the name (``'wehr/<category>/<idx>'``)
    of the first file of its :data:`WEHR_SHARED_AUDIO` group, or its own name."""
    key = f"{category}/{idx}"
    for group in WEHR_SHARED_AUDIO:
        if key in group:
            return "wehr/" + group[0]
    return "wehr/" + key


# CRCNS download paths (verified against the dataset's About page; matches the
# convention used by crcns_aa{1,2,4}: ``<dataset>/<archive>``).
_AC1_DOWNLOAD_SPECS = (
    ("crcns-ac1.zip",                 "ac-1/crcns-ac1.zip"),
    ("crcns-ac1-asari-results-1.zip", "ac-1/crcns-ac1-asari-results-1.zip"),
    ("crcns-ac1-asari-results-2.zip", "ac-1/crcns-ac1-asari-results-2.zip"),
)


def download_ac1(
    dest: Optional[str] = None,
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Fetch the three CRCNS-AC1 archives from the CRCNS download server.

    Requires a free CRCNS account (https://crcns.org/register). Credentials
    can be passed explicitly or sourced from ``$CRCNS_USERNAME`` /
    ``$CRCNS_PASSWORD``. Idempotent: skips archives that already exist on
    disk; extraction is handled lazily on first dataset instantiation.

    Returns the destination directory.
    """
    dest_path = str(default_cache_dir("CRCNS_AC1") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)
    for zip_name, nersc_path in _AC1_DOWNLOAD_SPECS:
        zip_path = os.path.join(dest_path, zip_name)
        if not os.path.exists(zip_path):
            crcns_download(nersc_path, zip_path,
                           username=username, password=password)
    return dest_path


def _coerce(value: Union[None, str, Iterable[str]]) -> Optional[Tuple[str, ...]]:
    """Accept None, a str, or an iterable; return tuple or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(value)


class CRCNSAC1Dataset(AudioNeuralDataset):
    """Unified loader for the Wehr + Asari subsets of CRCNS-AC1.

    Both subsets are intracellular Vm in anaesthetised rat auditory
    pathway — Wehr in A1 (whole-cell, sf=4 kHz), Asari in A1 + MGB
    (whole-cell + cell-attached, sf=10 kHz) — and both record natural-
    sound responses with multi-trial repeats per stimulus. The loader
    deduplicates stimuli across cells (via the shared NaN-sentinel
    paradigm) so the same waveform never gets a duplicate spectrogram
    when it was presented to multiple cells.

    Parameters
    ----------
    path : str, optional
        Directory holding (or about to hold) the three CRCNS-AC1 zips
        and their extracted contents. Defaults to ``default_cache_dir(
        'CRCNS_AC1')`` (overridable via ``$DEEPSTRF_DATA_DIR``).
    experimenter : str or iterable of str, optional
        ``'wehr'``, ``'asari'``, or both. Default loads both.
    sites : str or iterable of str, optional
        ``'A1'`` and/or ``'MGB'``. Default loads both. Wehr is all-A1;
        Asari has both areas.

        The **signal type is not a free choice** — it is determined by
        each cell's recording mode, because the recording mode dictates
        what signal physically exists:

        - **whole-cell** (Wehr A1, Asari A1) → ``'subthresh'``:
          MedGauss-detrended membrane potential in mV (signed). Action
          potentials were blocked (Wehr) or not analysed (Asari A1, per
          the paper); the synaptic input *is* the signal. Pair with MSE.
        - **cell-attached** (Asari MGB) → ``'spikes'``: spike count per
          bin (non-negative; one spike per threshold crossing, with a 1-ms
          refractory period; see ``smooth``). There is no intracellular Vm
          in cell-attached mode. Pair with Poisson.

        Each cell carries its resolved type in ``nrn_meta['signal_type']``;
        ``self.signal_type`` is that type if the loaded cohort is
        homogeneous, else ``'mixed'`` (loading A1 + MGB together mixes
        signed-mV and spike-rate neurons — filter by site / signal_type
        before training one model across them).
    dt_ms : float, default 5.0
        Output time-bin width in ms. The Goertzel STFT is parametrised
        to produce its frames at exactly this resolution (no two-step
        compute-then-downsample); the response is average-pooled to
        match.
    fmin, fmax : float
        Spectrogram frequency range in Hz. Defaults to the Asari 2025
        layout: ``(100.0, 45000.0)``. Pass ``fmax=25600.0`` to recover
        the Wehr 2024 setting (49 bands).
    bins_per_octave : int, default 6
        Spectrogram spectral density. With the defaults this yields
        ``F=53``.
    window_ms : float, optional
        STFT analysis-window length in ms. Defaults to ``2 * dt_ms``
        (legacy MATLAB ``overlap=2``).
    detrend_med_ms : float, default 100.0
        Median-filter window (ms) for the MedGauss baseline subtracted
        from each Vm trace. Larger windows remove only slow drift and
        preserve more low-frequency response dynamics; smaller windows
        detrend more aggressively. The 100 ms default matches the
        Rançon 2024/2025 pipeline; the choice is robust (residuals
        barely change between 100 and 1000 ms because the response is
        dominated by fast PSP transients).
    detrend_gauss_ms : float, default 10.0
        Gaussian-smoothing σ (ms) applied to the median-filtered
        baseline before subtraction.
    gating : RepeatGating, optional
        Per-repeat artifact-rejection thresholds. Default values gate
        out repeats with derivative-MAD jumps and excessive dynamic
        range; see :class:`._crcns_ac1_native.RepeatGating`.
    return_waveform : bool, default False
        If True, hand out the raw stimulus waveform per stim as a
        ``(1, T_audio)`` mono tensor resampled to ``audio_fs`` instead
        of the in-loader log-spectrogram, for use with a learnable
        ``wav2spec`` model front-end. Because the source waveforms have
        heterogeneous sample rates (Asari at 97656 Hz, Wehr differs), they
        are all resampled to the single ``audio_fs`` so the grid-lock
        (``T_audio = T_neural * hop``, ``hop = audio_fs * dt_ms / 1000``)
        holds dataset-wide. offset 0 — the waveform starts at stimulus
        onset, matching the response trace. Responses are identical to
        spectrogram mode.
    asari_level : {'sequence', 'segment'}, default 'sequence'
        Stimulus unit for the Asari cells. ``'sequence'``: one stimulus per
        presented sequence of fragments (the recording as played, with its
        long context); no leak-free train / test split exists at this level,
        since every fragment recurs in several sequences. ``'segment'``: one
        stimulus per fragment file; each sequence is cut at the fragment
        boundaries (cumulative file lengths) and the trials of all
        occurrences of a fragment are pooled as its repeats. The context
        before a fragment is then not part of the stimulus, and its effect on
        the response counts as trial-to-trial variability. Occurrences the
        recording stops inside are dropped. Wehr stimuli are unaffected.
    smooth : bool, default False
        Cell-attached (MGB) spikes only: if True, spread each spike over a
        21-ms Hann window before binning (the loader's former default). Off
        by default: the window is centred, so a smoothed bin holds spikes
        from up to ~10 ms later (the target is no longer strictly causal).
    exclude_sessions : iterable of str, default ``ASARI_EXCLUDED_SESSIONS``
        Recording sessions to leave out. The default drops one MGB cell
        whose spike detection fails (see ``ASARI_EXCLUDED_SESSIONS``); pass
        ``()`` to load every session.
    audio_fs : int, default 96000
        Common sample rate the waveforms are resampled to in waveform
        mode (ignored in spectrogram mode, where ``self.audio_fs`` is
        ``None``). The default 96 kHz exceeds twice the default ``fmax``
        (45 kHz) so no in-band content is lost, and grid-locks cleanly
        for any integer ``dt_ms`` (96000/1000 = 96 samples per ms).
    download : bool, default False
        If True, fetch the three archives via :func:`download_ac1`
        before extraction. Requires CRCNS credentials.
    username, password : str, optional
        CRCNS credentials. Default to ``$CRCNS_USERNAME`` /
        ``$CRCNS_PASSWORD`` env vars.

    Notes
    -----
    deepSTRF data paradigm — see ``docs/_source/md/data_paradigm.md``.
    Per-stim metadata:

    - ``stim_meta`` dicts hold a unique ``name`` (``'wehr/<category>/<idx>'``,
      ``'asari/<segment files joined by +>'`` for a sequence, or
      ``'asari/<segment file>'`` at ``asari_level='segment'``),
      ``experimenter``, ``category``, ``idx`` (Wehr) or ``class_n`` /
      ``segments`` / ``segment_files`` (Asari sequence; ``class_n`` is the
      class of the first file, sequences mix classes), ``description``
      (NOT unique for sequences: the same description recurs across
      classes), ``duration_s``. Wehr stimuli add ``overlap_group`` (files
      that share audio have the same value, see :meth:`group_split`).
      Asari segments add ``segment_file``, ``source`` (the CD track it was
      cut from, ``'<CD>/<track>'``) and ``source_name``, ``overlap_group``
      (``'asari/<source>'``) and ``occurrences``: the
      ``(sequence name, position, previous segment file or None, onset in
      s)`` of every occurrence pooled into it.
    - ``nrn_meta`` dicts hold ``experimenter``, ``session``,
      ``animal_id``, ``penetration``, ``date``, ``site``,
      ``recording_type``, ``signal_type`` (``'subthresh'`` /
      ``'spikes'``, derived from the recording mode), ``species``, plus
      ``_wehr_cell_idx`` for Wehr cells (used with ``WEHR_VALID_NEURONS``
      / ``WEHR_NEURONS_SPLIT_NATURAL`` for Rançon-paper reproducibility).

    References
    ----------
    Machens, Wehr & Zador (2004). *J. Neurosci.* 24(5):1089-1100.
    Asari & Zador (2009). *J. Neurophysiol.* 102(5):2638-2656.
    Rançon, Masquelier & Cottereau (2025). *Commun. Biol.* 8:1456.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        experimenter: Union[None, str, Iterable[str]] = ("wehr", "asari"),
        sites: Union[None, str, Iterable[str]] = ("A1", "MGB"),
        dt_ms: float = 5.0,
        fmin: float = 100.0,
        fmax: float = 45000.0,
        bins_per_octave: int = 6,
        window_ms: Optional[float] = None,
        detrend_med_ms: float = 100.0,
        detrend_gauss_ms: float = 10.0,
        gating: Optional[RepeatGating] = None,
        return_waveform: bool = False,
        asari_level: str = "sequence",
        smooth: bool = False,
        exclude_sessions: Iterable[str] = ASARI_EXCLUDED_SESSIONS,
        audio_fs: int = 96000,
        download: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        experimenters = _coerce(experimenter) or ("wehr", "asari")
        excluded = frozenset(exclude_sessions)
        sites_t = _coerce(sites) or ("A1", "MGB")
        for e in experimenters:
            assert e in ("wehr", "asari"), (
                f"experimenter must be 'wehr', 'asari', or both (got {e!r})"
            )
        for s in sites_t:
            assert s in ("A1", "MGB"), (
                f"sites must be 'A1' and/or 'MGB' (got {s!r})"
            )
        if asari_level not in ("sequence", "segment"):
            raise ValueError(f"asari_level must be 'sequence' or 'segment', got {asari_level!r}")
        assert dt_ms > 0, f"dt_ms must be positive (got {dt_ms})"
        assert fmax > fmin > 0, f"need 0 < fmin < fmax (got {fmin}, {fmax})"
        assert bins_per_octave >= 1, f"bins_per_octave >= 1 (got {bins_per_octave})"
        if return_waveform:
            ratio = audio_fs * dt_ms / 1000.0
            assert abs(ratio - round(ratio)) < 1e-6 and round(ratio) >= 1, (
                f"waveform mode needs audio_fs * dt_ms / 1000 to be a positive "
                f"integer (got audio_fs={audio_fs}, dt_ms={dt_ms} -> {ratio}). "
                f"Pick e.g. audio_fs=96000 (default), which grid-locks for any "
                f"integer dt_ms."
            )

        # --- resolve path ---
        if download:
            path = download_ac1(path, username=username, password=password)
        if path is None:
            path = str(default_cache_dir("CRCNS_AC1"))

        super().__init__(path, dt_ms)
        self.species = "rat"
        self.experimenters = experimenters
        self.sites = sites_t
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.bins_per_octave = int(bins_per_octave)
        self.window_ms = window_ms
        self.detrend_med_ms = float(detrend_med_ms)
        self.detrend_gauss_ms = float(detrend_gauss_ms)
        self.gating = gating or RepeatGating()
        self.return_waveform = bool(return_waveform)
        self.asari_level = asari_level
        # audio_fs is the in-loader sample rate only in waveform mode; in
        # spectrogram mode the stims keep heterogeneous native rates, so we
        # report None (the base class's "this is a spectrogram dataset" signal).
        self.audio_fs = int(audio_fs) if self.return_waveform else None
        # Rat audiogram (Heffner & Heffner 2007): ~250 Hz – 76 kHz.
        # Informational only; the in-loader audio is band-limited to
        # audio_fs / 2 (48 kHz at the 96 kHz default).
        self.hearing_range_hz = (250.0, 76000.0)

        self.F = n_bands_for(self.fmin, self.fmax, self.bins_per_octave)

        # --- extract zips lazily; locate the three subtrees ---
        wehr_dir, asari1_dir, asari2_dir = ensure_extracted(path)
        asari_stims = os.path.join(os.path.dirname(wehr_dir), "asari", "Stimuli")
        segment_level = asari_level == "segment"
        segment_cache: Dict[str, Tuple[np.ndarray, float]] = {}

        def _segment(f):
            """(waveform, sf) of an Asari segment file, loaded once."""
            if f not in segment_cache:
                segment_cache[f] = _load_asari_stim_by_relpath(asari_stims, f)
            return segment_cache[f]

        # --- pass 1: walk cells, dedup stims, buffer cleaned binned responses ---
        # ``unique_stims[key]`` -> int s_idx; ``stim_specs[s_idx]`` holds the
        # waveform + meta we'll spectrogram once at the end. ``buffer`` lists
        # the (s_idx, n_idx, (R, T) cleaned binned tensor) triples that need
        # to be placed into the (S, N) grid.
        unique_stims: Dict[Tuple, int] = {}
        stim_specs: List[Dict] = []
        buffer: List[Tuple[int, int, torch.Tensor]] = []
        nrn_meta: List[Dict] = []
        # Track per-cell, per-stim repeat-rejection counts for diagnostics.
        rejection_counter: Dict[str, int] = {"range": 0, "step": 0, "xcorr": 0, "kept": 0}
        cell_count_dropped_for_zero_stims = 0

        def process_cell(cell: CellRecord):
            if cell.meta["experimenter"] not in experimenters:
                return
            if cell.meta["site"] not in sites_t:
                return
            if cell.meta["session"] in excluded:
                return
            n_idx = len(nrn_meta)
            cell_added = False
            # Signal type is dictated by the recording mode: cell-attached
            # has spikes (no intracellular Vm); whole-cell has subthreshold
            # Vm (no spikes — blocked in Wehr, not analysed in Asari A1).
            cell_signal = (
                "spikes" if "attached" in cell.meta.get("recording_type", "").lower()
                else "subthresh"
            )
            seg_trials: Dict[str, List[np.ndarray]] = {}
            seg_occ: Dict[str, set] = {}

            def register(key, waveform, sf_stim, duration_ms, meta, binned):
                """Add one (stim, cell) response block; returns the stim index."""
                nonlocal cell_added
                T_resp = min(b.size for b in binned)
                if T_resp <= 0:
                    return None
                binned = np.stack([b[:T_resp] for b in binned])  # (R, T_resp)
                if key not in unique_stims:
                    unique_stims[key] = len(stim_specs)
                    stim_specs.append({
                        "waveform": waveform,
                        "sf_stim": sf_stim,
                        "duration_ms": duration_ms,
                        "meta": dict(meta),
                    })
                s_idx = unique_stims[key]
                buffer.append((s_idx, n_idx, torch.from_numpy(binned).float()))
                cell_added = True
                return s_idx

            for stim in cell.stims:
                # ---- clean + bin response ----
                if cell_signal == "subthresh":
                    cleaned, reasons = prepare_repeats(
                        stim.raw_repeats, stim.sf_resp, gating=self.gating,
                        detrend_med_ms=self.detrend_med_ms,
                        detrend_gauss_ms=self.detrend_gauss_ms,
                    )
                else:  # 'spikes' — cell-attached MGB only
                    # MedGauss is baked into detect_spikes_psth; we still gate
                    # via prepare_repeats first to drop motion artifacts.
                    cleaned_v, reasons = prepare_repeats(
                        stim.raw_repeats, stim.sf_resp, gating=self.gating,
                        detrend_med_ms=self.detrend_med_ms,
                        detrend_gauss_ms=self.detrend_gauss_ms,
                    )
                    # spike train x samples per bin: the bin average
                    # (bin_response) is then the spike count per bin
                    block = max(1, int(round(dt_ms * stim.sf_resp / 1000.0)))
                    cleaned = [
                        block * detect_spikes_psth(
                            r, stim.sf_resp, detrend_med_ms=self.detrend_med_ms,
                            detrend_gauss_ms=self.detrend_gauss_ms,
                            smooth_ms=21.0 if smooth else None)
                        for r in cleaned_v
                    ]
                for r in reasons:
                    rejection_counter[r] = rejection_counter.get(r, 0) + 1
                if not cleaned:
                    continue

                if segment_level and stim.meta["experimenter"] == "asari":
                    # Cut every repeat at the fragment boundaries (cumulative file
                    # lengths, exactly as the sequence was spliced); an occurrence
                    # the recording stops inside is dropped. The recorded duration
                    # is rounded (up to 0.5 ms shorter than the audio), so 1 ms of
                    # shortfall is accepted. Trials of all the occurrences of a
                    # fragment are pooled below.
                    n_min = min(r.size for r in cleaned) + int(round(stim.sf_resp / 1000.0))
                    onset, prev = 0.0, None
                    for pos, f in enumerate(stim.meta["segment_files"]):
                        dur = _segment(f)[0].size / _segment(f)[1]
                        a = int(round(onset * stim.sf_resp))
                        b = int(round((onset + dur) * stim.sf_resp))
                        if b <= n_min:
                            seg_trials.setdefault(f, []).extend(
                                bin_response(r[a:b], stim.sf_resp, dt_ms) for r in cleaned)
                            seg_occ.setdefault(f, set()).add(
                                (stim.meta["name"], pos, prev, round(onset, 4)))
                        onset, prev = onset + dur, f
                    continue

                # Bin each repeat to dt_ms grid, length-align across repeats.
                binned = [bin_response(r, stim.sf_resp, dt_ms) for r in cleaned]
                register(stim.key, stim.waveform, stim.sf_stim, stim.duration_ms,
                         stim.meta, binned)

            for f, trials in seg_trials.items():
                wav, sf = _segment(f)
                src, src_name = asari_fragment_sources()[f]
                class_n = int(re.match(r"class(\d+)/", f).group(1))
                s_idx = register(
                    ("asari", "segment", f), apply_ramp(wav, sf, 5.0), sf,
                    wav.size / sf * 1000.0,
                    {"name": f"asari/{f}", "experimenter": "asari",
                     "category": f"class{class_n}", "class_n": class_n,
                     "segment_file": f, "source": src, "source_name": src_name,
                     "overlap_group": f"asari/{src}", "description": src_name,
                     "duration_s": wav.size / sf, "occurrences": []},
                    trials)
                if s_idx is not None:
                    occ = stim_specs[s_idx]["meta"]["occurrences"]
                    occ.extend(o for o in sorted(seg_occ[f], key=str) if o not in occ)

            if cell_added:
                meta = dict(cell.meta)
                meta["signal_type"] = cell_signal
                nrn_meta.append(meta)
            else:
                nonlocal cell_count_dropped_for_zero_stims
                cell_count_dropped_for_zero_stims += 1

        # ----- iterate Wehr -----
        if "wehr" in experimenters and "A1" in sites_t:
            for cell in iterate_wehr_cells(wehr_dir):
                process_cell(cell)

        # ----- iterate Asari -----
        if "asari" in experimenters:
            asari_roots = [d for d in (asari1_dir, asari2_dir) if os.path.isdir(d)]
            for cell in iterate_asari_cells(asari_roots, sites=sites_t):
                process_cell(cell)

        if not nrn_meta:
            raise ValueError(
                f"No cells matched experimenter={experimenters} sites={sites_t}. "
                f"Try a wider filter or check that {path!r} contains the "
                f"extracted CRCNS-AC1 archives."
            )

        self.nrn_meta = nrn_meta
        self.N_neurons = len(self.nrn_meta)
        self._rejection_counter = rejection_counter
        self._dropped_cells_empty = cell_count_dropped_for_zero_stims

        # Dataset-level signal type: the single type if homogeneous, else
        # 'mixed'. A mixed cohort (whole-cell A1 + cell-attached MGB) holds
        # signed-mV and spike-rate neurons side by side — warn so callers
        # don't train one model across both without filtering.
        present = {m["signal_type"] for m in self.nrn_meta}
        self.signal_type = present.pop() if len(present) == 1 else "mixed"
        if self.signal_type == "mixed":
            warnings.warn(
                "CRCNSAC1Dataset loaded a mix of whole-cell (subthresh, signed mV) "
                "and cell-attached (spikes, non-negative) neurons. Their responses "
                "are in different units; filter by nrn_meta['signal_type'] (or by "
                "site) before training a single model. Set sites='A1' for "
                "subthreshold only, or sites='MGB' for spikes only.",
                RuntimeWarning,
            )

        # --- pass 2: compute spectrograms + assemble (S, N) response grid ---
        S = len(stim_specs)

        # First, for each stim, find the shortest binned response T across the
        # cells that heard it. T_canon_s is then min(this, T_spec, T_dur), so
        # responses are truncated rather than NaN-padded (NaN-padding would
        # falsely tag the cell as "missing" via the base class mask logic).
        shortest_resp_T: List[int] = [10**9] * S
        for (s_idx, _n_idx, tens) in buffer:
            t_r = int(tens.shape[1])
            if t_r < shortest_resp_T[s_idx]:
                shortest_resp_T[s_idx] = t_r

        NAN = torch.full((1, 1), float("nan"))
        self.stims = []
        self.stim_meta = []
        self.responses = [[NAN for _ in range(self.N_neurons)] for _ in range(S)]

        stim_T_out: List[int] = []
        for s_idx, spec in enumerate(stim_specs):
            wav_np = np.asarray(spec["waveform"], dtype=np.float64).ravel()
            sf_stim = float(spec["sf_stim"])
            # T_spec = logspectrogram's frame count = ceil(len / hop_native).
            # Compute it analytically (identical formula) so T_canon matches
            # spectrogram mode exactly even in waveform mode -> responses are
            # bit-identical across the two input representations.
            hop_native = max(int(round(self.dt * sf_stim / 1000.0)), 1)
            T_spec = int(np.ceil(len(wav_np) / hop_native))
            T_dur = int(round(spec["duration_ms"] / self.dt))
            T_resp_min = shortest_resp_T[s_idx]
            T_canon = min(t for t in (T_dur or 10**9, T_spec, T_resp_min) if t > 0)
            stim_T_out.append(T_canon)

            if self.return_waveform:
                stim_tensor = self._waveform_for_stim(wav_np, sf_stim, T_canon)
            else:
                S_db, _freqs = logspectrogram(
                    wav_np, sf_stim,
                    dt_ms=self.dt, fmin=self.fmin, fmax=self.fmax,
                    bins_per_octave=self.bins_per_octave,
                    window_ms=self.window_ms,
                )
                stim_tensor = torch.from_numpy(S_db[:, :T_canon]).unsqueeze(0).float()
            self.stims.append(stim_tensor)
            meta = spec["meta"]
            if meta["experimenter"] == "wehr":
                meta["overlap_group"] = wehr_overlap_group(meta["category"], meta["idx"])
            self.stim_meta.append(meta)

        for (s_idx, n_idx, tens) in buffer:
            T_canon = stim_T_out[s_idx]
            # Truncate (never pad with NaN — that would break the paradigm's
            # mask derivation since NaN means "this cell is missing here").
            self.responses[s_idx][n_idx] = tens[:, :T_canon]

        self.validate()

    def group_split(self, seed: Optional[int] = None,
                    fractions: Sequence[float] = (0.7, 0.1, 0.2)) -> Dict[str, List[int]]:
        """Split the stimuli into train / val / test, keeping shared audio together.

        **Wehr.** Files that share audio (same ``overlap_group``, see
        :data:`WEHR_SHARED_AUDIO`) always land in the same set. The stimuli
        stay whole streams and fragments, so their long context (adaptation)
        is kept. Groups holding a fragment file form the ``'fragments'``
        stratum (the streams that copy them go with them); the other streams
        are drawn per category.

        **Asari** (``asari_level='segment'`` only). The unit is the source
        track (``stim_meta['source']``): all cuts of a track and all their
        occurrences go to one set. The tracks are drawn fragment set by
        fragment set (the sets of tracks the Asari cells heard, A1 and MGB
        together, largest number of cells first), so the large sets get val
        and test tracks; a track drawn with an earlier set keeps its set.
        Seven tracks also play inside Wehr files (both cohorts used the same
        Cornell CDs); each takes the set of its Wehr group, so a model fitted
        to both cohorts never has that audio in training and test. At
        the sequence level no leak-free split exists (every fragment recurs
        in several sequences of its set), so this raises.

        In each stratum the units go, in a shuffled order, to the set
        furthest below its target share of that stratum. The draw is over the
        whole corpus (all 51 Wehr files; all Asari tracks and fragment sets),
        not over the stimuli of this instance: a given ``seed`` puts a sound in
        the same set whichever cells or sites are loaded.

        Parameters
        ----------
        seed : int, optional
            Seed of the draw. ``None`` draws a new random split at each call.
        fractions : sequence of 3 floats, default (0.7, 0.1, 0.2)
            Target train / val / test shares, in files (Wehr) or source tracks
            (Asari).

        Returns
        -------
        dict
            ``{'train': [...], 'val': [...], 'test': [...]}``: absolute stimulus
            indices (into ``stim_meta``), whatever the current selection. Use
            :meth:`stim_to_iter_indices` to build a ``torch.utils.data.Subset``.

        Raises
        ------
        ValueError
            If Asari stimuli are loaded at the sequence level.
        """
        if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(f"fractions must be 3 shares summing to 1, got {fractions}")
        if any(m["experimenter"] == "asari" and "source" not in m for m in self.stim_meta):
            raise ValueError(
                "No leak-free split exists for Asari sequences: every fragment recurs in "
                "several sequences of its set. Load asari_level='segment' to split.")
        keys = ("train", "val", "test")
        rng = random.Random(seed)

        def draw(units: Sequence[str], size: Dict[str, int], assign: Dict[str, str]):
            """Assign the not-yet-assigned ``units`` (one stratum), counting the
            already assigned ones toward the shares."""
            todo = sorted(u for u in set(units) if u not in assign)
            rng.shuffle(todo)
            total = sum(size[u] for u in set(units))
            count = dict.fromkeys(keys, 0)
            for u in set(units) - set(todo):
                count[assign[u]] += size[u]
            for u in todo:
                k = max(keys, key=lambda k: fractions[keys.index(k)] * total - count[k])
                assign[u] = k
                count[k] += size[u]

        assign: Dict[str, str] = {}
        # Wehr: units = overlap groups of the 51 files
        groups: Dict[str, List[str]] = {}
        for cat, n in _WEHR_CORPUS:
            for idx in range(1, n + 1):
                groups.setdefault(wehr_overlap_group(cat, idx), []).append(f"wehr/{cat}/{idx}")
        stratum = {g: "fragments" if any("/fragments/" in f for f in fs) else fs[0].split("/")[1]
                   for g, fs in groups.items()}
        size = {g: len(fs) for g, fs in groups.items()}
        for st in sorted(set(stratum.values())):
            draw([g for g in groups if stratum[g] == st], size, assign)
        # Asari: a track whose audio is also inside a Wehr file takes that file's set
        # (one model fitted to both cohorts must not see it in training and test)
        for src, wehr_groups in _asari_table()["wehr_links"].items():
            sets = {assign[g] for g in wehr_groups}
            assert len(sets) == 1, (src, wehr_groups)
            assign["asari/" + src] = sets.pop()
        # Asari: units = source tracks, drawn per fragment set
        for _n_cells, sources in _asari_table()["ensembles"]:
            draw(["asari/" + x for x in sources], {"asari/" + x: 1 for x in sources}, assign)

        split: Dict[str, List[int]] = {k: [] for k in keys}
        for i, m in enumerate(self.stim_meta):
            split[assign[m["overlap_group"]]].append(i)
        return split

    def _waveform_for_stim(
        self, waveform: np.ndarray, sf_stim: float, T_canon: int,
    ) -> torch.Tensor:
        """Resample a stim waveform to ``self.audio_fs`` and grid-lock it.

        Returns a ``(1, T_canon * hop)`` mono float32 tensor (``hop =
        audio_fs * dt_ms / 1000``). offset 0 — the source waveform starts at
        stimulus onset, matching the response trace, so no pre-silence inset
        is needed. Right-padded / cropped to the exact grid-locked length.
        """
        wav = torch.as_tensor(np.asarray(waveform).ravel(),
                              dtype=torch.float32).unsqueeze(0)   # (1, T_wav)
        if int(round(sf_stim)) != self.audio_fs:
            wav = torchaudio.functional.resample(
                wav, orig_freq=int(round(sf_stim)), new_freq=self.audio_fs,
            )
        T_audio = T_canon * self.hop
        n = wav.shape[-1]
        if n > T_audio:
            wav = wav[..., :T_audio]
        elif n < T_audio:
            wav = F.pad(wav, (0, T_audio - n), mode="constant", value=0.0)
        return wav
