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

import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._logspectrogram import logspectrogram, n_bands_for
from deepSTRF.datasets.audio._crcns_ac1_native import (
    CellRecord,
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


# NERSC mirror paths (verified against the dataset's About page; matches the
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
    """Fetch the three CRCNS-AC1 archives from the NERSC mirror.

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
    signal_type : {'subthresh', 'spikes'}, default ``'subthresh'``
        ``'subthresh'`` (default): MedGauss-detrended Vm in mV-relative
        units. Matches the Machens 2004 / Asari 2009 / Rançon 2025
        modelling target. Pair with MSE loss.
        ``'spikes'``: high-pass detrend → threshold → 21 ms Hann smooth.
        Matches the legacy Asari ``'psth'`` path; an opt-in alternative
        for callers who want a firing-rate-like target.
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
    gating : RepeatGating, optional
        Per-repeat artifact-rejection thresholds. Default values gate
        out repeats with derivative-MAD jumps and excessive dynamic
        range; see :class:`._crcns_ac1_native.RepeatGating`.
    download : bool, default False
        If True, fetch the three archives via :func:`download_ac1`
        before extraction. Requires CRCNS credentials.
    username, password : str, optional
        CRCNS credentials. Default to ``$CRCNS_USERNAME`` /
        ``$CRCNS_PASSWORD`` env vars.
    drop_neuron12_artifact : bool, default True
        Reproduce the legacy Wehr neuron-12 carve-out: drop response
        #11 (recording dropout) and truncate the second half of
        response #10 (drift). Match Rançon 2024/2025 numbers.

    Notes
    -----
    deepSTRF data paradigm — see ``docs/_source/md/data_paradigm.md``.
    Per-stim metadata:

    - ``stim_meta`` dicts hold ``experimenter``, ``category``, ``idx``
      (Wehr) or ``class_n`` / ``segments`` / ``segment_files`` (Asari),
      ``description``, ``duration_s``.
    - ``nrn_meta`` dicts hold ``experimenter``, ``session``,
      ``animal_id``, ``penetration``, ``date``, ``site``,
      ``recording_type``, ``species``, plus ``_wehr_cell_idx`` for
      Wehr cells (used with ``WEHR_VALID_NEURONS`` /
      ``WEHR_NEURONS_SPLIT_NATURAL`` for Rançon-paper reproducibility).

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
        signal_type: str = "subthresh",
        dt_ms: float = 5.0,
        fmin: float = 100.0,
        fmax: float = 45000.0,
        bins_per_octave: int = 6,
        window_ms: Optional[float] = None,
        gating: Optional[RepeatGating] = None,
        download: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        drop_neuron12_artifact: bool = True,
    ):
        experimenters = _coerce(experimenter) or ("wehr", "asari")
        sites_t = _coerce(sites) or ("A1", "MGB")
        for e in experimenters:
            assert e in ("wehr", "asari"), (
                f"experimenter must be 'wehr', 'asari', or both (got {e!r})"
            )
        for s in sites_t:
            assert s in ("A1", "MGB"), (
                f"sites must be 'A1' and/or 'MGB' (got {s!r})"
            )
        assert signal_type in ("subthresh", "spikes"), (
            f"signal_type must be 'subthresh' or 'spikes' (got {signal_type!r})"
        )
        assert dt_ms > 0, f"dt_ms must be positive (got {dt_ms})"
        assert fmax > fmin > 0, f"need 0 < fmin < fmax (got {fmin}, {fmax})"
        assert bins_per_octave >= 1, f"bins_per_octave >= 1 (got {bins_per_octave})"

        # --- resolve path ---
        if download:
            path = download_ac1(path, username=username, password=password)
        if path is None:
            path = str(default_cache_dir("CRCNS_AC1"))

        super().__init__(path, dt_ms)
        self.species = "rat"
        self.experimenters = experimenters
        self.sites = sites_t
        self.signal_type = signal_type
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.bins_per_octave = int(bins_per_octave)
        self.window_ms = window_ms
        self.gating = gating or RepeatGating()
        self.drop_neuron12_artifact = bool(drop_neuron12_artifact)

        self.F = n_bands_for(self.fmin, self.fmax, self.bins_per_octave)

        # --- extract zips lazily; locate the three subtrees ---
        wehr_dir, asari1_dir, asari2_dir = ensure_extracted(path)

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
            n_idx = len(nrn_meta)
            cell_added = False
            for stim in cell.stims:
                # ---- clean + bin response ----
                if signal_type == "subthresh":
                    cleaned, reasons = prepare_repeats(
                        stim.raw_repeats, stim.sf_resp, gating=self.gating,
                    )
                else:  # 'spikes'
                    # MedGauss is baked into detect_spikes_psth; we still gate
                    # via prepare_repeats first to drop motion artifacts.
                    cleaned_v, reasons = prepare_repeats(
                        stim.raw_repeats, stim.sf_resp, gating=self.gating,
                    )
                    cleaned = [detect_spikes_psth(r, stim.sf_resp) for r in cleaned_v]
                for r in reasons:
                    rejection_counter[r] = rejection_counter.get(r, 0) + 1
                if not cleaned:
                    continue

                # Bin each repeat to dt_ms grid, length-align across repeats.
                binned = [bin_response(r, stim.sf_resp, dt_ms) for r in cleaned]
                T_resp = min(b.size for b in binned)
                if T_resp <= 0:
                    continue
                binned = np.stack([b[:T_resp] for b in binned])  # (R, T_resp)

                # ---- register stim (dedup) ----
                key = stim.key
                if key not in unique_stims:
                    s_idx = len(stim_specs)
                    unique_stims[key] = s_idx
                    stim_specs.append({
                        "waveform": stim.waveform,
                        "sf_stim": stim.sf_stim,
                        "duration_ms": stim.duration_ms,
                        "meta": dict(stim.meta),
                    })
                else:
                    s_idx = unique_stims[key]

                buffer.append((s_idx, n_idx, torch.from_numpy(binned).float()))
                cell_added = True

            if cell_added:
                nrn_meta.append(dict(cell.meta))
            else:
                nonlocal cell_count_dropped_for_zero_stims
                cell_count_dropped_for_zero_stims += 1

        # ----- iterate Wehr -----
        if "wehr" in experimenters and "A1" in sites_t:
            for cell in iterate_wehr_cells(
                wehr_dir,
                drop_neuron12_artifact=self.drop_neuron12_artifact,
            ):
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

        # --- pass 2: compute spectrograms + assemble (S, N) response grid ---
        S = len(stim_specs)

        NAN = torch.full((1, 1), float("nan"))
        self.stims = []
        self.stim_meta = []
        # Pre-fill responses with NaN sentinels.
        self.responses = [[NAN for _ in range(self.N_neurons)] for _ in range(S)]

        # Spectrogram each unique stim at the target dt_ms.
        # Length convention: the spectrogram output has T = ceil(L_wave / hop)
        # frames; the binned response also has T_resp = L_resp // block frames.
        # We truncate both to the smaller length for a clean per-stim T.
        stim_T_out: List[int] = []
        for s_idx, spec in enumerate(stim_specs):
            S_db, _freqs = logspectrogram(
                spec["waveform"], spec["sf_stim"],
                dt_ms=self.dt, fmin=self.fmin, fmax=self.fmax,
                bins_per_octave=self.bins_per_octave,
                window_ms=self.window_ms,
            )
            T_spec = int(S_db.shape[1])
            T_dur = int(round(spec["duration_ms"] / self.dt))
            # Use the duration-implied T as the canonical T (matches the
            # response binning), but clip to whatever the spectrogram has.
            T_canon = min(T_dur, T_spec) if T_dur > 0 else T_spec
            stim_T_out.append(T_canon)

            self.stims.append(
                torch.from_numpy(S_db[:, :T_canon]).unsqueeze(0).float()  # (1, F, T)
            )
            self.stim_meta.append(spec["meta"])

        # Place buffered responses into the (S, N) grid, length-aligning to
        # the canonical T per stim.
        for (s_idx, n_idx, tens) in buffer:
            T_canon = stim_T_out[s_idx]
            # tens is (R, T_resp); truncate to T_canon (or pad with NaN if too
            # short — should be rare since both came from the same duration_ms).
            T_resp = int(tens.shape[1])
            if T_resp >= T_canon:
                aligned = tens[:, :T_canon]
            else:
                pad = torch.full((tens.shape[0], T_canon - T_resp), float("nan"))
                aligned = torch.cat([tens, pad], dim=1)
            self.responses[s_idx][n_idx] = aligned

        self.validate()
