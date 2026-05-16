"""Espejo (Lopez-Espejo et al. 2019) auditory cortex dataset.

Public Zenodo deposit (DOI ``10.5281/zenodo.3445557``) ships two
disjoint releases — natural sounds (NAT) and vocalization-modulated
noise (VMN) — recorded from awake passively-listening ferret A1. One
dataset class covers both via the ``stimuli={'nat', 'vmn'}`` constructor
arg; the two share no cells and have different F (18 vs 2), so they
cannot be concatenated.

The on-disk format is NEMS-flavored but we parse it directly with
``h5py`` + ``pandas`` — see ``deepSTRF.datasets.audio._espejo_native``.
No ``nems0`` dependency.
"""

from __future__ import annotations

import os
import re
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._espejo_native import (
    EspejoSite,
    extract_epoch_rasters,
    load_espejo_site,
    stim_occurrence_counts,
)
from deepSTRF.utils.data_download import (
    default_cache_dir,
    untar,
    zenodo_download,
)


# Public Zenodo record. https://doi.org/10.5281/zenodo.3445557
ESPEJO_ZENODO_RECORD = 3445557

# Per-stimuli-set constants. The Zenodo archive untars into ``<key>/NAT/`` or
# ``<key>/VMN/``, with per-site ``<exptid>_<hash>.tgz`` files inside.
_STIMULI_CONFIG = {
    "nat": {
        "archive_name": "A1_natural_sounds.tgz",
        "subdir": "A1_natural_sounds/NAT",
        "F": 18,
        "stimfmt": "ozgf",
    },
    "vmn": {
        "archive_name": "A1_voc_mod_noise.tgz",
        "subdir": "A1_voc_mod_noise/VMN",
        "F": 2,
        "stimfmt": "envelope",
    },
}


def download_espejo(stimuli: str, dest: Optional[str] = None) -> str:
    """Download one Espejo stimuli set from Zenodo into ``dest``.

    Parameters
    ----------
    stimuli : {'nat', 'vmn'}
    dest : str, optional
        Defaults to ``default_cache_dir('Espejo')`` (overridable via
        ``$DEEPSTRF_DATA_DIR``).

    Returns
    -------
    str
        The dataset root directory.

    Notes
    -----
    Idempotent: skips the archive if already present, and skips the
    untar step if the expected ``<subdir>/`` already exists.
    NAT is ~638 MB, VMN is ~25 MB.
    """
    assert stimuli in _STIMULI_CONFIG, (
        f"stimuli must be one of {list(_STIMULI_CONFIG)} (got {stimuli!r})"
    )
    cfg = _STIMULI_CONFIG[stimuli]
    dest_path = str(default_cache_dir("Espejo") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)

    archive_path = os.path.join(dest_path, cfg["archive_name"])
    if not os.path.exists(archive_path):
        zenodo_download(ESPEJO_ZENODO_RECORD, cfg["archive_name"], archive_path)

    extracted_dir = os.path.join(dest_path, cfg["subdir"])
    if not os.path.isdir(extracted_dir):
        untar(archive_path, dest_path)

    return dest_path


# Cell-id formats differ across the two sets:
#   NAT: 'AMT003c-11-1'  -> site=AMT003c, channel=11 (digits), unit=1 (digits)
#   VMN: 'btn144a-c1'    -> site=btn144a, channel=c (letter), unit=1 (digits)
# The 'site' is always the first dash-separated segment; the animal is the
# alphabetic prefix of the site (variable length, but typically 3 letters).
_CELL_ID_RE = re.compile(
    r"^(?P<site>(?P<animal>[A-Za-z]+)\d+[A-Za-z]?)-(?P<chan>[A-Za-z]*\d+)(?:-(?P<unit>\d+))?$"
)


def _parse_espejo_cell_id(cell_id: str) -> dict:
    """Best-effort decomposition of an Espejo cell id.

    Returns a dict with ``site``, ``animal_id``, ``channel``, ``unit``.
    Any field whose source is missing or unparseable is set to ``None``.
    """
    out = {"site": None, "animal_id": None, "channel": None, "unit": None}
    if not isinstance(cell_id, str):
        return out
    m = _CELL_ID_RE.match(cell_id)
    if m is None:
        # fallback: at least try to split off the site
        if "-" in cell_id:
            out["site"] = cell_id.split("-", 1)[0]
        return out
    out["site"] = m.group("site")
    out["animal_id"] = m.group("animal")
    out["channel"] = m.group("chan")
    out["unit"] = m.group("unit")  # may be None for 2-segment VMN ids
    return out


class Espejo_Dataset(AudioNeuralDataset):
    """A PyTorch dataset for Lopez-Espejo et al. (2019) ferret A1 recordings.


    =============== SOURCE ================

    Lopez Espejo M, Schwartz ZP, David SV. (2019) Spectral tuning of
    adaptation supports coding of sensory context in auditory cortex.
    *PLoS Computational Biology* 15(10): e1007430.
    https://doi.org/10.1371/journal.pcbi.1007430

    Data freely available at https://doi.org/10.5281/zenodo.3445557
    (no account required) — auto-fetched with ``download=True``.


    =============== DETAILS ================

    Awake, passively-listening adult ferret primary auditory cortex (A1),
    extracellularly recorded single units. Two disjoint releases (no
    cell overlap, different stimulus dimensionality — they cannot be
    concatenated):

    - ``stimuli='nat'``: 93 3-second natural sounds (animal vocalizations,
      speech, environmental, music). Stimuli stored as 18-band gammatone
      log-spectrograms (NEMS "ozgf"); ``F=18``. ~540 cells across 35
      experiment sites in 6 ferrets. Each site presents a subset of the
      stim bank.

    - ``stimuli='vmn'``: 30 3-second vocalization-modulated noise stimuli
      (two narrowband noise streams modulated by independent natural-
      vocalization envelopes). Stimuli stored as 2-band envelopes
      ("envelope" stimfmt); ``F=2``. ~200 cells across 103 sites in 5
      ferrets.

    Both releases sample at 100 Hz (``dt=10 ms`` native). The on-disk
    cochleagrams are log-compressed at source. Each occurrence epoch
    includes the published 0.5 s pre-stim + 0.5 s post-stim silence
    flanking the 3 s stimulus, so per-stim tensors are ``(1, F, 500)``
    (NAT) or ``(1, F, 400)`` (VMN).

    Estimation / test split follows the paper's
    ``split_by_occurrence_counts``: stimuli presented at the maximum
    repetition count within a site are the test set (~10 reps for NAT,
    ~15 reps for VMN), the rest are estimation (1 rep for NAT, ~3 reps
    for VMN). The per-stim ``n_repeats`` and ``split`` fields are
    surfaced in ``self.stim_meta`` so this can be cross-checked.


    =============== STRUCTURE ================

    Follows the standard deepSTRF data paradigm (see
    ``docs/_source/md/data_paradigm.md``).

    Espejo-specific metadata:

    - ``self.stims``           list of S tensors ``(1, F, T)`` — pre-computed
                               cochleagrams pulled directly from ``stim.h5``,
                               de-duplicated across sites.
    - ``self.responses``       list of S lists of N tensors ``(R_{s,n}, T)``;
                               ``(1, 1)`` NaN sentinel where cell n was not
                               recorded for stim s (different sites present
                               different stim subsets).
    - ``self.stim_meta``       list of S dicts ``{"name", "type"='nat'|'vmn',
                               "n_repeats", "split"='test'|'estimation',
                               "duration_s", "n_samples"}``.
    - ``self.neuron_metadata`` list of N dicts ``{"cell_id", "site",
                               "animal_id", "channel", "unit",
                               "experiment_set"='nat'|'vmn'}``. ``unit``
                               can be ``None`` for VMN cells (2-segment
                               cellids).


    =============== REMARKS ================

    - The raw waveforms for the NAT stimuli are not in the Zenodo deposit
      — only the pre-computed cochleagrams. The LBHB bitbucket mirrors
      the raw .wav files (see Lopez-Espejo et al. README). A future
      revision could expose ``stimfmt='waveform'`` for finer time
      resolution; the current loader fixes ``dt_ms = 10``.

    - ``stimuli='nat'`` and ``stimuli='vmn'`` instantiate disjoint
      populations with different ``F`` and cannot be concatenated.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        stimuli: Literal["nat", "vmn"] = "nat",
        dt_ms: float = 10.0,
        subset: Literal["all", "estimation", "test"] = "all",
        cells: Optional[Sequence[str]] = None,
        smooth: bool = False,
        download: bool = False,
    ):
        """
        Parameters
        ----------
        path : str, optional
            Path to the Espejo data folder (containing ``A1_natural_sounds/``
            and / or ``A1_voc_mod_noise/``). Defaults to the platformdirs
            cache (``$DEEPSTRF_DATA_DIR`` overrides).
        stimuli : {'nat', 'vmn'}
            Which release to load. The two are mutually exclusive (disjoint
            cells, different F); to use both, instantiate twice and keep
            them separate.
        dt_ms : float, default 10.0
            Time-bin width in ms. Currently fixed at 10 ms — the on-disk
            cochleagrams are precomputed at fs=100 Hz, and the response
            rasterizer aligns to that grid.
        subset : {'all', 'estimation', 'test'}, default 'all'
            If 'estimation' or 'test', only that stim subset is kept.
            Split follows the paper's ``split_by_occurrence_counts``:
            test = stims at max repetition count per site; estimation
            = stims at lower repetition counts.
        cells : sequence of str, optional
            Whitelist of cell IDs to include (intersection with what's
            on disk). None keeps all.
        smooth : bool, default False
            If True, smooth PSTHs with a 21 ms Hanning window (Hsu /
            Borst / Theunissen 2004). Off by default — Espejo is
            typically used as-is.
        download : bool, default False
            If True and the data is missing under ``path``, fetch the
            requested archive from Zenodo (record 3445557) and untar
            in place.
        """
        assert stimuli in _STIMULI_CONFIG, (
            f"stimuli must be one of {list(_STIMULI_CONFIG)} (got {stimuli!r})"
        )
        assert subset in ("all", "estimation", "test"), (
            f"subset must be 'all', 'estimation' or 'test' (got {subset!r})"
        )
        assert dt_ms == 10.0, (
            f"Espejo cochleagrams are precomputed at dt=10 ms; got dt_ms={dt_ms}. "
            f"Re-binning would also require re-deriving cochleagrams from "
            f"the raw waveforms (not in the Zenodo deposit)."
        )

        if path is None:
            path = str(default_cache_dir("Espejo"))
        if download:
            download_espejo(stimuli, path)

        super().__init__(path, dt_ms)

        cfg = _STIMULI_CONFIG[stimuli]
        self.species = "ferret"
        self.behavioral_state = "awake-passive"
        self.F = cfg["F"]
        self.stimuli = stimuli

        sites_dir = os.path.join(path, cfg["subdir"])
        if not os.path.isdir(sites_dir):
            raise FileNotFoundError(
                f"Espejo {stimuli!r} sites directory missing: {sites_dir}. "
                f"Pass download=True to fetch from Zenodo, or place the "
                f"per-site .tgz archives there manually."
            )

        archive_files = sorted(
            f for f in os.listdir(sites_dir)
            if f.endswith((".tgz", ".tar.gz"))
        )
        if not archive_files:
            raise FileNotFoundError(
                f"No per-site .tgz archives found in {sites_dir}."
            )

        ##############################
        # 1. load every site once
        ##############################

        cells_whitelist = set(cells) if cells is not None else None

        sites: list[EspejoSite] = []
        for fname in tqdm(archive_files, desc=f"Espejo {stimuli} sites"):
            site = load_espejo_site(os.path.join(sites_dir, fname))
            assert site.stim_format == cfg["stimfmt"], (
                f"site {site.site_id}: stimfmt {site.stim_format!r} != "
                f"expected {cfg['stimfmt']!r}"
            )
            sites.append(site)

        ##############################
        # 2. global cell list and the site indices per cell
        ##############################

        # A handful of cells (e.g. several por* in VMN) appear in multiple
        # .tgz archives — same site_id, different sessions (different
        # hashes in the filename). Treat these as one cell whose response
        # is the concatenation of all session rasters: per (cell, stim),
        # we pull rasters from every session-site that has both, then
        # cat along the repeat axis.
        cell_to_site_indices: dict[str, list[int]] = {}
        for site_idx, site in enumerate(sites):
            for cell in site.cellids:
                if cells_whitelist is not None and cell not in cells_whitelist:
                    continue
                cell_to_site_indices.setdefault(cell, []).append(site_idx)

        cells_ordered = sorted(cell_to_site_indices.keys())  # stable global order
        if not cells_ordered:
            raise ValueError(
                f"No cells matched the whitelist (cells={cells!r})."
            )

        self.neuron_metadata = [
            {
                "cell_id": c,
                "experiment_set": stimuli,
                **_parse_espejo_cell_id(c),
            }
            for c in cells_ordered
        ]
        self.N_neurons = len(self.neuron_metadata)

        ##############################
        # 3. global stim list — first-seen wins for the spectrogram, occurrence
        #    counts taken as the per-site max across the sites that played it
        ##############################

        # stim_name -> {"cochleagram": (F, T) np.ndarray,
        #               "max_reps": int (max across sites),
        #               "is_test": bool (test in at least one site),
        #               "per_site_reps": list[(site_idx, n_reps)]}
        # Paper convention (split_by_occurrence_counts, NEMS):
        # within each site, stims at the site's maximum occurrence count
        # are the test set, the rest are estimation. Globally we mark a
        # stim 'test' if any site classified it as such.
        stim_registry: dict[str, dict] = {}
        for site_idx, site in enumerate(sites):
            site_counts = stim_occurrence_counts(site.epochs)
            if not site_counts:
                continue
            site_max = max(site_counts.values())
            for sname, n_reps in site_counts.items():
                is_test_here = (n_reps == site_max) and (site_max > 1)
                if sname not in stim_registry:
                    coch = site.stim_cochleagrams.get(sname)
                    if coch is None:
                        continue
                    stim_registry[sname] = {
                        "cochleagram": coch,
                        "max_reps": n_reps,
                        "is_test": is_test_here,
                        "per_site_reps": [(site_idx, n_reps)],
                    }
                else:
                    stim_registry[sname]["max_reps"] = max(
                        stim_registry[sname]["max_reps"], n_reps
                    )
                    stim_registry[sname]["is_test"] = (
                        stim_registry[sname]["is_test"] or is_test_here
                    )
                    stim_registry[sname]["per_site_reps"].append((site_idx, n_reps))

        def _split_for_name(name: str) -> str:
            return "test" if stim_registry[name]["is_test"] else "estimation"

        stim_names = sorted(stim_registry.keys())  # stable global order

        # apply subset filter
        if subset != "all":
            stim_names = [n for n in stim_names if _split_for_name(n) == subset]

        if not stim_names:
            raise ValueError(
                f"No stims matched subset={subset!r}. Available splits in "
                f"this release: estimation, test."
            )

        ##############################
        # 4. fill self.stims, self.responses, self.stim_meta
        ##############################

        bin_s = dt_ms / 1000.0

        self.stims = []
        self.responses = []
        self.stim_meta = []

        for sname in stim_names:
            entry = stim_registry[sname]
            coch = entry["cochleagram"]  # (F, T)
            assert coch.shape[0] == self.F, (
                f"stim {sname}: cochleagram F={coch.shape[0]} != expected "
                f"self.F={self.F}"
            )
            T = int(coch.shape[1])
            spec = torch.from_numpy(coch).float().unsqueeze(0)  # (1, F, T)

            # responses: one tensor per cell. NaN sentinel where no session
            # the cell appears in played this stim. Otherwise concatenate
            # rasters across all the cell's sessions that played the stim.
            pop_resps = []
            sites_playing = {site_idx for site_idx, _ in entry["per_site_reps"]}
            for cell in cells_ordered:
                cell_sessions = cell_to_site_indices[cell]
                relevant = [si for si in cell_sessions if si in sites_playing]
                if not relevant:
                    pop_resps.append(torch.full((1, 1), float("nan")))
                    continue
                pieces = []
                for site_idx in relevant:
                    rasters = extract_epoch_rasters(
                        sites[site_idx], cell, sname, bin_s=bin_s, T=T,
                    )
                    if rasters.shape[0] > 0:
                        pieces.append(rasters)
                if not pieces:
                    pop_resps.append(torch.full((1, 1), float("nan")))
                else:
                    stacked = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
                    pop_resps.append(torch.from_numpy(stacked).float())

            self.stims.append(spec)
            self.responses.append(pop_resps)
            self.stim_meta.append({
                "name": sname,
                "type": stimuli,
                "n_repeats": int(entry["max_reps"]),
                "split": _split_for_name(sname),
                "duration_s": float(T * bin_s),
                "n_samples": T,
            })

        if smooth:
            self.smooth_responses(window_ms=21.0)

        self.validate()
