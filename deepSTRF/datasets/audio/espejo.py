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
import warnings
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
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
    stream_download,
    untar,
    zenodo_download,
)


# Public Zenodo record. https://doi.org/10.5281/zenodo.3445557
ESPEJO_ZENODO_RECORD = 3445557

# Raw NAT waveforms are NOT in the Zenodo deposit (only the precomputed
# cochleagrams are). They live on the LBHB baphy bitbucket mirror, split
# across two sound dirs; sounds are keyed by filename (the ``STIM_<name>``
# epoch label minus the ``STIM_`` prefix). See the dataset README. The
# natural-sounds-set-3 bank Espejo used is under ``sounds_set3/`` (the older
# ``sounds/`` dir is tried as a fallback for any straggler).
_NAT_WAVEFORM_BASE = (
    "https://bitbucket.org/lbhb/baphy/raw/master/"
    "Config/lbhb/SoundObjects/%40NaturalSounds"
)
_NAT_WAVEFORM_SUBDIRS = ("sounds_set3", "sounds")
# Published protocol: each 4 s sound is flanked by 0.5 s pre- and post-stim
# silence, so the cochleagram (and the waveform) begin with 0.5 s of silence.
_NAT_PRESTIM_S = 0.5
# Native sample rate of the bitbucket wavs (44.1 kHz mono PCM16, 4 s each).
_NAT_WAVEFORM_FS = 44100


def download_espejo_nat_waveforms(
    names: Sequence[str],
    dest: Optional[str] = None,
    *,
    progress: bool = True,
) -> Dict[str, str]:
    """Fetch the raw NAT waveforms from the LBHB baphy bitbucket mirror.

    Parameters
    ----------
    names : sequence of str
        Stim names as they appear in ``stim_meta`` (``STIM_<file>.wav``); the
        ``STIM_`` prefix is stripped to get the on-mirror filename.
    dest : str, optional
        Parent directory; wavs are cached under ``<dest>/nat_waveforms/``.
        Defaults to ``default_cache_dir('Espejo')``.
    progress : bool, default True
        Show a tqdm bar over the (missing) downloads.

    Returns
    -------
    dict
        ``name -> local wav path`` for every name found on the mirror.

    Notes
    -----
    Idempotent — already-cached wavs are skipped. Each filename is tried in
    ``sounds_set3/`` first, then ``sounds/``. Names found in neither are
    collected and surfaced by the caller (the dataset raises on genuine
    misses so waveform mode never silently substitutes silence).
    """
    dest_path = Path(default_cache_dir("Espejo") if dest is None else dest)
    wav_dir = dest_path / "nat_waveforms"
    wav_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, str] = {}
    missing: list[str] = []
    todo = list(dict.fromkeys(names))  # de-dup, keep order
    bar = tqdm(todo, desc="Espejo NAT waveforms", disable=not progress)
    for name in bar:
        fn = name[len("STIM_"):] if name.startswith("STIM_") else name
        local = wav_dir / fn
        if local.exists():
            out[name] = str(local)
            continue
        for sub in _NAT_WAVEFORM_SUBDIRS:
            url = f"{_NAT_WAVEFORM_BASE}/{sub}/{quote(fn)}"
            try:
                stream_download(url, local, progress=False)
                out[name] = str(local)
                break
            except Exception:
                continue  # 404 in this subdir (or transient) -> try the next
        else:
            missing.append(fn)
    if missing:
        warnings.warn(
            f"{len(missing)} NAT waveform(s) not found on the bitbucket mirror "
            f"(tried {_NAT_WAVEFORM_SUBDIRS}): {missing[:5]}"
            + (" ..." if len(missing) > 5 else ""),
            RuntimeWarning,
        )
    return out

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


class EspejoDataset(AudioNeuralDataset):
    """PyTorch dataset for Lopez-Espejo et al. (2019) ferret A1 recordings.

    Awake, passively-listening adult ferret primary auditory cortex (A1),
    extracellularly recorded single units. The dataset ships in two disjoint
    releases (no cell overlap, different stimulus dimensionality — they
    cannot be concatenated), selected by the ``stimuli`` argument:

    - ``'nat'``: 93 3-second natural sounds (animal vocalizations, speech,
      environmental, music), stored as 18-band gammatone log-spectrograms
      (NEMS "ozgf", ``F=18``). ~540 cells across 35 sites in 6 ferrets;
      each site presents a subset of the stim bank.
    - ``'vmn'``: 30 3-second vocalization-modulated noise stimuli (two
      narrowband noise streams modulated by independent natural-vocalization
      envelopes), stored as 2-band envelopes ("envelope" stimfmt, ``F=2``).
      ~200 cells across 103 sites in 5 ferrets.

    Both releases sample at 100 Hz (``dt=10 ms`` native); the on-disk
    cochleagrams are log-compressed at source. Each occurrence epoch
    includes the published 0.5 s pre-stim + 0.5 s post-stim silence flanking
    the 3 s stimulus, so per-stim tensors are ``(1, F, 500)`` (NAT) or
    ``(1, F, 400)`` (VMN). The estimation / test split follows the paper's
    ``split_by_occurrence_counts`` and is surfaced via the per-stim
    ``n_repeats`` and ``split`` metadata fields.

    Data are freely available at https://doi.org/10.5281/zenodo.3445557 (no
    account required) and auto-fetched with ``download=True``.

    Notes
    -----
    Follows the standard deepSTRF data paradigm (see
    ``docs/_source/md/data_paradigm.md``). Espejo-specific metadata:

    - ``stim_meta`` dicts hold ``name``, ``type`` (``'nat'`` / ``'vmn'``),
      ``n_repeats``, ``split`` (``'test'`` / ``'estimation'``),
      ``duration_s`` and ``n_samples``.
    - ``nrn_meta`` dicts hold ``cell_id``, ``site``, ``animal_id``,
      ``channel``, ``unit`` and ``experiment_set`` (``'nat'`` / ``'vmn'``).
      ``unit`` can be ``None`` for VMN cells (2-segment cellids).

    The ``(1, 1)`` NaN sentinel marks ``(stim, neuron)`` pairs the cell was
    not recorded for (different sites present different stim subsets). Only
    the pre-computed cochleagrams are in the Zenodo deposit (the raw NAT
    waveforms are mirrored on the LBHB bitbucket); the loader fixes
    ``dt_ms = 10``.

    References
    ----------
    Lopez Espejo, Schwartz & David (2019). "Spectral tuning of adaptation
    supports coding of sensory context in auditory cortex." *PLoS
    Computational Biology* 15(10): e1007430.
    https://doi.org/10.1371/journal.pcbi.1007430
    """

    def __init__(
        self,
        path: Optional[str] = None,
        stimuli: Literal["nat", "vmn"] = "nat",
        dt_ms: float = 10.0,
        subset: Literal["all", "estimation", "test"] = "all",
        cells: Optional[Sequence[str]] = None,
        smooth: bool = False,
        return_waveform: bool = False,
        audio_fs: int = 44100,
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
        return_waveform : bool, default False
            If True (``stimuli='nat'`` only), hand out the raw natural-sound
            waveform per stim as a ``(1, T_audio)`` mono tensor at
            ``audio_fs`` instead of the precomputed ozgf cochleagram, for use
            with a learnable ``wav2spec`` model front-end. The raw wavs are
            **not** in the Zenodo deposit — they are fetched from the LBHB
            baphy bitbucket mirror (see the README) and cached under
            ``<path>/nat_waveforms/``. The 4 s sound is inset at the published
            0.5 s pre-stim silence offset so it grid-locks to the cochleagram
            frames (``T_audio = T_neural * hop``, ``hop = audio_fs·dt/1000``).
            Responses are identical to cochleagram mode. **VMN is unsupported**
            (its stimuli are synthesized 2-band envelopes with no raw audio).
        audio_fs : int, default 44100
            Sample rate for waveform mode (the native rate of the mirror
            wavs; ignored — and reported as ``None`` — in cochleagram mode).
            44.1 kHz grid-locks at dt=10 ms (hop=441).
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
        if return_waveform:
            assert stimuli == "nat", (
                f"return_waveform=True is only supported for stimuli='nat' "
                f"(VMN stimuli are synthesized 2-band vocalization-modulated "
                f"noise with no natural raw waveform on the mirror). Got "
                f"stimuli={stimuli!r}."
            )
            ratio = audio_fs * dt_ms / 1000.0
            assert abs(ratio - round(ratio)) < 1e-6 and round(ratio) >= 1, (
                f"waveform grid-lock needs audio_fs * dt_ms / 1000 to be a "
                f"positive integer (got audio_fs={audio_fs}, dt_ms={dt_ms} -> "
                f"{ratio}). The default audio_fs=44100 grid-locks at dt=10 ms "
                f"(hop=441)."
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
        self.return_waveform = bool(return_waveform)
        # audio_fs is the in-loader sample rate only in waveform mode; in
        # cochleagram mode the stims have no associated audio, so report None
        # (the base class's "this is a spectrogram dataset" signal).
        self.audio_fs = int(audio_fs) if self.return_waveform else None
        self.hearing_range_hz = (200.0, 40000.0)  # ferret (informational)

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

        self.nrn_meta = [
            {
                "cell_id": c,
                "experiment_set": stimuli,
                **_parse_espejo_cell_id(c),
            }
            for c in cells_ordered
        ]
        self.N_neurons = len(self.nrn_meta)

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

        # In waveform mode, fetch the raw NAT wavs (cached under
        # <path>/nat_waveforms/) for the kept stims. Raise on genuine misses
        # so the (stim, response) grid stays identical to cochleagram mode.
        wav_paths: Dict[str, str] = {}
        if self.return_waveform:
            wav_paths = download_espejo_nat_waveforms(stim_names, dest=path)
            missing = [n for n in stim_names if n not in wav_paths]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} NAT stim(s) have no raw waveform on the "
                    f"bitbucket mirror, so waveform mode cannot align them with "
                    f"the responses: {missing[:5]}"
                    + (" ..." if len(missing) > 5 else "")
                    + ". Use return_waveform=False (cochleagram mode) or contact "
                    "the dataset authors (see the README) for the missing sounds."
                )

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
            if self.return_waveform:
                # Grid-locked raw waveform (1, T*hop), the sound inset at the
                # published 0.5 s pre-stim silence offset (T keeps the
                # cochleagram frame count so responses stay identical).
                stim = self._nat_waveform_for_stim(wav_paths[sname], T)
            else:
                stim = torch.from_numpy(coch).float().unsqueeze(0)  # (1, F, T)

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

            self.stims.append(stim)
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

    def _nat_waveform_for_stim(self, wav_path: str, T_neural: int) -> torch.Tensor:
        """Load a NAT sound and grid-lock it to ``(1, T_neural * hop)``.

        The 4 s sound is inset at the published 0.5 s pre-stim silence offset
        (``_NAT_PRESTIM_S``) so it aligns with the cochleagram, whose first
        ~50 bins are that same silence; the trailing 0.5 s is the post-stim
        silence. Resampled to ``self.audio_fs`` if the source rate differs,
        then cropped to the exact grid-locked length.
        """
        wav, sr = torchaudio.load(wav_path)                  # (C, T_wav), float32
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)              # -> mono
        if int(sr) != self.audio_fs:
            wav = torchaudio.functional.resample(wav, int(sr), self.audio_fs)
        T_audio = T_neural * self.hop
        pre = int(round(_NAT_PRESTIM_S * self.audio_fs))
        out = torch.zeros(1, T_audio, dtype=torch.float32)
        end = min(pre + wav.shape[-1], T_audio)
        if end > pre:
            out[:, pre:end] = wav[:, : end - pre]
        return out
