"""Le, Bjoring & Meliza (2025), *Nature Communications* — zebra finch dataset.

"The zebra finch auditory cortex reconstructs occluded syllables in
conspecific song." DOI: 10.1038/s41467-025-63182-y. Data:
10.6084/m9.figshare.29203457. Code: github.com/melizalab/auditory-restoration.

Single-unit extracellular recordings from the auditory pallium of
anesthetized adult zebra finches, in response to 8 natural song motifs (and
in cohort 3, 8 scrambled-syntax pseudo-motifs) presented in up to 7 variants
per critical interval (CI) to probe the neural correlate of auditory
restoration.

**Sub-experiments** (one ``experiment=`` per instance; concat for the union):

``nat8a``
    Cohorts 1 & 2 — natural motifs (8 birds × 2 CIs × {C, G, N, GB, CB}).
    Cohort 1 (alpha) had a familiarity manipulation; cohort 2 (beta) did
    not. No masking variants.
``nat8b``
    Cohort 3 — same natural motifs renamed ``nat8mk0..7``, full set of 7
    variants per CI (adds GM, CM).
``synth8b``
    Cohort 3 — 8 scrambled-syntax pseudo-motifs, full variant set.

**Per-CI variants:**

``C`` (Continuous)
    Unmodified motif; shared across both CIs.
``G`` (Gap)
    CI replaced by silence.
``N`` (Noise)
    CI-duration noise burst in isolation.
``GB`` (Gap + Burst)
    CI replaced by noise within the motif; the illusion-inducing stimulus.
``CB`` (Continuous + Burst)
    Motif unchanged, noise added on top of the CI.
``GM`` (Gap-Masked)
    Whole motif masked, CI deleted (``nat8b`` / ``synth8b`` only).
``CM`` (Continuous-Masked)
    Whole motif masked, CI intact (``nat8b`` / ``synth8b`` only). CM is
    CI-independent, so it lives once per motif.
"""
from __future__ import annotations

import csv
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.utils.data_download import default_cache_dir, figshare_download, unzip

# Soft dep — the `gammatone` PyPI package provides the same filter bank used
# in Le, Bjoring & Meliza (2025) Methods p. 10. Imported lazily inside the
# spectrogram method so `from deepSTRF.datasets.audio import Meliza2025Dataset`
# still works in environments that don't have it.
try:
    from gammatone.gtgram import gtgram as _gtgram
except ImportError:  # pragma: no cover - exercised only when extra missing
    _gtgram = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENTS: Tuple[str, ...] = ("nat8a", "nat8b", "synth8b")
VARIANTS: Tuple[str, ...] = ("C", "G", "N", "GB", "CB", "GM", "CM")

# experiment → tuple of response subdirectories (in cohort order)
_EXP_RESPONSE_DIRS: Dict[str, Tuple[str, ...]] = {
    "nat8a":   ("nat8a-alpha-responses", "nat8a-beta-responses"),
    "nat8b":   ("nat8b-responses",),
    "synth8b": ("synth8b-responses",),
}
# Mapping response-subdir → cohort (per the dataset README.yml).
_RESPDIR_TO_COHORT: Dict[str, int] = {
    "nat8a-alpha-responses": 1,
    "nat8a-beta-responses":  2,
    "nat8b-responses":       3,
    "synth8b-responses":     3,
}

# Mapping from the verbose nat8a condition strings (used both in the pprox
# ``condition`` field of cohort 1 and in the nat8a stim filename stems) to
# (variant_code, ci_index).
_NAT8A_CONDITION_MAP: Dict[str, Tuple[str, Optional[int]]] = {
    "continuous":         ("C",  None),
    "gap1":               ("G",  1),
    "gap2":               ("G",  2),
    "noise1":             ("N",  1),
    "noise2":             ("N",  2),
    "gapnoise1":          ("GB", 1),
    "gapnoise2":          ("GB", 2),
    "continuousnoise1":   ("CB", 1),
    "continuousnoise2":   ("CB", 2),
}

# nat8b stem grammar:   ep_<motif>_<variant>[(_g<ci>)][(_snr<n>)]
# synth8b stem grammar: ep_<motif>_<variant>[(_g<ci_pos><ci_letter>)][(_snr<n>)]
# where ci is "1"/"2" (nat8b) or "2a"/"4a" (synth8b); CM and C have no CI tag.
_EP_STEM_RE = re.compile(
    r"^ep_(?P<motif>[^_]+)"
    r"_(?P<variant>C|G|N|GB|CB|GM|CM)"
    r"(?:_g(?P<ci>[A-Za-z0-9]+))?"
    r"(?:_snr(?P<snr>-?\d+))?$"
)
# nat8a stem grammar: <motif>_<verbose_condition>
_NAT8A_STEM_RE = re.compile(r"^(?P<motif>[^_]+)_(?P<condition>[a-z0-9]+)$")

FIGSHARE_DOI = "10.6084/m9.figshare.29203457"
FIGSHARE_ARTICLE_ID = "29203457"
# The figshare article ships a single zip; after unzip the data lives in this
# subdirectory of the cache. Used by the auto-download path to detect a
# successful prior unpack.
FIGSHARE_UNPACKED_DIR = "zebf-auditory-restoration-1"

# Paper-faithful gammatone spectrogram parameters (Le, Bjoring & Meliza 2025,
# Methods p. 10): 50 log-spaced bands from 1 to 8 kHz, 2.5 ms analysis window,
# 1 ms hop, ``log(power + 1)`` compression.
_PAPER_N_BANDS = 50
_PAPER_FMIN_HZ = 1000.0
_PAPER_FMAX_HZ = 8000.0
_PAPER_WINDOW_MS = 2.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _UnitSkipped(Exception):
    """Internal: signal that a pprox file should be dropped silently."""


def _parse_stim_stem(stem: str, experiment: str) -> Dict[str, Any]:
    """Parse a stim filename stem into ``{motif, critical_interval, variant}``.

    Returns ``critical_interval=None`` for stimuli that are CI-independent
    (the ``C`` continuous variant for all experiments, and ``CM`` for the
    nat8b/synth8b experiments — see the module docstring).
    """
    if experiment == "nat8a":
        m = _NAT8A_STEM_RE.match(stem)
        if m is None:
            raise ValueError(f"unrecognised nat8a stim stem: {stem!r}")
        cond = m.group("condition")
        if cond not in _NAT8A_CONDITION_MAP:
            raise ValueError(f"unknown nat8a condition {cond!r} in {stem!r}")
        variant, ci = _NAT8A_CONDITION_MAP[cond]
        return {"motif": m.group("motif"), "critical_interval": ci, "variant": variant}

    # nat8b / synth8b
    m = _EP_STEM_RE.match(stem)
    if m is None:
        raise ValueError(f"unrecognised {experiment} stim stem: {stem!r}")
    variant = m.group("variant")
    ci_raw = m.group("ci")
    if variant in ("C", "CM"):  # CI-independent in this dataset
        ci: Any = None
    elif ci_raw is None:
        # G/N/GB/CB/GM all require a CI tag — fall through with None and
        # let the caller decide whether to drop.
        ci = None
    else:
        # nat8b uses bare integers; synth8b uses "2a"/"4a".
        try:
            ci = int(ci_raw)
        except ValueError:
            ci = ci_raw  # keep as a string for synth8b
    return {"motif": m.group("motif"), "critical_interval": ci, "variant": variant}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_age_to_days(age: str) -> Optional[int]:
    """Parse ephys-birds.csv age strings like '1y234d' into total days."""
    if not age or age == "unknown":
        return None
    m = re.match(r"^(?:(\d+)y)?(?:(\d+)d)?$", age)
    if m is None:
        return None
    years = int(m.group(1) or 0)
    days = int(m.group(2) or 0)
    return years * 365 + days


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Meliza2025Dataset(AudioNeuralDataset):
    """deepSTRF wrapper for one sub-experiment of Le, Bjoring & Meliza (2025).

    Instantiate one per experiment (``"nat8a"`` | ``"nat8b"`` | ``"synth8b"``)
    and concatenate with ``concat_neural_datasets`` (or ``ds_a + ds_b``) for
    the union; the three experiments share no stimuli, so the bidirectional
    selection rule in the base class hides cross-experiment NaN-only entries
    automatically.

    Layout on disk (as shipped on figshare):

        <path>/
        ├── metadata/
        │   ├── recordings.csv          area for nat8a-beta / nat8b / synth8b
        │   ├── song-birds.csv          motif name mapping + CI timings (ms)
        │   └── ephys-birds.csv         cohort, sex, age, familiarity group
        ├── nat8a-alpha-responses/      cohort 1 (familiarity manipulation)
        ├── nat8a-beta-responses/       cohort 2
        ├── nat8a-stimuli/              shared by alpha + beta
        ├── nat8b-responses/            cohort 3 (natural-syntax)
        ├── nat8b-stimuli/
        ├── synth8b-responses/          cohort 3 (scrambled-syntax)
        └── synth8b-stimuli/

    Two pprox schemas coexist in the archive: the legacy ``spec:2/pprox`` (used
    only by nat8a-alpha; spike times in ms, ``condition`` field encodes
    variant) and ``spec:2/stimtrial`` (everything else; spike times in s,
    stimulus dict carries the full filename stem). Both are handled below.

    Parameters
    ----------
    path
        Filesystem path to the unpacked figshare archive (the directory that
        contains ``metadata/``, ``*-responses/``, ``*-stimuli/``).
    experiment
        One of ``"nat8a"`` | ``"nat8b"`` | ``"synth8b"``. ``nat8a`` unifies
        the two cohorts (alpha + beta) that share the same stim set; use
        ``select_pop_by_nrn_attr("cohort", 1)`` to restrict to the
        familiarity sub-experiment.
    dt_ms
        Bin width in ms. Default 5; pass ``dt_ms=1`` for paper-faithful
        spectrogram + response binning (the paper uses 1 ms throughout).
    n_bands
        Number of gammatone bands. Default 50, matching the paper.
    fmin, fmax
        Low / high edges of the gammatone filter bank, in Hz. Defaults
        1000 / 8000 — the paper's range.
    window_ms
        Gammatone analysis-window width, in ms. Default 2.5, matching the
        paper. Hop is ``dt_ms``.
    smooth
        If True (default), apply a 21 ms Hanning smoother to all PSTHs
        (Hsu / Borst / Theunissen 2004).
    keep_areas
        Optional iterable of area strings to filter on (per the per-unit
        ``area`` metadata field; values vary by cohort).
    compute_reliability
        If True (default), pre-compute per-neuron Sahani–Linden signal
        power, noise power, and SNR (length-weighted across stims) and
        attach them to ``nrn_meta``. Set to ``False`` for fast
        iteration when reliability filtering is not needed.
    download
        If ``True`` and ``path=None``, fetches the ~105 MB figshare archive
        (DOI ``10.6084/m9.figshare.29203457``) into the deepSTRF cache and
        unpacks it. Idempotent: skips the download if the unpacked tree is
        already present.

    Notes
    -----
    Stim-side metadata fields:
        - ``name``               : filename stem.
        - ``motif``              : e.g. ``"B189"`` (nat8a) or ``"nat8mk0"`` (nat8b).
        - ``critical_interval``  : ``int`` (1/2) for per-CI variants, ``None``
                                   for ``C`` and ``CM`` (CI-independent), or
                                   a string like ``"2a"`` for synth8b.
        - ``variant``            : one of ``VARIANTS``.
        - ``syntax``             : ``"natural"`` or ``"scrambled"``.
        - ``experiment``         : ``"nat8a"`` | ``"nat8b"`` | ``"synth8b"``.
        - ``sample_rate_hz``     : native sample rate of the source wav.
        - ``duration_s``         : duration of the source wav, in seconds.
        - ``ci_onset_s/ci_offset_s`` : critical-interval bounds in seconds
                                   (NaN for C/CM and for synth8b — the CI
                                   table only covers nat8a/nat8b).

    Per-neuron metadata fields:
        ``cell_id``, ``animal_id``, ``animal_uuid``, ``cohort``,
        ``experiment``, ``area``, ``hemisphere``, ``familiar_motifs``
        (list of motif IDs the bird was reared with; empty unless cohort 1),
        ``sex``, ``age_days``, ``pprox_file``.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        experiment: str = "nat8b",
        dt_ms: float = 5.0,
        n_bands: int = _PAPER_N_BANDS,
        fmin: float = _PAPER_FMIN_HZ,
        fmax: float = _PAPER_FMAX_HZ,
        window_ms: float = _PAPER_WINDOW_MS,
        smooth: bool = True,
        keep_areas: Optional[Sequence[str]] = None,
        compute_reliability: bool = True,
        download: bool = False,
    ):
        if experiment not in EXPERIMENTS:
            raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {experiment!r}")
        if _gtgram is None:
            raise ImportError(
                "Meliza2025Dataset requires the `gammatone` package to compute "
                "the paper's spectrogram representation. `pip install gammatone`."
            )

        if path is None:
            if not download:
                raise ValueError("Provide `path=` or set `download=True`.")
            path = self._figshare_download()
        elif download:
            # honour explicit `path=` + `download=True`: cache there if missing
            path = self._figshare_download(Path(path))

        super().__init__(str(path), float(dt_ms))
        self.path = str(path)
        self.experiment = experiment
        self.species = "zebra finch"
        self.behavioral_state = "anesthetized"
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.window_ms = float(window_ms)
        self.F = int(n_bands)

        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"dataset path does not exist: {root}")

        # --- 1. Metadata tables (CSVs in metadata/) -------------------------
        # ephys-birds.csv: cohort, sex, age, familiarity group — keyed by bird short name.
        # song-birds.csv:  motif name mapping (bird ↔ nat8mkN) + CI timings (ms).
        # recordings.csv:  area assignment for cohort 2/3 sites — keyed by site = bird_pen_site.
        self._birds_table = self._load_ephys_birds(root / "metadata" / "ephys-birds.csv")
        self._songbirds_table = self._load_song_birds(root / "metadata" / "song-birds.csv")
        self._recordings_table = self._load_recordings(root / "metadata" / "recordings.csv")
        self._familiarity_groups = self._build_familiarity_groups()

        # --- 2. Stimuli: scan, parse, compute spectrograms ------------------
        stim_dir = root / f"{experiment}-stimuli"
        if not stim_dir.is_dir():
            raise FileNotFoundError(f"stim directory not found: {stim_dir}")

        stim_records = []
        for wav in sorted(stim_dir.glob("*.wav")):
            try:
                rec = _parse_stim_stem(wav.stem, experiment)
            except ValueError as e:
                warnings.warn(f"skipping {wav.name}: {e}")
                continue
            rec["name"] = wav.stem
            rec["wav_path"] = wav
            stim_records.append(rec)
        if not stim_records:
            raise FileNotFoundError(f"no parseable .wav files under {stim_dir}")

        # Stable order: motif → ci (None last) → variant.
        def _sort_key(r):
            ci = r["critical_interval"]
            ci_key = (1, "") if ci is None else (0, str(ci))
            return (r["motif"], ci_key, r["variant"])
        stim_records.sort(key=_sort_key)

        self.stims: List[torch.Tensor] = []
        self.stim_meta: List[Dict[str, Any]] = []
        self._stim_idx_by_name: Dict[str, int] = {}
        for rec in stim_records:
            sr_native, n_samples, duration_s, spec = self._load_stim(rec["wav_path"])
            ci_on, ci_off = self._ci_bounds_seconds(rec, sr_native, duration_s)
            self.stims.append(spec)
            self.stim_meta.append({
                "name":              rec["name"],
                "motif":             rec["motif"],
                "critical_interval": rec["critical_interval"],
                "variant":           rec["variant"],
                "syntax":            "scrambled" if experiment == "synth8b" else "natural",
                "experiment":        experiment,
                "sample_rate_hz":    int(sr_native),
                "duration_s":        float(duration_s),
                "ci_onset_s":        ci_on,
                "ci_offset_s":       ci_off,
            })
            self._stim_idx_by_name[rec["name"]] = len(self.stims) - 1

        S = len(self.stims)
        T_per_stim = [s.shape[-1] for s in self.stims]

        # --- 3. Units: walk every pprox file across the response dirs -------
        per_unit_rows: List[List[torch.Tensor]] = []
        nrn_meta: List[Dict[str, Any]] = []
        for resp_dir_name in _EXP_RESPONSE_DIRS[experiment]:
            resp_dir = root / resp_dir_name
            if not resp_dir.is_dir():
                warnings.warn(f"response directory missing: {resp_dir}")
                continue
            cohort = _RESPDIR_TO_COHORT[resp_dir_name]
            pprox_files = sorted(resp_dir.glob("*.pprox"))
            if not pprox_files:
                warnings.warn(f"no .pprox files under {resp_dir}")
                continue
            for pprox_path in pprox_files:
                try:
                    row, nrn = self._load_unit(pprox_path, cohort, T_per_stim)
                except _UnitSkipped as e:
                    warnings.warn(f"skipping {pprox_path.name}: {e}")
                    continue
                if keep_areas is not None and nrn["area"] not in keep_areas:
                    continue
                per_unit_rows.append(row)
                nrn_meta.append(nrn)

        N = len(nrn_meta)
        if N == 0:
            raise RuntimeError(
                f"no usable units found for experiment={experiment!r} under {root}"
            )
        self.nrn_meta = nrn_meta
        self.N_neurons = N

        # Transpose to (S, N) storage layout.
        self.responses = [
            [per_unit_rows[n][s] for n in range(N)] for s in range(S)
        ]

        if smooth:
            self.smooth_responses(window_ms=21.0)

        if compute_reliability:
            self._attach_reliability_metrics()

        self.validate()

    # -----------------------------------------------------------------------
    # Auto-download (figshare)
    # -----------------------------------------------------------------------

    @staticmethod
    def _figshare_download(target_root: Optional[Path] = None) -> Path:
        """Fetch + unpack the figshare archive, return the unpacked root.

        Idempotent: if the ``zebf-auditory-restoration-1/`` subdirectory
        already exists under ``target_root`` (or the deepSTRF cache when
        ``target_root`` is None), no work is done. Otherwise:

        1. Resolve the article's single file via the figshare REST API.
        2. Stream-download the ~105 MB zip into ``target_root``.
        3. Unzip in place.
        """
        base = target_root.expanduser() if target_root is not None else default_cache_dir("Meliza_2025")
        base.mkdir(parents=True, exist_ok=True)
        unpacked = base / FIGSHARE_UNPACKED_DIR
        if unpacked.is_dir():
            return unpacked
        zip_path = figshare_download(FIGSHARE_ARTICLE_ID, base)
        unzip(zip_path, base)
        if not unpacked.is_dir():
            raise RuntimeError(
                f"figshare unpack did not produce expected directory "
                f"{FIGSHARE_UNPACKED_DIR!r} under {base}"
            )
        return unpacked

    # -----------------------------------------------------------------------
    # Per-neuron reliability metrics
    # -----------------------------------------------------------------------

    def _attach_reliability_metrics(self) -> None:
        """Compute per-neuron Sahani-Linden signal_power / noise_power / snr
        and attach them to ``self.nrn_meta``.

        For each neuron, gather its ``(R_s, T_s)`` responses across stims,
        pad into a ``(S_eff, 1, R_max, T_max)`` NaN tensor, and call the
        functional metrics from :mod:`deepSTRF.metrics`. Stim-axis
        aggregation is length-weighted (BLUE under variance ∝ 1/T_b — see
        ``metrics_paradigm.md`` §11). Cells with no qualifying stim get NaN.

        Stored fields:
            ``signal_power``  : Sahani-Linden SP_n (length-weighted)
            ``noise_power``   : NP_n
            ``snr``           : SP_n / NP_n  (∞ if NP_n ≈ 0)

        Split-half ``cc_max`` (Hsu / Schoppe) is intentionally NOT
        precomputed: it requires up to 126 disjoint half-trial correlations
        per (stim, neuron) and would dominate instantiation time. Users who
        need it can call ``normalized_corrcoef(method='hsu')`` on demand.
        """
        # Lazy import to avoid a circular dep at module load (utils.data → datasets).
        from deepSTRF.metrics import signal_power, noise_power, snr

        S = len(self.stims)
        for n in range(self.N_neurons):
            per_stim = []
            for s in range(S):
                r = self.responses[s][n]
                if r.shape == (1, 1) and torch.isnan(r).all():
                    continue
                per_stim.append(r)
            if not per_stim:
                self.nrn_meta[n].update(
                    signal_power=float("nan"),
                    noise_power=float("nan"),
                    snr=float("nan"),
                )
                continue
            R_max = max(int(r.shape[0]) for r in per_stim)
            T_max = max(int(r.shape[1]) for r in per_stim)
            big = torch.full((len(per_stim), 1, R_max, T_max), float("nan"))
            for i, r in enumerate(per_stim):
                R_i, T_i = r.shape
                big[i, 0, :R_i, :T_i] = r
            sp_n = signal_power(big, reduction="none").squeeze().item()
            np_n = noise_power(big, reduction="none").squeeze().item()
            snr_n = snr(big, reduction="none").squeeze().item()
            self.nrn_meta[n].update(
                signal_power=sp_n,
                noise_power=np_n,
                snr=snr_n,
            )

    # -----------------------------------------------------------------------
    # Spectrogram loading
    # -----------------------------------------------------------------------

    def _load_stim(self, wav_path: Path) -> Tuple[int, int, float, torch.Tensor]:
        """Return (native_sr, n_samples, duration_s, spec (1, F, T) tensor).

        Paper-faithful gammatone spectrogram (Le, Bjoring & Meliza 2025
        Methods p. 10): filter bank at the source sample rate, ``log(P+1)``
        power compression, no resampling. The on-disk wavs are already at the
        paper-stated RMS levels (−27 dBFS for nat8a, −30 dBFS for nat8b /
        synth8b), so no further amplitude normalization is needed.
        """
        wav, sr_native = sf.read(str(wav_path), always_2d=False)
        if wav.ndim == 2:  # collapse stereo if any
            wav = wav.mean(axis=-1)
        wav = np.asarray(wav, dtype=np.float64)
        n_samples_native = wav.shape[-1]
        duration_s = n_samples_native / float(sr_native)

        spec = _gtgram(
            wav,
            sr_native,
            window_time=self.window_ms * 1e-3,
            hop_time=self.dt * 1e-3,
            channels=self.F,
            f_min=self.fmin,
            f_max=self.fmax,
        )  # (F, T), bands ordered low → high.
        spec = np.log1p(np.clip(spec, a_min=0.0, a_max=None))
        spec_t = torch.as_tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, F, T)

        return int(sr_native), int(n_samples_native), float(duration_s), spec_t

    # -----------------------------------------------------------------------
    # Per-unit loading
    # -----------------------------------------------------------------------

    def _load_unit(
        self,
        pprox_path: Path,
        cohort: int,
        T_per_stim: List[int],
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        with open(pprox_path) as f:
            coll = json.load(f)

        schema = coll.get("$schema", "")
        legacy = schema.endswith("pprox.json#") or schema.endswith("pprox.json")

        if legacy:
            nrn = self._neuron_meta_from_legacy(pprox_path, coll, cohort)
            trials_by_stim = self._index_trials_legacy(coll)
            bin_fn = self._bin_trial_legacy
        else:
            nrn = self._neuron_meta_from_stimtrial(pprox_path, coll, cohort)
            trials_by_stim = self._index_trials_stimtrial(coll)
            bin_fn = self._bin_trial_stimtrial

        S = len(self._stim_idx_by_name)
        row: List[torch.Tensor] = [torch.full((1, 1), float("nan")) for _ in range(S)]

        for stim_name, trials in trials_by_stim.items():
            s = self._stim_idx_by_name.get(stim_name)
            if s is None:
                continue
            T = T_per_stim[s]
            psth = torch.zeros((len(trials), T), dtype=torch.float32)
            for r, trial in enumerate(trials):
                psth[r] = bin_fn(trial, T_bins=T)
            row[s] = psth

        if all(t.shape == (1, 1) and torch.isnan(t).all() for t in row):
            raise _UnitSkipped("no trials matched any known stimulus")
        return row, nrn

    # --- legacy spec:2/pprox.json (nat8a-alpha only) -----------------------

    def _index_trials_legacy(self, coll: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group legacy-pprox trials by reconstructed stim filename stem.

        In the legacy schema, ``trial['stimulus']`` is just the motif id and
        ``trial['condition']`` is the verbose variant suffix; concatenated
        with an underscore they recreate the wav filename stem in
        ``nat8a-stimuli/`` (e.g. ``"R253" + "_" + "continuousnoise2"``).
        """
        by_stim: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for trial in coll.get("pprox", []):
            motif = trial.get("stimulus")
            cond = trial.get("condition")
            if motif is None or cond is None:
                continue
            by_stim[f"{motif}_{cond}"].append(trial)
        return by_stim

    def _bin_trial_legacy(self, trial: Dict[str, Any], T_bins: int) -> torch.Tensor:
        """Bin a legacy-pprox trial (spike times in ms, key=``event``)."""
        events_ms = np.asarray(trial.get("event", []), dtype=np.float64)
        stim_on_ms = float(trial.get("stim_on", 0.0))
        rel_ms = events_ms - stim_on_ms
        idx = np.floor(rel_ms / self.dt).astype(np.int64)
        ok = (idx >= 0) & (idx < T_bins)
        counts = np.bincount(idx[ok], minlength=T_bins)[:T_bins]
        return torch.as_tensor(counts, dtype=torch.float32)

    def _neuron_meta_from_legacy(
        self, pprox_path: Path, coll: Dict[str, Any], cohort: int
    ) -> Dict[str, Any]:
        bird = coll.get("bird") or pprox_path.stem.split("_", 1)[0]
        animal_uuid = coll.get("bird-uuid")
        birds_info = self._birds_table.get(bird, {})
        # cohort-1 pprox carries area in ``location`` directly.
        area = coll.get("location") or "unknown"
        hemisphere = coll.get("hemisphere")
        familiar_group = coll.get("familiar")
        familiar_motifs = self._familiarity_groups.get(familiar_group, []) if familiar_group else []
        return {
            "cell_id":         pprox_path.stem,
            "animal_id":       bird,
            "animal_uuid":     animal_uuid or birds_info.get("uuid"),
            "cohort":          cohort,
            "experiment":      self.experiment,
            "area":            area,
            "hemisphere":      hemisphere,
            "familiar_motifs": familiar_motifs,
            "sex":             birds_info.get("sex") or coll.get("bird-sex"),
            "age_days":        birds_info.get("age_days") or _parse_age_to_days(coll.get("bird-age", "")),
            "pprox_file":      pprox_path.name,
        }

    # --- spec:2/stimtrial.json (nat8a-beta, nat8b, synth8b) -----------------

    def _index_trials_stimtrial(self, coll: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        by_stim: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for trial in coll.get("pprox", []):
            stim = trial.get("stimulus")
            if isinstance(stim, dict):
                name = stim.get("name")
            elif isinstance(stim, str):
                # Defensive: some stimtrial files might still use string refs.
                name = Path(stim).stem
            else:
                name = None
            if name:
                by_stim[name].append(trial)
        return by_stim

    def _bin_trial_stimtrial(self, trial: Dict[str, Any], T_bins: int) -> torch.Tensor:
        """Bin a stimtrial-pprox trial (spike times in s, key=``events``).

        The ``interval`` field gives the trial-recording window relative to
        ``stimulus.interval[0]`` — events here are already relative to stim
        onset, so no offset correction is needed.
        """
        events_s = np.asarray(trial.get("events", []), dtype=np.float64)
        dt_s = self.dt * 1e-3
        idx = np.floor(events_s / dt_s).astype(np.int64)
        ok = (idx >= 0) & (idx < T_bins)
        counts = np.bincount(idx[ok], minlength=T_bins)[:T_bins]
        return torch.as_tensor(counts, dtype=torch.float32)

    def _neuron_meta_from_stimtrial(
        self, pprox_path: Path, coll: Dict[str, Any], cohort: int
    ) -> Dict[str, Any]:
        # Pprox file stem grammar:  <bird>_<pen>_<site>_c<cluster>
        # e.g. "C43_1_1_c109"  →  bird=C43, site_key=C43_1_1
        parts = pprox_path.stem.split("_")
        bird = parts[0]
        site_key = "_".join(parts[:3]) if len(parts) >= 3 else bird
        site_info = self._recordings_table.get(site_key, {})
        birds_info = self._birds_table.get(bird, {})
        # The top-level ``bird`` field on stimtrial files is the UUID; we keep
        # both forms for traceability.
        return {
            "cell_id":         pprox_path.stem,
            "animal_id":       bird,
            "animal_uuid":     coll.get("bird") or birds_info.get("uuid"),
            "cohort":          cohort,
            "experiment":      self.experiment,
            "area":            site_info.get("area", "unknown"),
            "hemisphere":      site_info.get("hemisphere"),
            "familiar_motifs": [],
            "sex":             birds_info.get("sex"),
            "age_days":        birds_info.get("age_days"),
            "pprox_file":      pprox_path.name,
        }

    # -----------------------------------------------------------------------
    # Metadata helpers
    # -----------------------------------------------------------------------

    def _load_ephys_birds(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            warnings.warn(f"missing {path}; per-bird metadata will be limited.")
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for row in _read_csv_rows(path):
            out[row["bird"]] = {
                "uuid":     row.get("uuid"),
                "sex":      row.get("sex"),
                "group":    row.get("group"),
                "age_days": _parse_age_to_days(row.get("age", "")),
                "cohort":   int(row["cohort"]) if row.get("cohort") else None,
                "sire":     row.get("sire"),
            }
        return out

    def _load_song_birds(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Build a motif→CI-timings table keyed by both nat8a (bird) and nat8b
        (``nat8mkN``) motif names so it works regardless of experiment.

        Each entry also carries the canonical nat8a-style ``bird`` name in
        ``"bird"``; ``_build_familiarity_groups`` uses that to dedupe.
        """
        if not path.exists():
            warnings.warn(f"missing {path}; CI timings will be NaN for nat8a/nat8b.")
            return {}
        out: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"cis": {}})
        for row in _read_csv_rows(path):
            bird = row["bird"]
            encoded = row.get("nat8b_encoding") or row.get("nat8b_coding")
            try:
                gap = int(row["gap"])
                start_ms = float(row["gap_start"])
                stop_ms = float(row["gap_stop"])
            except (KeyError, ValueError):
                continue
            for key in (bird, encoded):
                if not key:
                    continue
                out[key]["cis"][gap] = (start_ms, stop_ms)
                out[key]["uuid"] = row.get("uuid")
                out[key]["group"] = row.get("group")
                out[key]["bird"] = bird  # canonical nat8a-style name
        return dict(out)

    def _load_recordings(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """site_key (= ``bird_pen_site``) → {area, hemisphere, probe, stimulus_set}."""
        if not path.exists():
            warnings.warn(f"missing {path}; area/hemisphere will be 'unknown'.")
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for row in _read_csv_rows(path):
            site = row.get("site")
            if not site:
                continue
            area = row.get("area")
            out[site] = {
                "bird":          row.get("bird"),
                "probe":         row.get("probe"),
                "stimulus_set":  row.get("stimulus_set"),
                "hemisphere":    row.get("hemisphere"),
                "area":          area if area and area != "N/A" else "unknown",
            }
        return out

    def _build_familiarity_groups(self) -> Dict[str, List[str]]:
        """Cohort 1 had two rearing rooms (e.g. ``178B`` and ``180B``); each
        room was paired with half of the 8 motif vocalizers. The room→motifs
        mapping lives implicitly in song-birds.csv (each song bird belongs to
        one room). Reconstruct it here so we can attach ``familiar_motifs``
        to cohort-1 units based on the ``familiar`` field of their pprox.

        Listed motifs use the nat8a-style bird names (``"R56"``, ``"B189"``,
        ...) since cohort 1 only ever heard those stims; users mapping into
        nat8b naming can resolve via ``self._songbirds_table[name]``.
        """
        groups: Dict[str, set] = defaultdict(set)
        for info in self._songbirds_table.values():
            grp = info.get("group")
            bird = info.get("bird")
            if grp and bird:
                groups[grp].add(bird)
        return {grp: sorted(motifs) for grp, motifs in groups.items()}

    def _ci_bounds_seconds(
        self,
        rec: Dict[str, Any],
        sr_native: int,
        duration_s: float,
    ) -> Tuple[float, float]:
        """Return ``(ci_onset_s, ci_offset_s)`` for this stim's metadata.

        CI bounds come from ``song-birds.csv`` and are listed in ms relative
        to motif onset. Returns ``(NaN, NaN)`` for CI-independent variants
        (C, CM), for synth8b (no entries in song-birds.csv), and for any
        motif we can't resolve.
        """
        ci = rec["critical_interval"]
        motif = rec["motif"]
        if ci is None or motif not in self._songbirds_table:
            return float("nan"), float("nan")
        cis = self._songbirds_table[motif].get("cis", {})
        # The CSV key is always an int (1/2); strip any string CI tag we may
        # have produced for synth8b (which won't be in the table anyway).
        try:
            ci_key = int(ci)
        except (TypeError, ValueError):
            return float("nan"), float("nan")
        if ci_key not in cis:
            return float("nan"), float("nan")
        start_ms, stop_ms = cis[ci_key]
        return start_ms / 1000.0, stop_ms / 1000.0

    # -----------------------------------------------------------------------
    # Convenience selectors for the restoration paradigm
    # -----------------------------------------------------------------------

    def select_variant(self, variant: str) -> List[int]:
        """Restrict iteration to one variant code, e.g. ``'GB'``."""
        return self.select_stims_by_attr("variant", variant)

    def select_motif(self, motif: str) -> List[int]:
        """Restrict iteration to all variants of one motif."""
        return self.select_stims_by_attr("motif", motif)

    def select_critical_interval(self, ci) -> List[int]:
        """Restrict iteration to one CI index (or ``None`` for C/CM variants)."""
        return self.select_stims_by_attr("critical_interval", ci)

    def select_restoration_quartet(
        self,
        motif: str,
        ci,
        variants: Sequence[str] = ("C", "CB", "GB", "GM"),
    ) -> List[int]:
        """Select the stim set used in the paper's core restoration analysis
        for one (motif, CI). Defaults to C / CB / GB / GM — the four
        trajectories compared in Fig. 4. Returns the selected stim indices.

        The ``C`` continuous and ``CM`` masked variants are CI-independent and
        are kept whenever they appear in ``variants``, regardless of ``ci``.
        """
        wanted = set(variants)
        ci_indep = {"C", "CM"}
        keep = [
            i for i, m in enumerate(self.stim_meta)
            if m["motif"] == motif
            and m["variant"] in wanted
            and (m["critical_interval"] == ci or m["variant"] in ci_indep)
        ]
        self.S_sel = sorted(keep)
        return self.S_sel
