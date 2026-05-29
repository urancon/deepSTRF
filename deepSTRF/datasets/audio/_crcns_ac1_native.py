"""Native (Python-only) ingestion of the CRCNS-AC1 release.

The CRCNS AC1 archive contains intracellular whole-cell Vm recordings
from anesthetised rat auditory cortex (Wehr 2002-2003) and rat A1 + MGB
(Asari 2005-2007). The official release ships raw recording .mat files
+ stimulus waveforms; previous deepSTRF loaders depended on MATLAB
preprocessing that produced intermediate per-cell .mat files. This
module parses the original CRCNS files directly via ``scipy.io.loadmat``
— no MATLAB runtime needed.

Two top-level iterators expose one cell at a time:

- :func:`iterate_wehr_cells` — walks ``crcns-ac1/wehr/Results/`` and
  yields ``(cell_meta, [stim_record, ...])``. One cell per session
  directory; one trigger per ``.mat`` trial file; stims grouped by
  ``trigger.param.description`` so that repeated presentations of the
  same fragment become a per-stim repeat axis.
- :func:`iterate_asari_cells` — walks ``asari-results-{1,2}/`` and
  yields the same shape. Each natural-sound recording carries multiple
  triggers, each playing a different *sequence* (e.g.
  ``'Sequence 1: 2  1  3  1  4'``) that splices 5 segments from
  ``Stimuli/class<N>/<idx>.mat``.

Response cleanup is applied at the iterator level so per-cell artifacts
(drift, dropouts, motion-induced jumps) are gated before any caller
sees the data. See :func:`prepare_repeats` for the pipeline.
"""
from __future__ import annotations

import os
import re
import warnings
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy import signal as scsig


# ---------------------------------------------------------------------------
# .mat helpers
# ---------------------------------------------------------------------------

def _load_mat(path: str) -> dict:
    """Load a CRCNS-style .mat file with struct fields exposed as attributes."""
    return sio.loadmat(path, squeeze_me=False, struct_as_record=False)


def _unstruct(s):
    """Unwrap a (1, 1) object ndarray holding a single struct, else return as-is."""
    if isinstance(s, np.ndarray) and s.dtype == object and s.size == 1:
        return s.flat[0]
    return s


def _scalar(arr) -> float | int | str:
    """Extract a scalar from a 0-d / 1-element ndarray (handles MATLAB nesting)."""
    if isinstance(arr, np.ndarray):
        return arr.flat[0]
    return arr


def _trace_mv(response_struct) -> Tuple[np.ndarray, float]:
    """Decode a CRCNS ``response`` struct into (mV trace as float64, sf)."""
    trace = np.asarray(response_struct.trace, dtype=np.float64).ravel()
    scale = float(_scalar(response_struct.scale))
    return trace * scale, float(_scalar(response_struct.sf))


# ---------------------------------------------------------------------------
# Zip auto-extract (mirrors the Wingert / AA1 layout)
# ---------------------------------------------------------------------------

# Each entry: (zip name as it lives in `path/`, NERSC mirror sub-path,
# extract-to-subdir relative to `path/`, return-this-subdir relative to `path/`,
# anchor file under the extract-to-subdir whose existence means already-unpacked).
_AC1_ARCHIVES = (
    ("crcns-ac1.zip", "ac-1/crcns-ac1.zip",
     "crcns-ac1", "crcns-ac1/wehr",
     "wehr/Summary.mat"),
    ("crcns-ac1-asari-results-1.zip", "ac-1/crcns-ac1-asari-results-1.zip",
     "crcns-ac1-asari-results-1", "crcns-ac1-asari-results-1/asari-results-1",
     "asari-results-1"),
    ("crcns-ac1-asari-results-2.zip", "ac-1/crcns-ac1-asari-results-2.zip",
     "crcns-ac1-asari-results-2", "crcns-ac1-asari-results-2/asari-results-2",
     "asari-results-2"),
)


def ensure_extracted(path: str) -> Tuple[str, str, str]:
    """Ensure the three AC1 zips are unpacked under ``path``.

    Returns ``(wehr_dir, asari1_dir, asari2_dir)`` — absolute paths to the
    subdirectories the iterators expect.

    Idempotent: if a zip has already been unpacked, it is left alone.
    Raises ``FileNotFoundError`` if a zip is missing and download has
    not been attempted upstream.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"CRCNS-AC1 path does not exist: {path!r}. Pass download=True or "
            f"point `path=` at a directory containing the three CRCNS-AC1 zips."
        )

    out_dirs: List[str] = []
    for zip_name, _nersc_path, extract_to, return_sub, anchor in _AC1_ARCHIVES:
        extract_root = os.path.join(path, extract_to)
        anchor_path = os.path.join(extract_root, anchor)
        if not os.path.exists(anchor_path):
            zip_path = os.path.join(path, zip_name)
            if not os.path.exists(zip_path):
                raise FileNotFoundError(
                    f"Missing both extracted dir ({anchor_path!r}) and source zip "
                    f"({zip_path!r}). Download the CRCNS AC1 archive from "
                    f"https://crcns.org/data-sets/ac/ac-1/about (free account) "
                    f"and place {zip_name!r} under {path!r}, or pass download=True."
                )
            os.makedirs(extract_root, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_root)
        out_dirs.append(os.path.join(path, return_sub))

    return out_dirs[0], out_dirs[1], out_dirs[2]


# ---------------------------------------------------------------------------
# Response cleanup
# ---------------------------------------------------------------------------

def medgauss_detrend(
    x: np.ndarray,
    sf: float,
    *,
    med_ms: float = 100.0,
    gauss_ms: float = 10.0,
) -> np.ndarray:
    """Subtract a slow MedGauss baseline (median then gaussian low-pass).

    Matches the Rançon 2025 detrending recipe carried over from the old
    asari.py: a median filter clips short-time spikes / outliers, then
    a gaussian smooths what remains into a slowly varying baseline that
    captures DC drift. Kernel sizes are given in *milliseconds* so the
    same call scales transparently across the two sub-datasets
    (Wehr sf=4000, Asari sf=10000).
    """
    if x.size == 0:
        return x
    med_size = max(1, int(round(med_ms * sf / 1000.0)))
    gauss_sig = max(1e-3, gauss_ms * sf / 1000.0)
    baseline = gaussian_filter1d(
        median_filter(x, size=med_size, mode="reflect"),
        sigma=gauss_sig,
        mode="reflect",
    )
    return x - baseline


def _repeat_step_fraction(
    x: np.ndarray,
    sf: float,
    *,
    step_mv: float = 30.0,
    step_min_ms: float = 5.0,
) -> float:
    """Fraction of samples sitting on a sustained baseline step ≥ ``step_mv``.

    Defined as samples that, after a ``step_min_ms`` median-smoothing,
    still sit > ``step_mv`` away from the trace median. This catches
    motion / dropout artifacts (large baseline shifts that persist
    for tens of ms) while ignoring spikes (brief excursions that the
    median filter clips). On clean detrended Vm trials this fraction
    is well under 1e-3.
    """
    if x.size < 3:
        return 0.0
    clip_size = max(1, int(round(step_min_ms * sf / 1000.0)))
    if clip_size > 1 and x.size >= clip_size:
        smooth = median_filter(x, size=clip_size, mode="reflect")
    else:
        smooth = x
    deviation = np.abs(smooth - np.median(smooth))
    return float((deviation > step_mv).mean())


def _repeat_dynamic_range_ok(x: np.ndarray, *, abs_mv_max: float = 200.0) -> bool:
    """True if the detrended trace stays inside a reasonable Vm range.

    Whole-cell Vm sits around -60 mV with a few mV of fluctuation. After
    detrending the absolute amplitude should be well under 100 mV; a
    trace exceeding ``abs_mv_max`` (default 200 mV) is almost certainly
    a recording artifact / amplifier saturation episode.
    """
    if x.size == 0:
        return False
    return bool(np.nanmax(np.abs(x)) <= abs_mv_max)


def _crosstrial_consistency(traces: np.ndarray) -> np.ndarray:
    """Per-repeat Pearson r vs. the leave-one-out median across other repeats.

    ``traces``: (R, T). Returns (R,) array of r values in [-1, 1]. With
    R < 3 the test is uninformative (a single comparison gives r=±1
    trivially) and the function returns all ones, deferring rejection
    to the per-repeat tests.
    """
    R, _T = traces.shape
    if R < 3:
        return np.ones(R, dtype=np.float64)

    out = np.empty(R, dtype=np.float64)
    for i in range(R):
        others = np.delete(traces, i, axis=0)
        ref = np.median(others, axis=0)
        a = traces[i] - traces[i].mean()
        b = ref - ref.mean()
        denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-12
        out[i] = float((a * b).sum() / denom)
    return out


@dataclass
class RepeatGating:
    """Tunable thresholds for per-repeat artifact gating.

    Three independent tests run on the detrended trace:

    1. **Dynamic range** — any sample with ``|Vm| > abs_mv_max`` flags
       the repeat. Catches amplifier saturation / hard clipping.
    2. **Sustained step** — fraction of samples sitting on a median-
       smoothed baseline shifted by ``> step_mv`` from the trace median;
       repeat dropped if this fraction exceeds ``max_step_frac``. Catches
       movement / dropout artifacts that survive the MedGauss detrend
       because they're sharper than the 100 ms baseline.
    3. **Cross-trial consistency** — per-repeat Pearson r against the
       leave-one-out median across other repeats; drop if r below
       ``min_xcorr``. The principled test for "this repeat disagrees
       with its peers". Requires R ≥ 3 to be informative.

    Defaults are calibrated against the CRCNS-AC1 traces: clean repeats
    score ``abs_mv_max ≪ 150 mV``, ``step_frac < 1e-4``, ``r > 0.2``;
    bad repeats fail at least one.
    """
    abs_mv_max: float = 150.0     # |trace| ceiling after detrending
    step_mv: float = 30.0         # baseline-step amplitude (mV) considered an artifact
    max_step_frac: float = 5e-3   # repeat dropped if >0.5% of samples sit on the step
    min_xcorr: float = 0.05       # cross-trial Pearson r floor (R>=3 only)


def prepare_repeats(
    raw_repeats: Sequence[np.ndarray],
    sf: float,
    *,
    detrend: bool = True,
    detrend_med_ms: float = 100.0,
    detrend_gauss_ms: float = 10.0,
    gating: Optional[RepeatGating] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """Apply MedGauss detrending + per-repeat artifact gating.

    Parameters
    ----------
    raw_repeats : sequence of 1-D ndarrays
        Membrane-potential traces (mV), each one repeat of the same
        stimulus presentation. Lengths may differ slightly; the gating
        operates on a common-length stack (truncated to the shortest).
    sf : float
        Sampling rate of the raw traces (Hz).
    detrend : bool
        If True (default), subtract the MedGauss baseline before gating.
    detrend_med_ms, detrend_gauss_ms : float
        Median window / gaussian σ in ms. See :func:`medgauss_detrend`.
    gating : RepeatGating, optional
        Per-repeat rejection thresholds (see the dataclass docstring).

    Returns
    -------
    kept : list of 1-D ndarrays
        Cleaned, detrended repeats that survived the gating.
    reasons : list of str
        One entry per *input* repeat: ``"kept"`` or the name of the
        first test it failed (``"jump"`` / ``"range"`` / ``"xcorr"``).
        Useful for downstream provenance / diagnostics.
    """
    gating = gating or RepeatGating()
    if len(raw_repeats) == 0:
        return [], []

    # Detrend per-repeat
    cleaned = [
        medgauss_detrend(np.asarray(r, dtype=np.float64), sf,
                         med_ms=detrend_med_ms, gauss_ms=detrend_gauss_ms)
        if detrend else np.asarray(r, dtype=np.float64)
        for r in raw_repeats
    ]

    R = len(cleaned)
    L = min(int(t.size) for t in cleaned) if cleaned else 0

    # Optional cross-trial consistency check (LOO-median Pearson r)
    if gating.min_xcorr > 0.0 and L > 0 and R >= 3:
        stack = np.stack([t[:L] for t in cleaned])
        xcorr_r = _crosstrial_consistency(stack)
    else:
        xcorr_r = np.ones(R)

    kept: List[np.ndarray] = []
    reasons: List[str] = []
    for i, t in enumerate(cleaned):
        if not _repeat_dynamic_range_ok(t, abs_mv_max=gating.abs_mv_max):
            reasons.append("range")
            continue
        if _repeat_step_fraction(t, sf,
                                 step_mv=gating.step_mv) > gating.max_step_frac:
            reasons.append("step")
            continue
        if xcorr_r[i] < gating.min_xcorr:
            reasons.append("xcorr")
            continue
        kept.append(t)
        reasons.append("kept")

    return kept, reasons


def detect_spikes_psth(
    x: np.ndarray,
    sf: float,
    *,
    detrend_med_ms: float = 100.0,
    detrend_gauss_ms: float = 10.0,
    spk_clip_med_ms: float = 1.0,
    threshold_sigma: float = 2.5,
    smooth_ms: float = 21.0,
) -> np.ndarray:
    """Convert a Vm trace into a Hann-smoothed PSTH proxy.

    Faithful to the Asari 2009 + Rançon 2025 ``signal_type='spikes'``
    recipe: high-pass detrend (MedGauss subtract) → median-filter spike
    clip → threshold at ``threshold_sigma × σ`` → Hann smooth at
    ``smooth_ms``. Output is in arbitrary units proportional to spike
    rate (not a calibrated firing rate).
    """
    y = medgauss_detrend(np.asarray(x, dtype=np.float64), sf,
                         med_ms=detrend_med_ms, gauss_ms=detrend_gauss_ms)
    # High-pass: subtract a short median-filtered version (clips slow waveforms)
    clip_size = max(1, int(round(spk_clip_med_ms * sf / 1000.0)))
    y_hp = y - median_filter(y, size=clip_size, mode="reflect")
    thr = threshold_sigma * float(np.std(y_hp))
    spikes = (y_hp > thr).astype(np.float64)
    # Hann smooth
    win_n = max(3, int(round(smooth_ms * sf / 1000.0)))
    hann = scsig.windows.hann(win_n)
    psth = scsig.convolve(spikes, hann, mode="same", method="direct")
    return psth


# ---------------------------------------------------------------------------
# Time-domain helpers
# ---------------------------------------------------------------------------

def apply_ramp(samples: np.ndarray, sf: float, ramp_ms: float) -> np.ndarray:
    """Apply a cosine-squared onset/offset ramp (the MATLAB MakeRamp).

    The original ramp tapers the first / last ``ramp_ms`` of a segment
    using a quarter-cosine (so ``sample * sin²(π/2 · t/ramp)``). Used at
    every segment boundary in Asari sequences and at the onset/offset of
    each Wehr trial.
    """
    out = np.array(samples, dtype=np.float64, copy=True)
    n_ramp = max(1, int(round(ramp_ms * sf / 1000.0)))
    n_ramp = min(n_ramp, out.size // 2)
    if n_ramp <= 0:
        return out
    t = np.arange(n_ramp) / n_ramp
    ramp = np.sin(0.5 * np.pi * t) ** 2
    out[:n_ramp] *= ramp
    out[-n_ramp:] *= ramp[::-1]
    return out


def bin_response(response: np.ndarray, sf: float, dt_ms: float) -> np.ndarray:
    """Average-pool a 1-D response trace to a coarser ``dt_ms`` grid.

    Truncates to a whole number of bins (no fractional-bin handling —
    the caller has already aligned response and stim time spans via the
    trigger).
    """
    block = max(1, int(round(dt_ms * sf / 1000.0)))
    if response.size < block:
        return np.array([response.mean()] if response.size else [], dtype=np.float64)
    n_bins = response.size // block
    return response[: n_bins * block].reshape(n_bins, block).mean(axis=1)


# ---------------------------------------------------------------------------
# Wehr ingest
# ---------------------------------------------------------------------------

# Trigger types we DO want for natural-sound analysis. The 'nauralsound'
# (sic) spelling is a typo that survived into the CRCNS release; the
# 'naturalsound' spelling is also used.
_WEHR_NATURAL_TRIGGER_TYPES = ("naturalsound", "nauralsound", "natural")

# Subdirectories under wehr/Stimuli/ that hold natural-sound waveforms.
_WEHR_STIM_DIRS = ("fragments", "category1", "category2", "category3")


def _build_wehr_stim_index(stims_root: str) -> Dict[str, Tuple[str, int, str]]:
    """Map ``stimulus.param.description`` → (category, idx, .mat path).

    The trigger only carries the description string, so we precompute a
    reverse lookup across every fragment / category .mat file in
    ``wehr/Stimuli/``. Description-to-file is 1:1 in the released
    archive.
    """
    index: Dict[str, Tuple[str, int, str]] = {}
    for sub in _WEHR_STIM_DIRS:
        sub_path = os.path.join(stims_root, sub)
        if not os.path.isdir(sub_path):
            continue
        for fname in sorted(os.listdir(sub_path)):
            if not fname.endswith(".mat"):
                continue
            try:
                idx = int(os.path.splitext(fname)[0])
            except ValueError:
                continue
            full = os.path.join(sub_path, fname)
            m = _load_mat(full)
            inner = _unstruct(_unstruct(m["stimulus"]).param)
            descr = str(_scalar(inner.description))
            if descr in index:
                # The categoryN files all have desc like
                # 'natural sound stream (0-1.6 kHz), No.1' — unique across
                # categories — but be defensive in case of collisions.
                warnings.warn(
                    f"Duplicate Wehr stim description {descr!r}: "
                    f"{index[descr][2]} vs {full}. Keeping first.",
                    RuntimeWarning,
                )
                continue
            index[descr] = (sub, idx, full)
    return index


def _load_wehr_stim_waveform(stim_record: Tuple[str, int, str]) -> Tuple[np.ndarray, float]:
    """Load a Wehr stimulus .mat → (waveform, sf)."""
    _cat, _idx, path = stim_record
    m = _load_mat(path)
    stim_obj = _unstruct(m["stimulus"])
    samples = np.asarray(stim_obj.samples, dtype=np.float64).ravel()
    inner = _unstruct(stim_obj.param)
    sf = float(_scalar(inner.sf))
    return samples, sf


@dataclass
class StimRecord:
    """Per-stim record yielded by the ingest iterators."""
    key: Tuple                  # dedup key, e.g. ('wehr', 'fragments', 5)
    waveform: np.ndarray        # 1-D float64 at sf_stim
    sf_stim: float
    duration_ms: float
    raw_repeats: List[np.ndarray]   # list of 1-D mV traces at sf_resp
    sf_resp: float
    meta: Dict


@dataclass
class CellRecord:
    """Per-cell record yielded by the ingest iterators."""
    meta: Dict
    stims: List[StimRecord] = field(default_factory=list)


def iterate_wehr_cells(wehr_root: str) -> Iterator[CellRecord]:
    """Yield one ``CellRecord`` per Wehr session directory.

    Each session = one cell. Within a session, trial files are grouped
    by ``trigger.param.description``; multiple trials with the same
    description become a per-stim repeat axis.

    The dataset-specific artifacts the legacy ``wehr.py`` hand-coded for
    cell index 12 (out-of-distribution response #11, mid-trace drift on
    response #10) are now caught by the artifact gating in
    :func:`prepare_repeats` — see ``RepeatGating`` — and don't need a
    hand-rolled carve-out anymore.
    """
    results_dir = os.path.join(wehr_root, "Results")
    stims_dir = os.path.join(wehr_root, "Stimuli")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"wehr/Results not found at {results_dir!r}")
    if not os.path.isdir(stims_dir):
        raise FileNotFoundError(f"wehr/Stimuli not found at {stims_dir!r}")

    stim_index = _build_wehr_stim_index(stims_dir)

    session_names = sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d[:4].isdigit()
    )

    for cell_idx, session in enumerate(session_names):
        session_dir = os.path.join(results_dir, session)
        date, _animal, penet = session.split("-")
        # Collect trial files (one trigger each), grouped by stim description.
        stim_groups: Dict[str, List[Tuple[str, dict]]] = {}
        # also stash a few per-trial-file metadata items for the cell-level dict
        per_file_meta: List[Dict] = []
        for fname in sorted(os.listdir(session_dir)):
            if not fname.endswith(".mat"):
                continue
            fpath = os.path.join(session_dir, fname)
            try:
                m = _load_mat(fpath)
            except Exception as exc:
                warnings.warn(f"failed to load {fpath}: {exc}", RuntimeWarning)
                continue
            trigs = m.get("triggers")
            if trigs is None or trigs.size == 0:
                continue
            # one trigger per Wehr trial file
            t = _unstruct(trigs.flat[0])
            if not hasattr(t, "type") or not hasattr(t, "param"):
                continue
            ttype = str(_scalar(t.type)).lower().strip()
            if ttype not in _WEHR_NATURAL_TRIGGER_TYPES:
                continue
            param_struct = _unstruct(t.param)
            if not hasattr(param_struct, "description") or not hasattr(param_struct, "duration"):
                continue
            descr = str(_scalar(param_struct.description))
            if descr not in stim_index:
                warnings.warn(
                    f"{fname}: stim description {descr!r} not in stimulus index; skipped.",
                    RuntimeWarning,
                )
                continue
            stim_groups.setdefault(descr, []).append((fpath, m))

        if not stim_groups:
            continue

        cell_meta = {
            "experimenter": "wehr",
            "session": session,
            "animal_id": "mw",
            "penetration": int(penet),
            "date": date,
            "site": "A1",
            "recording_type": "whole-cell",
            "species": "rat",
            "_wehr_cell_idx": cell_idx,
        }

        stim_records: List[StimRecord] = []
        for descr, file_records in stim_groups.items():
            cat, idx, _spath = stim_index[descr]

            # Slice the Vm trace for each trial = each repeat.
            raw_repeats: List[np.ndarray] = []
            sf_resp = None
            duration_ms = None
            for fpath, m in file_records:
                resp = _unstruct(m["response"])
                trace_mv, sf = _trace_mv(resp)
                t = _unstruct(m["triggers"].flat[0])
                pstruct = _unstruct(t.param)
                trig_time = int(_scalar(t.time)) - 1  # MATLAB 1-indexed
                trig_dur_ms = float(_scalar(pstruct.duration))
                n_samples = int(round(trig_dur_ms * sf / 1000.0))
                lo = max(0, trig_time)
                hi = min(trace_mv.size, lo + n_samples)
                segment = trace_mv[lo:hi]
                raw_repeats.append(segment)
                if sf_resp is None:
                    sf_resp = sf
                    duration_ms = trig_dur_ms

            waveform, sf_stim = _load_wehr_stim_waveform(stim_index[descr])
            stim_records.append(StimRecord(
                key=("wehr", cat, idx),
                waveform=waveform,
                sf_stim=sf_stim,
                duration_ms=duration_ms or (waveform.size / sf_stim * 1000.0),
                raw_repeats=raw_repeats,
                sf_resp=sf_resp,
                meta={
                    "experimenter": "wehr",
                    "category": cat,
                    "idx": idx,
                    "description": descr,
                    "duration_s": (duration_ms or 0.0) / 1000.0,
                },
            ))

        yield CellRecord(meta=cell_meta, stims=stim_records)


# ---------------------------------------------------------------------------
# Asari ingest
# ---------------------------------------------------------------------------

# Trigger types of interest (natural-sound sequences only — tones / TORC
# / DMR / MHT / MCN / random-chords are out of v1 scope).
_ASARI_NATURAL_TRIGGER_TYPES = ("naturalsound", "natural sound")

# Description regex: '(class3)' / '(class5)' etc., embedded in the param.ID.description
_RX_CLASS = re.compile(r"\bclass\s*([1-6])\b", re.IGNORECASE)
# Sequence parsing: 'Sequence 1: 2  1  3  1  4' (variable whitespace, 1+ ints)
_RX_SEQ = re.compile(r"Sequence\s+\d+\s*:\s*((?:\d+\s+)*\d+)", re.IGNORECASE)


def _asari_seq_segments(description: str) -> Optional[List[int]]:
    """Parse a sequence description into the list of segment indices, or None."""
    m = _RX_SEQ.search(description)
    if not m:
        return None
    return [int(s) for s in m.group(1).split()]


def _load_asari_stim_by_relpath(stims_root: str, rel_path: str) -> Tuple[np.ndarray, float]:
    """Load an Asari stimulus from ``stims_root/<rel_path>`` → (waveform, sf).

    ``rel_path`` is a posix-style relative path like ``'class6/3.mat'`` or
    ``'class3/3l.mat'`` (suffixed variants are present in the archive).
    """
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    fpath = os.path.join(stims_root, *rel_path.split("/"))
    m = _load_mat(fpath)
    stim_obj = _unstruct(m["stimulus"])
    # The released archive uses both 'sample' (asari) and 'samples' (wehr).
    if hasattr(stim_obj, "sample"):
        samples = np.asarray(stim_obj.sample, dtype=np.float64).ravel()
    else:
        samples = np.asarray(stim_obj.samples, dtype=np.float64).ravel()
    inner = _unstruct(stim_obj.param)
    sf = float(_scalar(inner.sf))
    return samples, sf


def _build_asari_stim_filemap(param_struct) -> Dict[int, str]:
    """Build ``{1-indexed stim index: rel_path}`` from ``param.stimulus``.

    The recording's ``param.stimulus`` is a (1, K) struct array. Entry
    ``param.stimulus(i)`` is the i-th stimulus in this session, with a
    ``.file`` field like ``'class6/3.mat'``. The trigger's sequence
    description refers to these by 1-indexed position, NOT by the
    suffix-bearing filename — so we have to read the lookup explicitly.
    """
    if not hasattr(param_struct, "stimulus"):
        return {}
    stim_arr = param_struct.stimulus
    out: Dict[int, str] = {}
    for k in range(stim_arr.size):
        s = stim_arr.flat[k]
        if hasattr(s, "file"):
            file_val = s.file
            if isinstance(file_val, np.ndarray) and file_val.size > 0:
                out[k + 1] = str(file_val.flatten()[0])
    return out


def _splice_asari_sequence_by_paths(
    rel_paths: Sequence[str],
    stims_root: str,
    *,
    segment_ramp_ms: float = 5.0,
) -> Tuple[np.ndarray, float]:
    """Reconstruct an Asari sequence waveform by concatenating segments.

    Each segment is identified by its ``Stimuli/`` relative path (e.g.
    ``'class6/3.mat'``) — the canonical identity the Asari pipeline
    uses internally. A 5 ms cosine-squared ramp is applied to the
    onset + offset of each segment (Asari 2009 Methods: ramps are
    applied at every segment boundary, even with no interstimulus
    interval). All segments share the same sampling rate in the
    released archive (97656 Hz).
    """
    pieces = []
    sf_common = None
    for rel_path in rel_paths:
        wav, sf = _load_asari_stim_by_relpath(stims_root, rel_path)
        if sf_common is None:
            sf_common = sf
        elif abs(sf - sf_common) > 1.0:
            raise RuntimeError(
                f"Asari sequence mixes sf: {sf} vs {sf_common}"
            )
        pieces.append(apply_ramp(wav, sf, segment_ramp_ms))
    if not pieces:
        return np.array([], dtype=np.float64), 1.0
    return np.concatenate(pieces), float(sf_common)


def iterate_asari_cells(
    asari_roots: Sequence[str],
    *,
    sites: Sequence[str] = ("A1", "MGB"),
    stims_root: Optional[str] = None,
) -> Iterator[CellRecord]:
    """Yield one ``CellRecord`` per Asari session directory.

    ``asari_roots`` accepts one or both of ``asari-results-{1,2}/``.
    ``stims_root`` defaults to ``<asari_roots[0]>/../crcns-ac1/asari/Stimuli``
    — the convention from :func:`ensure_extracted`. Override only if
    you've laid the data out differently.
    """
    sites = tuple(s.upper() for s in sites)

    for root in asari_roots:
        if not os.path.isdir(root):
            warnings.warn(f"asari root {root!r} does not exist; skipping.", RuntimeWarning)
            continue
        if stims_root is None:
            # root is e.g. <path>/crcns-ac1-asari-results-1/asari-results-1;
            # we want   <path>/crcns-ac1/asari/Stimuli.
            ac1_root = os.path.dirname(os.path.dirname(root))
            stims_guess = os.path.join(ac1_root, "crcns-ac1", "asari", "Stimuli")
        else:
            stims_guess = stims_root

        if not os.path.isdir(stims_guess):
            raise FileNotFoundError(
                f"asari/Stimuli not found at {stims_guess!r}. Pass stims_root="
                f"explicitly, or extract crcns-ac1.zip first."
            )

        for session in sorted(os.listdir(root)):
            session_dir = os.path.join(root, session)
            if not os.path.isdir(session_dir):
                continue
            # Session id format: yyyymmdd-ha[1-4]-NNN
            parts = session.split("-")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            date, animal_id, penet = parts[0], parts[1], parts[2]

            # Walk recording files; gather natural-sound triggers grouped
            # by the resolved (segment-files tuple). We extract the response
            # slice IMMEDIATELY (so the full mat dict can be GC'd before
            # moving to the next file) — Asari recordings are ~10 MB each
            # and accumulating them would easily OOM a 32 GB box.
            #
            # The lookup ``param.stimulus[i].file`` gives the canonical
            # stimulus identity (e.g. ``'class6/3.mat'``); the trigger's
            # description provides the integer ordering. We resolve the
            # full filename tuple per-trigger and key the group on that —
            # NOT on the ``param.ID.description`` class<N> tag, which is
            # missing for most A1 sessions (only ~10 of 51 sessions
            # actually populate that field).
            sequence_groups: Dict[Tuple[str, ...], List[Tuple[np.ndarray, float, float, Sequence[int]]]] = {}
            cell_site: Optional[str] = None
            rec_type: Optional[str] = None

            for fname in sorted(os.listdir(session_dir)):
                if not fname.endswith(".mat"):
                    continue
                fpath = os.path.join(session_dir, fname)
                try:
                    m = _load_mat(fpath)
                except Exception as exc:
                    warnings.warn(f"failed to load {fpath}: {exc}", RuntimeWarning)
                    continue
                param = _unstruct(m.get("param"))
                if param is None:
                    continue

                # Build the per-recording stim-index → file-path lookup table.
                # No filemap → can't decode sequences → skip the file.
                file_map = _build_asari_stim_filemap(param)
                if not file_map:
                    continue

                # Site classification: 'cortex' substring in recording.site → A1,
                # else MGB. (Matches the legacy asari.py heuristic.)
                if hasattr(param, "recording"):
                    rstruct = _unstruct(param.recording)
                    site_str = str(_scalar(rstruct.site)) if hasattr(rstruct, "site") else ""
                    site = "A1" if "cortex" in site_str.lower() else "MGB"
                    if cell_site is None:
                        cell_site = site
                    rtype = str(_scalar(rstruct.type)) if hasattr(rstruct, "type") else ""
                    if rec_type is None:
                        rec_type = rtype
                else:
                    site = cell_site or "A1"

                trigs = m.get("triggers")
                if trigs is None or trigs.size == 0:
                    continue
                # Decode the full response trace once per file (sf is shared).
                resp_struct = _unstruct(m["response"])
                trace_mv, sf_resp = _trace_mv(resp_struct)
                # Walk every natural-sound trigger in the file and stash its
                # response slice.
                for j in range(trigs.size):
                    t = _unstruct(trigs.flat[j])
                    if not hasattr(t, "type") or not hasattr(t, "param"):
                        continue
                    ttype = str(_scalar(t.type)).lower().strip()
                    if ttype not in _ASARI_NATURAL_TRIGGER_TYPES:
                        continue
                    pstruct = _unstruct(t.param)
                    if not hasattr(pstruct, "description") or not hasattr(pstruct, "duration"):
                        continue
                    seq_descr = str(_scalar(pstruct.description))
                    segments = _asari_seq_segments(seq_descr)
                    if segments is None:
                        continue
                    # Resolve segments → file paths via the filemap. Some
                    # triggers reference indices that aren't in this
                    # recording's filemap (rare; usually indicates a
                    # mid-session config change). Skip those triggers.
                    try:
                        resolved = tuple(file_map[int(s)] for s in segments)
                    except KeyError:
                        warnings.warn(
                            f"{session}/{fname}: trigger references stim idx not in "
                            f"this recording's param.stimulus; skipping.",
                            RuntimeWarning,
                        )
                        continue
                    trig_time = int(_scalar(t.time)) - 1
                    trig_dur_ms = float(_scalar(pstruct.duration))
                    n_samples = int(round(trig_dur_ms * sf_resp / 1000.0))
                    lo = max(0, trig_time)
                    hi = min(trace_mv.size, lo + n_samples)
                    # IMPORTANT: copy() so we don't keep a view that pins the
                    # full trace alive across iterations.
                    resp_slice = trace_mv[lo:hi].copy()
                    canon_descr = " ".join(seq_descr.split())
                    sequence_groups.setdefault(resolved, []).append(
                        (resp_slice, sf_resp, trig_dur_ms, tuple(segments), canon_descr)
                    )
                del m, trace_mv  # explicit hint for the GC between files

            if cell_site is None or cell_site not in sites:
                continue
            if not sequence_groups:
                continue

            cell_meta = {
                "experimenter": "asari",
                "session": session,
                "animal_id": animal_id,
                "penetration": int(penet) if penet.isdigit() else penet,
                "date": date,
                "site": cell_site,
                "recording_type": (rec_type or "").lower().replace(" ", "-"),
                "species": "rat",
            }

            stim_records: List[StimRecord] = []
            for resolved_files, occurrences in sequence_groups.items():
                # Splice the actual waveform from the resolved file paths.
                try:
                    waveform, sf_stim = _splice_asari_sequence_by_paths(
                        resolved_files, stims_guess,
                    )
                except (KeyError, FileNotFoundError) as exc:
                    warnings.warn(
                        f"{session}: cannot splice {resolved_files!r}: {exc}",
                        RuntimeWarning,
                    )
                    continue

                # Derive class_n from the file paths (e.g. 'class6/3.mat' → 6).
                # Mixed-class sequences are rare; use the first segment's class.
                class_n: Optional[int] = None
                if resolved_files:
                    m_cls = re.match(r"class(\d+)/", resolved_files[0])
                    if m_cls:
                        class_n = int(m_cls.group(1))
                raw_repeats = [r for (r, _sf, _dur, _seg, _d) in occurrences]
                sf_resp_common = occurrences[0][1]
                duration_ms = occurrences[0][2]
                segments_int = occurrences[0][3]  # the segment-int list (1-indexed)
                sequence_descr = occurrences[0][4]  # 'Sequence N: a b c d e'

                stim_records.append(StimRecord(
                    key=("asari", resolved_files),
                    waveform=waveform,
                    sf_stim=sf_stim,
                    duration_ms=duration_ms,
                    raw_repeats=raw_repeats,
                    sf_resp=sf_resp_common,
                    meta={
                        "experimenter": "asari",
                        "category": f"class{class_n}" if class_n is not None else "mixed",
                        "class_n": class_n,
                        "segments": tuple(segments_int),
                        "segment_files": resolved_files,
                        "sequence": sequence_descr,
                        "description": sequence_descr,
                        "duration_s": duration_ms / 1000.0,
                    },
                ))

            yield CellRecord(meta=cell_meta, stims=stim_records)
