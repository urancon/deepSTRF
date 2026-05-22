"""Downer 2025 / Ahmed 2025 squirrel-monkey auditory cortex dataset.

Multi-unit (channel-level threshold-crossing) spike recordings from auditory
cortex of three squirrel monkeys, presented with TIMIT sentences and monkey
vocalizations. Two stimulus classes are loaded separately via the
``stimuli={'timit', 'mvocs'}`` constructor arg.

Source: Zenodo DOI ``10.5281/zenodo.16175377`` (29 GB archive).

Citations:
- Ahmed B, Downer JD, Malone BJ, Makin JG (2025). *Deep neural networks
  explain spiking activity in auditory cortex.* PLoS Computational Biology
  21(8):e1013334. https://doi.org/10.1371/journal.pcbi.1013334
- Downer JD, Bigelow J, Runfeldt M, Malone BJ (2021). *Temporally precise
  population coding of dynamic sounds by auditory cortex.* J Neurophysiol.

Each ``*_MUspk.mat`` file holds threshold-crossing spike times for one
channel; spike-sorting was not feasible, so each channel is a single
"multi-unit" (= one "neuron" in deepSTRF parlance). The published count
is 1718 multi-units across 41 sessions (Ahmed 2025, p4–5).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from tqdm import tqdm

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.utils.data_download import default_cache_dir


# Public Zenodo record. https://doi.org/10.5281/zenodo.16175377
DOWNER_ZENODO_RECORD = 16175377


# Fine-area assignments are encoded as YAML *comments* in
# ``sessions_metadata.yml`` (under ``area_wise_sessions:``), not as YAML
# data. We transcribe them once here and use them as the source of truth
# for the ``area`` ``nrn_meta`` key.
#
# Mapping: session_id (str) -> (area_group, fine_area).
#   area_group ∈ {'core', 'non-primary'}
#   fine_area  ∈ {'A1', 'R', 'ML', 'AL', 'CL', 'CPB', 'RPB'}  (None if unknown)
#
# Sessions present in the Zenodo upload but absent from these comments
# fall back to ``area_group`` from the YAML data with ``fine_area=None``.
_FINE_AREA_BY_SESSION: dict[str, tuple[str, str]] = {
    # Core / A1
    "180627": ("core", "A1"), "180719": ("core", "A1"), "180810": ("core", "A1"),
    "180814": ("core", "A1"), "180720": ("core", "A1"), "180731": ("core", "A1"),
    "190604": ("core", "A1"), "190606": ("core", "A1"), "190726": ("core", "A1"),
    "190801": ("core", "A1"), "191209": ("core", "A1"), "200313": ("core", "A1"),
    "191113": ("core", "A1"), "191125": ("core", "A1"), "191206": ("core", "A1"),
    "200206": ("core", "A1"), "200207": ("core", "A1"), "180808": ("core", "A1"),
    # Core / R
    "180807": ("core", "R"), "200213": ("core", "R"), "180501": ("core", "R"),
    # Non-primary / belt / CL
    "191219": ("non-primary", "CL"),
    # Non-primary / belt / ML
    "180613": ("non-primary", "ML"), "190703": ("non-primary", "ML"),
    "200219": ("non-primary", "ML"), "191115": ("non-primary", "ML"),
    "200205": ("non-primary", "ML"),
    # Non-primary / belt / AL
    "180622": ("non-primary", "AL"), "191211": ("non-primary", "AL"),
    "180413": ("non-primary", "AL"), "180420": ("non-primary", "AL"),
    # Non-primary / parabelt / CPB
    "180724": ("non-primary", "CPB"), "200318": ("non-primary", "CPB"),
    "191121": ("non-primary", "CPB"),
    # Non-primary / parabelt / RPB
    "180502": ("non-primary", "RPB"), "180728": ("non-primary", "RPB"),
    "190605": ("non-primary", "RPB"), "191210": ("non-primary", "RPB"),
    "180717": ("non-primary", "RPB"), "180730": ("non-primary", "RPB"),
    "200212": ("non-primary", "RPB"),
}


def _parse_sessions_metadata(path: Union[str, Path]) -> dict:
    """Parse ``sessions_metadata.yml`` from a Downer-style dataset root.

    Returns a dict::

        {
          'hemisphere':    {session_id: 'LH'|'RH'},
          'animal':        {session_id: 'b'|'c'|'f'},
          'area_group':    {session_id: 'core'|'non-primary'},
          'fine_area':     {session_id: 'A1'|'R'|'ML'|'AL'|'CL'|'CPB'|'RPB'|None},
          'coord':         {session_id: (x, y)},
          'bad':           set(session_id),
          'n_reps_canonical': {'timit': 11, 'mvocs': 15},
        }
    """
    yml_path = Path(path) / "sessions_metadata.yml"
    with open(yml_path) as fh:
        m = yaml.safe_load(fh)

    hemisphere: dict[str, str] = {}
    animal: dict[str, str] = {}
    for key in ("c_RH_sessions", "b_RH_sessions", "f_RH_sessions", "c_LH_sessions"):
        animal_id, hemi = key[0], key[2:4]
        for sid in m[key]:
            sid = str(sid)
            hemisphere[sid] = hemi
            animal[sid] = animal_id

    area_group: dict[str, str] = {}
    for grp in ("core", "non-primary"):
        for sid in m["area_wise_sessions"][grp]:
            area_group[str(sid)] = grp

    fine_area = {sid: lbl for sid, (_, lbl) in _FINE_AREA_BY_SESSION.items()}

    coord = {str(sid): tuple(xy) for sid, xy in m["session_coordinates"].items()}
    bad = {str(sid) for sid in m["bad_sessions"]}

    return {
        "hemisphere": hemisphere,
        "animal": animal,
        "area_group": area_group,
        "fine_area": fine_area,
        "coord": coord,
        "bad": bad,
        "n_reps_canonical": dict(m["stim_wise_num_repeats"]),
    }


# ``Ch49``, ``Ch5p``, ``Ch10s2`` -- digits then an optional alphanumeric suffix
# (annotates a re-mounted / second-set recording on the same channel number).
_CHAN_RE = re.compile(r"^Ch(\d+)([A-Za-z][A-Za-z0-9]*)?$")


def _parse_channel_filename(fname: str) -> Optional[tuple[str, str, int, str]]:
    """Parse ``<animal>_<session>_Ch<N>[suffix]_MUspk.mat`` -> (animal, session, ch_int, suffix).

    ``suffix`` is the trailing letter/digit annotation after the channel
    number (e.g. ``'p'``, ``'s2'``), or the empty string. Returns ``None``
    for non-MUspk files (e.g. ``TRIALINFO.mat``).
    """
    if "MUspk" not in fname or not fname.endswith("_MUspk.mat"):
        return None
    base = fname[: -len("_MUspk.mat")]
    parts = base.split("_")
    if len(parts) < 3:
        return None
    animal, session, chan_token = parts[0], parts[1], parts[2]
    m = _CHAN_RE.match(chan_token)
    if m is None:
        return None
    ch_int = int(m.group(1))
    suffix = m.group(2) or ""
    return animal, session, ch_int, suffix


def _enumerate_neurons(
    path: Union[str, Path],
    meta: dict,
    *,
    animals: Union[str, Sequence[str]] = "all",
    areas: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Walk ``sessions/<id>/*_MUspk.mat`` and build the ``nrn_meta`` list.

    Filters (all applied at construction time):
        animals  -- 'all' or iterable of 'b'/'c'/'f'.
        areas    -- ('core',) or ('non-primary',) or fine labels
                    ('A1','R','ML','AL','CL','CPB','RPB'); None == no filter.
        sessions -- iterable of session ids (str), or None.

    Each entry has keys: cell_id, session_id, animal_id, hemisphere,
    area_group, area, channel, n_channels_in_session, coord_x, coord_y,
    recording_type.
    """
    root = Path(path) / "sessions"
    if not root.is_dir():
        raise FileNotFoundError(f"Expected sessions directory: {root}")

    if animals == "all":
        animal_whitelist: Optional[set[str]] = None
    else:
        animal_whitelist = set(animals)

    if sessions is None:
        session_whitelist: Optional[set[str]] = None
    else:
        session_whitelist = set(sessions)

    # 'primary' is an Ahmed-2025-style alias for the YAML's 'core' group.
    area_whitelist: Optional[set[str]]
    if areas is None:
        area_whitelist = None
    else:
        area_whitelist = {("core" if a == "primary" else a) for a in areas}

    nrn_meta: list[dict] = []
    for session_id in sorted(os.listdir(root)):
        sess_dir = root / session_id
        if not sess_dir.is_dir():
            continue
        if session_whitelist is not None and session_id not in session_whitelist:
            continue
        if session_id in meta["bad"]:
            # Defensive: bad sessions are absent from the public release, but
            # honour the YAML flag if a private copy still has them.
            continue

        chan_files = sorted(f for f in os.listdir(sess_dir) if "MUspk" in f)
        n_chan = len(chan_files)

        sess_animal = meta["animal"].get(session_id)
        sess_hemi = meta["hemisphere"].get(session_id)
        sess_area_grp = meta["area_group"].get(session_id)
        sess_area = meta["fine_area"].get(session_id)
        sess_coord = meta["coord"].get(session_id, (None, None))

        if animal_whitelist is not None and sess_animal not in animal_whitelist:
            continue
        if area_whitelist is not None and not (
            sess_area_grp in area_whitelist or sess_area in area_whitelist
        ):
            continue

        for fname in chan_files:
            parsed = _parse_channel_filename(fname)
            if parsed is None:
                continue
            animal_pref, parsed_sid, ch_int, ch_suffix = parsed
            # cell_id keeps the on-disk channel token verbatim so suffix
            # variants (Ch8p, Ch10s2) stay distinct.
            cell_id = f"{animal_pref}_{parsed_sid}_Ch{ch_int}{ch_suffix}"
            nrn_meta.append(
                {
                    "cell_id": cell_id,
                    "session_id": parsed_sid,
                    "animal_id": animal_pref,
                    "hemisphere": sess_hemi,
                    "area_group": sess_area_grp,
                    "area": sess_area,
                    "channel": ch_int,
                    "channel_suffix": ch_suffix or None,
                    "n_channels_in_session": n_chan,
                    "coord_x": sess_coord[0],
                    "coord_y": sess_coord[1],
                    "recording_type": "multi-unit",
                }
            )

    return nrn_meta


def _load_timit_stims(path: Union[str, Path]) -> list[dict]:
    """Load the 499 TIMIT sentences from ``stimuli/out_sentence_details_timit_all_loudness.mat``.

    Returns a list of dicts (one per ``sentdet`` entry) with keys
    ``name``, ``stim_id``, ``sound`` (np.ndarray, 16 kHz mono),
    ``soundf``, ``duration_s``, ``befaft_s``. The packaged ``aud``
    cochleagram is *not* loaded — we recompute mel from ``sound``
    using the deepSTRF audio pipeline.
    """
    mat = sio.loadmat(
        Path(path) / "stimuli" / "out_sentence_details_timit_all_loudness.mat",
        squeeze_me=True, struct_as_record=False, variable_names=["sentdet"],
    )
    sd = mat["sentdet"]
    out: list[dict] = []
    for s in sd:
        sound = np.asarray(s.sound, dtype=np.float32).reshape(-1)
        soundf = int(np.asarray(s.soundf).item())
        dur = float(np.asarray(s.duration).item())
        ba = np.asarray(s.befaft)
        befaft = tuple(float(x) for x in ba.ravel()[:2]) if ba.size else (0.0, 0.0)
        out.append({
            "name": str(s.name),
            "stim_id": int(np.asarray(s.sentId).item()),
            "sound": sound,
            "soundf": soundf,
            "duration_s": dur,
            "befaft_s": befaft,
        })
    return out


def _load_mvocs_stims(path: Union[str, Path]) -> tuple[list[dict], dict[int, int]]:
    """Load the 303 unique monkey vocalizations from the concat WAV.

    The release ships ``stimuli/MonkVocs_15Blocks.wav`` (41 kHz stereo,
    ~28 min) with the play-order in ``SqMoPhys_MVOCStimcodes.mat``
    (``mVocsStimCodes`` + ``mVocsStimOnTimes``, 780 slots). Each voc ID
    (1..303) appears in multiple slots; we take **the first occurrence**
    as the canonical waveform and **the minimum inter-onset interval**
    across that ID's occurrences as the canonical duration (which excludes
    the variable inter-stim silence that follows each voc).

    Returns
    -------
    entries : list of dict
        One dict per unique voc ID, sorted by ``stim_id``, with the same
        shape as ``_load_timit_stims`` so the downstream pipeline is shared:
        ``name``, ``stim_id``, ``sound`` (np.float32 mono at native 41 kHz),
        ``soundf``, ``duration_s``, ``befaft_s = (0.0, 0.0)``.
    wav_rep_counts : dict
        ``stim_id -> rep_count_in_canonical_wav``. The Ahmed 2025 paper's
        "test" subset of 11 vocs corresponds to the IDs whose
        ``rep_count_in_canonical_wav == 15`` (see the WAV's design).
    """
    import soundfile as sf  # local import — already a runtime dep

    stimuli_dir = Path(path) / "stimuli"
    code_mat = sio.loadmat(
        stimuli_dir / "SqMoPhys_MVOCStimcodes.mat",
        squeeze_me=True, struct_as_record=False,
    )
    codes = np.asarray(code_mat["mVocsStimCodes"], dtype=np.int64)
    onsets = np.asarray(code_mat["mVocsStimOnTimes"], dtype=np.float64)

    wav_path = stimuli_dir / "MonkVocs_15Blocks.wav"
    wav_info = sf.info(str(wav_path))
    total_dur = wav_info.duration
    sr = int(wav_info.samplerate)

    # Per-slot duration: next-onset gap (or WAV-end gap for the last slot).
    slot_dur = np.empty_like(onsets)
    slot_dur[:-1] = np.diff(onsets)
    slot_dur[-1] = total_dur - onsets[-1]

    # Group slot durations by voc ID; per-ID canonical duration = min.
    durs_by_id: dict[int, list[float]] = {}
    first_onset_by_id: dict[int, float] = {}
    for code, on, d in zip(codes, onsets, slot_dur):
        cid = int(code)
        durs_by_id.setdefault(cid, []).append(float(d))
        if cid not in first_onset_by_id:
            first_onset_by_id[cid] = float(on)

    out: list[dict] = []
    wav_rep_counts: dict[int, int] = {}
    for cid in sorted(durs_by_id):
        canon_dur = float(min(durs_by_id[cid]))
        on0 = first_onset_by_id[cid]
        wav_rep_counts[cid] = len(durs_by_id[cid])
        # Read the snippet for this voc (any occurrence works -- they
        # all play the same recorded waveform; we use the first).
        start_frame = int(round(on0 * sr))
        stop_frame = start_frame + int(round(canon_dur * sr))
        snippet, snippet_sr = sf.read(
            str(wav_path), start=start_frame, stop=stop_frame, dtype="float32",
        )
        assert snippet_sr == sr
        if snippet.ndim == 2:
            snippet = snippet.mean(axis=1)  # stereo -> mono
        out.append({
            "name": f"mvoc_{cid:03d}",
            "stim_id": cid,
            "sound": snippet.astype(np.float32),
            "soundf": sr,
            "duration_s": canon_dur,
            "befaft_s": (0.0, 0.0),
        })
    return out, wav_rep_counts


def _load_session_trial(channel_file: Union[str, Path]) -> dict:
    """Load just the ``trial`` struct from a session's first MUspk file.

    All channels within a session share the same trial struct (verified
    on session 180501 — plain Ch and Chp variants have identical
    ``stimon`` and ``*Stimcode`` arrays).
    """
    m = sio.loadmat(channel_file, squeeze_me=True, struct_as_record=False,
                    variable_names=["trial"])
    t = m["trial"]
    return {
        "stimon": np.asarray(t.stimon, dtype=np.float64),
        "timitStimcode": np.asarray(t.timitStimcode, dtype=np.int64),
        "mVocStimcode": np.asarray(t.mVocStimcode, dtype=np.int64),
    }


def _load_spike_times(channel_file: Union[str, Path]) -> np.ndarray:
    """Load only ``spike.spktimes`` from a MUspk file (skipping the large events matrix)."""
    m = sio.loadmat(channel_file, squeeze_me=True, struct_as_record=False,
                    variable_names=["spike"])
    return np.asarray(m["spike"].spktimes, dtype=np.float64).reshape(-1)


def _bin_spikes_per_rep(
    spktimes: np.ndarray, onsets: np.ndarray, T: int, dt_s: float,
) -> torch.Tensor:
    """Bin per-rep spike counts into a ``(R, T)`` tensor.

    ``onsets`` is the array of stimulus onset times (s) for the R reps.
    Bin edges: ``[onset + i*dt_s, onset + (i+1)*dt_s)`` for i = 0..T-1.
    """
    R = onsets.shape[0]
    out = np.zeros((R, T), dtype=np.float32)
    edges = np.linspace(0.0, T * dt_s, T + 1)
    for r, t0 in enumerate(onsets):
        rel = spktimes - t0
        # only spikes within the stim window matter
        rel = rel[(rel >= 0.0) & (rel < T * dt_s)]
        if rel.size:
            out[r], _ = np.histogram(rel, bins=edges)
    return torch.from_numpy(out)


def _discover_high_rep_set(
    per_session_codes: list[np.ndarray], min_reps: int,
) -> set[int]:
    """Return the set of stim IDs that get >= min_reps reps in at least one session.

    Used to identify the canonical "test" subset (TIMIT: min_reps=11,
    mVocs: min_reps=15) without hard-coding stim IDs — robust if the
    upstream upload ever re-shuffles the ordering.
    """
    high: set[int] = set()
    for codes in per_session_codes:
        codes = codes[codes > 0]
        if codes.size == 0:
            continue
        uniq, counts = np.unique(codes, return_counts=True)
        high.update(int(u) for u, c in zip(uniq, counts) if c >= min_reps)
    return high


class Downer2025Dataset(AudioNeuralDataset):
    """Squirrel-monkey auditory cortex (Downer 2025 / Ahmed 2025).

    Multi-unit threshold-crossing spike trains from 41 sessions across
    3 animals (B, C, F), recorded passively while the animal listened
    to TIMIT speech and monkey vocalizations. 1718 multi-units total
    (one per recording channel).

    Two stim modes are loaded independently:

    - ``stimuli='timit'`` — 499 unique English sentences (489 single-rep
      + 10 with 11 reps; the 10 form the canonical test subset per
      Ahmed 2025).
    - ``stimuli='mvocs'`` — 303 unique monkey vocalizations (292 single-
      rep + 11 with 15 reps; the 11 form the canonical test subset).

    Both modes share the same recording channels but the per-session
    response counts differ, so a (cell, stim) pair has ``(1, 1)`` NaN
    where the channel was in a session that did not play that stim.

    By default, both stim modes go through the same mel pipeline at
    ``audio_fs=16000`` / ``fmax=8000`` to match the Ahmed 2025 baseline
    (cochleagram capped at 8 kHz) and to allow two instances of this
    class (one per stim mode) to be concatenated via
    ``deepSTRF.utils.concat_neural_datasets``.

    Notes
    -----
    Phase-1 skeleton — stim and response loading are not yet implemented
    (will land in subsequent commits). Instantiating with ``_enumerate_only=True``
    populates ``self.nrn_meta`` and ``self.N_neurons`` for inspection.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        stimuli: Literal["timit", "mvocs"] = "timit",
        dt_ms: float = 5.0,
        n_mels: int = 32,
        compression: Literal["cubic", "log1p", "none"] = "cubic",
        smooth: bool = True,
        subset: Literal["all", "estimation", "test"] = "all",
        animals: Union[str, Sequence[str]] = "all",
        areas: Optional[Sequence[str]] = None,
        sessions: Optional[Sequence[str]] = None,
        audio_fs: int = 16000,
        fmax: int = 8000,
        download: bool = False,
        _enumerate_only: bool = False,
    ):
        """
        Parameters
        ----------
        path : str, optional
            Path to the unpacked dataset root (the directory containing
            ``sessions/`` and ``stimuli/``). Defaults to
            ``default_cache_dir('Downer2025')``.
        stimuli : {'timit', 'mvocs'}
            Which stim class to load. The two share recording channels
            but are loaded independently; concatenate two instances if
            you want both.
        dt_ms : float, default 5.0
            Neural time-bin width. The paper's main analysis uses 50 ms;
            5 ms matches NS1 and gives users the freedom to re-bin.
        n_mels : int, default 32
        compression : {'cubic', 'log1p', 'none'}, default 'cubic'
            Spectrogram amplitude compression. Cubic root matches the
            other deepSTRF audio datasets.
        smooth : bool, default True
            Hsu 2004 21 ms PSTH smoothing.
        subset : {'all', 'estimation', 'test'}
            ``'estimation'`` keeps the single-rep stims, ``'test'`` keeps
            the canonical high-rep subset (10 TIMIT IDs or 11 mVocs IDs).
        animals : 'all' or iterable of {'b','c','f'}
        areas : iterable of {'core'|'primary', 'non-primary'} or fine labels
            {'A1','R','ML','AL','CL','CPB','RPB'}. ``'primary'`` is an
            alias for ``'core'``. None = no filter.
        sessions : iterable of session-id strings, or None.
        audio_fs : int, default 16000
            Common sample rate both stim classes are resampled to before
            mel. mVocs source is 41 kHz stereo; TIMIT is already 16 kHz.
        fmax : int, default 8000
            Mel-band high cutoff. Matches Ahmed 2025's cochleagram.
        download : bool, default False
            Fetch the 29 GB Zenodo archive if missing.
        _enumerate_only : bool, default False
            Phase-1 internal flag. Populates ``self.nrn_meta`` and
            ``self.N_neurons`` then returns, skipping stim and response
            loading. Will be removed once phases 2–3 are in.
        """
        assert stimuli in ("timit", "mvocs"), f"stimuli must be 'timit' or 'mvocs' (got {stimuli!r})"
        assert subset in ("all", "estimation", "test"), (
            f"subset must be 'all', 'estimation' or 'test' (got {subset!r})"
        )
        assert compression in ("cubic", "log1p", "none"), (
            f"compression must be 'cubic', 'log1p' or 'none' (got {compression!r})"
        )
        assert n_mels > 0 and dt_ms > 0 and audio_fs > 0 and fmax > 0

        if path is None:
            path = str(default_cache_dir("Downer2025"))
        if download:
            # Phase 7 — Zenodo fetcher lands in a later commit.
            raise NotImplementedError("download=True will land in Phase 7")

        super().__init__(path, dt_ms)

        self.species = "squirrel monkey"
        self.behavioral_state = "awake-passive"
        self.stimuli = stimuli
        self.subset = subset
        self.F = int(n_mels)
        self.audio_fs = int(audio_fs)
        self.fmax = int(fmax)
        self.compression = compression
        self.smooth = bool(smooth)

        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Downer2025 root not found: {path}. "
                f"Pass download=True (Phase 7) or supply path= manually."
            )

        sess_meta = _parse_sessions_metadata(path)
        self._session_meta = sess_meta

        self.nrn_meta = _enumerate_neurons(
            path, sess_meta,
            animals=animals, areas=areas, sessions=sessions,
        )
        self.N_neurons = len(self.nrn_meta)

        if _enumerate_only:
            # Inspection escape hatch — skip stim/response loading.
            return

        ########################################
        # 1. group neurons by session for I/O batching
        ########################################
        from collections import defaultdict
        sess_neurons: dict[str, list[tuple[int, str]]] = defaultdict(list)
        sessions_root = Path(path) / "sessions"
        for i, nm in enumerate(self.nrn_meta):
            sid = nm["session_id"]
            suf = nm["channel_suffix"] or ""
            fname = f"{nm['animal_id']}_{sid}_Ch{nm['channel']}{suf}_MUspk.mat"
            sess_neurons[sid].append((i, str(sessions_root / sid / fname)))

        ########################################
        # 2. read trial structs (once per session)
        ########################################
        trial_by_sess: dict[str, dict] = {}
        for sid, neurons in sess_neurons.items():
            trial_by_sess[sid] = _load_session_trial(neurons[0][1])

        if stimuli == "timit":
            stim_field = "timitStimcode"
            canon_key = "timit"
            stim_type_label = "timit"
            entries = _load_timit_stims(path)
            min_reps = int(sess_meta["n_reps_canonical"][canon_key])
            # TIMIT design = per-session: 489 single-rep + 10 at 11 reps.
            # Session-level >= 11 reps robustly identifies the 10 test IDs.
            high_rep_ids = _discover_high_rep_set(
                [trial_by_sess[sid][stim_field] for sid in trial_by_sess],
                min_reps=min_reps,
            )
        else:
            stim_field = "mVocStimcode"
            canon_key = "mVocs"
            stim_type_label = "mvoc"
            entries, wav_rep_counts = _load_mvocs_stims(path)
            min_reps = int(sess_meta["n_reps_canonical"][canon_key])
            # mVocs design = canonical WAV: 11 IDs at exactly 15 reps,
            # the other 292 at variable (1-30) reps. Per-session discovery
            # would over-count (sessions can multiply WAV-level reps).
            high_rep_ids = {cid for cid, n in wav_rep_counts.items() if n == min_reps}

        ########################################
        # 3. build mel + stim_meta
        ########################################
        if subset == "test":
            keep = lambda e: e["stim_id"] in high_rep_ids
        elif subset == "estimation":
            keep = lambda e: e["stim_id"] not in high_rep_ids
        else:
            keep = lambda e: True
        entries = [e for e in entries if keep(e)]

        hop = int(round(self.dt * self.audio_fs / 1000))
        assert hop > 0, f"dt_ms={self.dt} too small for audio_fs={self.audio_fs}"
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_fs, n_fft=10 * hop, hop_length=hop,
            n_mels=self.F, f_max=float(self.fmax),
        )

        dt_s = self.dt / 1000.0
        S = len(entries)
        T_by_stim: list[int] = [0] * S

        for s_idx, entry in enumerate(entries):
            wav = torch.from_numpy(entry["sound"]).unsqueeze(0)
            if entry["soundf"] != self.audio_fs:
                wav = torchaudio.functional.resample(wav, entry["soundf"], self.audio_fs)
            spec = mel_tf(wav)  # (1, F, T_spec)
            if self.compression == "cubic":
                spec = spec.pow(1.0 / 3.0)
            elif self.compression == "log1p":
                spec = torch.log1p(spec)
            T_canon = int(round(entry["duration_s"] * 1000.0 / self.dt))
            T_by_stim[s_idx] = T_canon
            T_spec = spec.shape[-1]
            if T_spec > T_canon:
                spec = spec[..., :T_canon]
            elif T_spec < T_canon:
                spec = F.pad(spec, (0, T_canon - T_spec), mode="constant", value=0.0)
            is_repeat = entry["stim_id"] in high_rep_ids
            self.stims.append(spec)
            meta = {
                "name": entry["name"],
                "type": stim_type_label,
                "stim_id": entry["stim_id"],
                "duration_s": entry["duration_s"],
                "n_samples": int(entry["sound"].shape[0]),
                "split": "test" if is_repeat else "estimation",
                "n_reps_canonical": min_reps if is_repeat else 1,
                "befaft_s": entry["befaft_s"],
            }
            if stimuli == "mvocs":
                # mVocs design has variable canonical reps (1-30) per voc;
                # surface the actual WAV count so users can sub-filter.
                meta["n_reps_in_wav"] = wav_rep_counts[entry["stim_id"]]
            self.stim_meta.append(meta)

        ########################################
        # 4. bin spike times into per-(stim, neuron) responses
        ########################################
        N = self.N_neurons
        stim_idx_by_id = {e["stim_id"]: i for i, e in enumerate(entries)}
        self.responses = [[None] * N for _ in range(S)]

        for sid, neurons in tqdm(
            sess_neurons.items(), desc=f"Downer2025 {stimuli} sessions",
        ):
            tr = trial_by_sess[sid]
            codes = tr[stim_field]
            onsets = tr["stimon"]
            sess_stim_reps: dict[int, np.ndarray] = {}
            for code_val in np.unique(codes):
                if code_val == 0:
                    continue
                if int(code_val) not in stim_idx_by_id:
                    continue
                sess_stim_reps[int(code_val)] = onsets[codes == code_val]

            for neuron_idx, chan_path in neurons:
                spks = _load_spike_times(chan_path)
                for stim_id_mat, rep_onsets in sess_stim_reps.items():
                    s_idx = stim_idx_by_id[stim_id_mat]
                    self.responses[s_idx][neuron_idx] = _bin_spikes_per_rep(
                        spks, rep_onsets, T_by_stim[s_idx], dt_s,
                    )

        ########################################
        # 5. NaN sentinels for unobserved (stim, neuron) pairs
        ########################################
        nan11 = torch.full((1, 1), float("nan"))
        for s_idx in range(S):
            for n in range(N):
                if self.responses[s_idx][n] is None:
                    self.responses[s_idx][n] = nan11.clone()

        if self.smooth:
            self.smooth_responses(window_ms=21.0)
        self.validate()

    # ------------------------------------------------------------------
    # Paper-faithful tuning criterion (Ahmed et al. 2025, §"Trial-to-trial
    # neural variability"). Opt-in like ``compute_neuron_quality()`` — it
    # mutates ``nrn_meta`` in place.
    # ------------------------------------------------------------------
    def compute_paper_tuning(
        self,
        n_resamples: int = 10_000,
        dt_ms_analysis: float = 50.0,
        seed: int = 0,
        alpha: float = 0.05,
        delta: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """Replicate Ahmed 2025's tuned / well-tuned multi-unit criterion.

        For each neuron and each of the M test-split stims, the method
        randomly samples a pair of reps (with replacement across iterations
        but never the same rep within a pair), concatenates them into long
        sequences U and V (each of length sum_s T_s_coarse), and computes
        their Pearson correlation. The null distribution is built the same
        way but with each V circularly shifted by a random offset before
        correlating. ``n_resamples`` iterations yield two empirical
        distributions per neuron.

        ``tuned`` -- one-sided Wilcoxon rank-sum test (true > null) at
        ``alpha`` (default 0.05). Paper target: 1195 (TIMIT) / 1231 (mVocs).

        ``well_tuned`` -- additionally requires
        ``(mean(true) - mean(null)) / std(null) >= delta`` (default 0.5).
        Paper target: 404 (TIMIT) / 489 (mVocs).

        Writes four floats / booleans to each ``nrn_meta[i]``, prefixed by
        the current stim mode::

            ahmed2025_{timit|mvocs}_tuned          : bool
            ahmed2025_{timit|mvocs}_well_tuned     : bool
            ahmed2025_{timit|mvocs}_p_wilcoxon     : float
            ahmed2025_{timit|mvocs}_delta_normalized : float

        Parameters
        ----------
        n_resamples : int, default 10_000
            Pair-resamplings per neuron. The paper used 100_000; 10_000 is
            ~10x faster and gives a stable Wilcoxon (the criterion only
            cares about the rank order of the two distributions).
        dt_ms_analysis : float, default 50.0
            Coarse bin width for the long-sequence correlations. Must be an
            integer multiple of ``self.dt``. Paper used 50 ms for the main
            results, 20 ms for Fig 4.
        seed : int, default 0
            RNG seed for reproducibility.
        alpha, delta : float
            Significance and effect-size thresholds (see above).
        verbose : bool, default True
            Show a tqdm progress bar.

        Returns
        -------
        summary : dict
            Aggregate counts ``{'tuned': int, 'well_tuned': int,
            'n_with_data': int, 'stimuli': str}``.

        Notes
        -----
        Re-bins ``self.responses`` to ``dt_ms_analysis`` on the fly via
        summing; if ``smooth=True`` was passed to the constructor the
        smoothing has already been applied at ``self.dt``. For the
        strictest paper match instantiate with ``smooth=False`` — though
        in practice the 21 ms Hanning smoothing × subsequent 50 ms
        re-binning makes the smoothing nearly invisible.
        """
        from scipy.stats import ranksums  # local import — scipy is a runtime dep

        ratio = dt_ms_analysis / self.dt
        if abs(ratio - round(ratio)) > 1e-6 or ratio < 1:
            raise ValueError(
                f"dt_ms_analysis ({dt_ms_analysis}) must be an integer multiple "
                f"of self.dt ({self.dt})"
            )
        rebin = int(round(ratio))

        prefix = f"ahmed2025_{self.stimuli}"
        test_s_idces = [s for s, m in enumerate(self.stim_meta) if m["split"] == "test"]
        if not test_s_idces:
            raise RuntimeError(
                "No test-split stims in current dataset (subset filter dropped them?)."
            )

        # Coarse T per test stim
        T_coarse: list[int] = []
        for s in test_s_idces:
            T5 = int(self.stims[s].shape[-1])
            T_coarse.append(T5 // rebin)
        T_total = sum(T_coarse)

        rng = np.random.default_rng(seed)
        n_tuned = 0
        n_well = 0
        n_with_data = 0

        iterator = tqdm(range(self.N_neurons),
                        desc=f"paper tuning {self.stimuli}",
                        disable=not verbose)
        for n in iterator:
            # Pull and re-bin each test stim's reps for this neuron.
            rebinned: list[np.ndarray] = []
            ok = True
            for s, Tc in zip(test_s_idces, T_coarse):
                r = self.responses[s][n]
                if r.shape[0] < 2 or bool(r.isnan().any()):
                    ok = False
                    break
                if Tc < 2:
                    ok = False
                    break
                r_trunc = r[:, :Tc * rebin]
                r_coarse = r_trunc.reshape(r.shape[0], Tc, rebin).sum(dim=-1)
                rebinned.append(r_coarse.numpy().astype(np.float64))
            if not ok:
                self.nrn_meta[n][f"{prefix}_tuned"] = False
                self.nrn_meta[n][f"{prefix}_well_tuned"] = False
                self.nrn_meta[n][f"{prefix}_p_wilcoxon"] = float("nan")
                self.nrn_meta[n][f"{prefix}_delta_normalized"] = float("nan")
                continue
            n_with_data += 1

            # Build U, V (n_resamples, T_total) by picking 2 distinct reps per stim
            U = np.empty((n_resamples, T_total), dtype=np.float64)
            V = np.empty((n_resamples, T_total), dtype=np.float64)
            offset = 0
            for r_coarse, Tc in zip(rebinned, T_coarse):
                R = r_coarse.shape[0]
                first = rng.integers(0, R, size=n_resamples)
                second = rng.integers(0, R - 1, size=n_resamples)
                second = np.where(second >= first, second + 1, second)
                U[:, offset:offset + Tc] = r_coarse[first]
                V[:, offset:offset + Tc] = r_coarse[second]
                offset += Tc

            # Null V: circular shift per iteration (non-zero shift)
            shifts = rng.integers(1, T_total, size=n_resamples)
            col_idx = (np.arange(T_total)[None, :] - shifts[:, None]) % T_total
            V_null = np.take_along_axis(V, col_idx, axis=1)

            true_corrs = _row_pearson(U, V)
            null_corrs = _row_pearson(U, V_null)
            true_corrs = true_corrs[~np.isnan(true_corrs)]
            null_corrs = null_corrs[~np.isnan(null_corrs)]
            if true_corrs.size < 2 or null_corrs.size < 2:
                self.nrn_meta[n][f"{prefix}_tuned"] = False
                self.nrn_meta[n][f"{prefix}_well_tuned"] = False
                self.nrn_meta[n][f"{prefix}_p_wilcoxon"] = float("nan")
                self.nrn_meta[n][f"{prefix}_delta_normalized"] = float("nan")
                continue

            _, p = ranksums(true_corrs, null_corrs, alternative="greater")
            null_std = float(null_corrs.std())
            null_mean = float(null_corrs.mean())
            true_mean = float(true_corrs.mean())
            delta_norm = (true_mean - null_mean) / null_std if null_std > 0 else 0.0
            tuned = bool(p < alpha)
            well = bool(tuned and delta_norm >= delta)
            self.nrn_meta[n][f"{prefix}_tuned"] = tuned
            self.nrn_meta[n][f"{prefix}_well_tuned"] = well
            self.nrn_meta[n][f"{prefix}_p_wilcoxon"] = float(p)
            self.nrn_meta[n][f"{prefix}_delta_normalized"] = float(delta_norm)
            n_tuned += int(tuned)
            n_well += int(well)

        return {"stimuli": self.stimuli, "n_with_data": n_with_data,
                "tuned": n_tuned, "well_tuned": n_well}


def _row_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two ``(n_rows, T)`` arrays.

    Returns a ``(n_rows,)`` array. Rows where either A or B has zero
    variance yield NaN.
    """
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den
