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

import yaml

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

    area_whitelist = set(areas) if areas is not None else None

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
        smooth : bool, default True
            Hsu 2004 21 ms PSTH smoothing.
        subset : {'all', 'estimation', 'test'}
            ``'estimation'`` keeps the single-rep stims, ``'test'`` keeps
            the canonical high-rep subset (10 TIMIT IDs or 11 mVocs IDs).
        animals : 'all' or iterable of {'b','c','f'}
        areas : iterable of {'core','non-primary'} or fine labels
            {'A1','R','ML','AL','CL','CPB','RPB'}. None = no filter.
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
            # Phase-1 inspection mode — skip stim/response loading.
            return

        # Phases 2–3: load stims (TIMIT or mVocs) + per-(stim,neuron) responses.
        raise NotImplementedError(
            "Downer2025Dataset stim/response loading is not yet implemented. "
            "Use _enumerate_only=True for Phase-1 neuron-metadata inspection."
        )
