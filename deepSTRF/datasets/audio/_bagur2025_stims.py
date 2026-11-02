"""Sound table, stimulus metadata and duplicate-waveform groups for Bagur 2025.

The release's ``Code/Utilities/SoundInfo.mat`` lists the 140 sounds in data
order (``StimName``, ``SoundDur``, ``StartFreq``, ``StopFreq``, ``MeanFreq``).
Categories, levels, AM rates and carriers are parsed from the names. Tone
frequencies are measured from the audio (sweep endpoints and AM carriers are
mapped onto these measured values): the names give rounded labels (the
ramp tones "6, 9, 15 kHz" are 6.26, 9.80 and 15.34 kHz, log-spaced 4-24 kHz),
and the authors' tables disagree for one ramp (``Up/Down4+6+9+24kHz``:
``SoundNom.RampFreq`` and ``MeanFreq`` give all five tones; the audio has the
four of the name). Labels are kept in ``frequency_label_khz``.

Internal to :mod:`deepSTRF.datasets.audio.bagur2025`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from deepSTRF.datasets.audio import _trial_audio as _ta

ZENODO_RECORD = 13941450
NATIVE_FS = 192000
ONSET_MS = 500.0
ONSET_SAMPLES = 96000
EPHYS_BINS = 300                 # 5 ms bins, -500..+1000 ms
EPHYS_DT_MS = 5.0
TWOP_FRAMES = 47                 # ~31.7 ms frames, -0.4683..0.9899 s
AREAS = ("AC", "TH", "IC")
METHODS = ("2p", "ephys")
REPS = {("AC", "2p"): 14, ("TH", "2p"): 15, ("IC", "2p"): 13,
        ("IC", "ephys"): 14, ("TH", "ephys"): 15}
DATA_FILES = {("AC", "2p"): "Data_AC.mat", ("TH", "2p"): "Data_TH.mat",
              ("IC", "2p"): "Data_IC.mat", ("IC", "ephys"): "Data_ICE.mat",
              ("TH", "ephys"): "Data_THE.mat"}
CATEGORIES = ("blank", "pure_tones", "up_ramps", "down_ramps", "am", "chirp_durations", "chirps")
DEFAULT_LEVEL_DB = 70.0          # ramps and AM carry no level in their name (SoundNomenclature.m)

_TONE = re.compile(r"^tono(\d+)dB_(\d+)kHz$")
_RAMP = re.compile(r"^(Up|Down)(.+)Hz$")
_AM = re.compile(r"^Sin(\d+)Hz(\d+kHz|Whitenoise)$")
_CHIRP_DUR = re.compile(r"^Chirp(Up|Down)(\d+)kHz(\d+)dB(\d+)ms$")
_CHIRP = re.compile(r"^Chirp(Up|Down)close.*?(\d+)dB$")


def read_sound_info(code_dir) -> List[dict]:
    """Rows of ``Code/Utilities/SoundInfo.mat`` in data order: ``name``,
    ``start_freq_khz``, ``stop_freq_khz``, ``mean_freq_khz`` (None for NaN) and
    ``sound_dur_s``."""
    from scipy.io import loadmat
    p = Path(code_dir) / "Code" / "Utilities" / "SoundInfo.mat"
    si = loadmat(p, squeeze_me=True)
    f = lambda v: None if not np.isfinite(v) else float(v)      # noqa: E731
    return [{"name": str(n), "start_freq_khz": f(a), "stop_freq_khz": f(b),
             "mean_freq_khz": f(m), "sound_dur_s": float(d)}
            for n, a, b, m, d in zip(si["StimName"], si["StartFreq"], si["StopFreq"],
                                     si["MeanFreq"], si["SoundDur"])]


def load_wav_int16(wav_dir, name: str) -> np.ndarray:
    """Samples of ``<wav_dir>/<name>.wav`` as float64 int16 units (silence for
    the blank)."""
    if name == "blank":
        return np.zeros(NATIVE_FS, np.float64)                    # 1 s of silence
    x, fs = sf.read(Path(wav_dir) / f"{name}.wav", dtype="int16")
    if fs != NATIVE_FS:
        raise ValueError(f"{name}.wav: {fs} Hz, expected {NATIVE_FS}")
    return x.astype(np.float64)


def tone_frequencies_khz(wav: np.ndarray, fs: int = NATIVE_FS, rel_db: float = -20.0,
                         merge_rel: float = 0.03) -> List[float]:
    """Frequencies (kHz, 3 decimals) of the spectral peaks of ``wav`` within
    ``rel_db`` of the strongest; peaks closer than ``merge_rel`` (relative) are
    one tone (AM sidebands, leakage)."""
    spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav))))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / fs)
    thr = spec.max() * 10 ** (rel_db / 20.0)
    loc = np.flatnonzero((spec[1:-1] > thr) & (spec[1:-1] >= spec[:-2]) & (spec[1:-1] >= spec[2:])) + 1
    tones: List[List[int]] = []
    for i in loc:                                          # ascending frequency
        if tones and freqs[i] - freqs[tones[-1][-1]] <= merge_rel * freqs[i]:
            tones[-1].append(i)
        else:
            tones.append([i])
    return [round(float(freqs[max(g, key=lambda j: spec[j])]) / 1000.0, 3) for g in tones]


def _parse(name: str) -> dict:
    """Category and name-derived fields of one sound."""
    if name == "blank":
        return {"category": "blank", "group": "blank", "level_db": None}
    m = _TONE.match(name)
    if m:
        lvl = float(m.group(1))
        return {"category": "pure_tones", "group": f"pure_{int(lvl)}dB", "level_db": lvl,
                "frequency_label_khz": [float(m.group(2))]}
    m = _CHIRP_DUR.match(name)
    if m:
        return {"category": "chirp_durations", "group": "chirp_durations",
                "level_db": float(m.group(3)), "direction": m.group(1).lower(),
                "chirp_duration_ms": float(m.group(4))}
    m = _CHIRP.match(name)
    if m:
        d, lvl = m.group(1).lower(), float(m.group(2))
        return {"category": "chirps", "group": f"{d}_chirps_{int(lvl)}dB", "level_db": lvl,
                "direction": d}
    m = _AM.match(name)
    if m:
        carrier = "white_noise" if m.group(2) == "Whitenoise" else float(m.group(2)[:-3])
        return {"category": "am", "group": "am", "level_db": DEFAULT_LEVEL_DB,
                "am_rate_hz": float(m.group(1)), "carrier": carrier,
                "frequency_label_khz": [] if carrier == "white_noise" else [carrier]}
    m = _RAMP.match(name)
    if m:
        d = m.group(1).lower()
        tones = ([4.0, 6.0, 9.0, 15.0, 24.0] if m.group(2) == "multi"
                 else [float(v) for v in m.group(2)[:-1].split("+")])
        return {"category": f"{d}_ramps", "group": f"{d}_ramps", "level_db": DEFAULT_LEVEL_DB,
                "direction": d, "frequency_label_khz": tones}
    raise ValueError(f"unrecognised Bagur 2025 sound name {name!r}")


def build_stim_table(code_dir, wav_dir) -> List[dict]:
    """``stim_meta`` rows in data order (see the dataset docs for every key)."""
    rows = []
    for i, info in enumerate(read_sound_info(code_dir)):
        name = info["name"]
        m = {"name": name, "category": None, "group": None, "level_db": None,
             "frequency_khz": [], "frequency_label_khz": [],
             "start_freq_khz": info["start_freq_khz"],
             "stop_freq_khz": info["stop_freq_khz"], "duration_ms": None, "direction": None,
             "am_rate_hz": None, "carrier": None, "index_in_release": i}
        m.update(_parse(name))
        wav = load_wav_int16(wav_dir, name)
        m["duration_ms"] = 0.0 if name == "blank" else 1000.0 * len(wav) / NATIVE_FS
        if m["category"] in ("pure_tones", "up_ramps", "down_ramps") or (
                m["category"] == "am" and m["carrier"] != "white_noise"):
            m["frequency_khz"] = tone_frequencies_khz(wav)        # measured; labels are rounded
        elif m["category"] in ("chirps", "chirp_durations"):
            m["frequency_label_khz"] = [info["start_freq_khz"], info["stop_freq_khz"]]
        rows.append(m)
    # sweep endpoints and AM carriers on the measured tone grid (labels are rounded)
    grid = {}
    for m in rows:
        if len(m["frequency_khz"]) == len(m["frequency_label_khz"]):
            grid.update(zip(m["frequency_label_khz"], m["frequency_khz"]))
    for m in rows:
        if m["category"] in ("chirps", "chirp_durations"):
            m["frequency_khz"] = [grid.get(f, f) for f in m["frequency_label_khz"]]
        elif m["category"] == "am" and m["carrier"] != "white_noise":
            m["carrier"] = m["frequency_khz"][0]
    return rows


def twop_grid(t_s: np.ndarray) -> Tuple[float, float]:
    """(dt_ms, buffer_start_ms) of the 2-photon frames from the released frame
    times ``t_s`` (s from onset). The times are frame RIGHT edges, like the 5 ms
    axis (``t[0]`` minus the period is -500 ms exactly). The period (~31.70 ms)
    is rounded to whole 192 kHz samples (6086 = 31.698 ms) so the waveform grid
    locks; ``ValueError`` if that drifts by >= 1 ms over the trial."""
    period = float(np.diff(np.asarray(t_s, np.float64)).mean()) * 1000.0
    dt = round(period * NATIVE_FS / 1000.0) * 1000.0 / NATIVE_FS
    drift = len(t_s) * abs(period - dt)
    if drift >= 1.0:
        raise ValueError(f"2p frame period {period:.4f} ms vs grid-locked {dt:.4f} ms: "
                         f"{drift:.2f} ms drift over the trial")
    return dt, float(t_s[0]) * 1000.0 - period


_GROUPS_JSON = Path(__file__).with_name("_bagur2025_stim_groups.json")


def find_duplicates(wav_dir, names: Sequence[str], thresh: float = 0.999) -> List[List[str]]:
    """Groups of sounds that are the same waveform up to a gain (sign included)."""
    return _ta.find_gain_copies({n: load_wav_int16(wav_dir, n) for n in names if n != "blank"},
                                thresh)


def load_duplicate_groups() -> List[List[str]]:
    """The duplicate-waveform groups of the release, shipped with the loader."""
    with open(_GROUPS_JSON) as f:
        return json.load(f)["duplicates"]


def trial_waveform(wav: np.ndarray, n_bins: int, hop: int) -> np.ndarray:
    """Trial buffer of ``n_bins * hop`` samples with the sound at +500 ms."""
    return _ta.trial_waveform(wav, n_bins, hop, ONSET_SAMPLES)


def logspec_trial(wav: np.ndarray, n_bins: int, dt_ms: float, fmin: float, fmax: float,
                  bins_per_octave: int, window_ms: Optional[float] = None,
                  onset_samples: int = ONSET_SAMPLES) -> np.ndarray:
    """(F, n_bins) absolute-floor log spectrogram of the trial buffer."""
    return _ta.logspec_trial(wav, n_bins, dt_ms, fmin, fmax, bins_per_octave, onset_samples,
                             window_ms, NATIVE_FS)


def split_units(stim_meta: Sequence[dict], duplicates: Sequence[Sequence[str]]
                ) -> Tuple[List[str], List[str]]:
    """Per sound: the unit it is drawn with (a duplicate group's first name, or
    the sound itself) and that unit's stratum (its category). Time-reversed
    ramps and chirps are separate units."""
    rep = {n: sorted(g)[0] for g in duplicates for n in g}
    units = [rep.get(m["name"], m["name"]) for m in stim_meta]
    return units, [m["category"] for m in stim_meta]
