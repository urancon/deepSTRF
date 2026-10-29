"""Sound table, stimulus metadata and stimulus builders for Gosselin 2025.

Internal to :mod:`deepSTRF.datasets.audio.gosselin2025`.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

from deepSTRF.datasets.audio._logspectrogram import logspectrogram

ZENODO_RECORD = 17271813
NATIVE_FS = 192000
ONSET_MS = 200.0
EPHYS_BINS = 900            # 1 ms bins, -200..+700 ms
AC_DT_MS = 43.5
AC_BINS = 24                # -200..+844 ms
ONSET_SAMPLES = int(round(ONSET_MS * NATIVE_FS / 1000.0))      # 38400
N_REPS = 12
DECOMPOSED_CLIPS = ("dauphin2", "India", "bird", "bird3")

#: Release CSV names that do not match their wav file.
NAME_FIXES = {"Ramps_down7050_16khz": "Ramps_down7050_16kHz"}

CATEGORY_SLUGS = {
    "PURE TONES": "pure_tones", "RAMPS": "ramps",
    "LOW CHORDS": "low_chords", "MEDIUM CHORDS": "medium_chords",
    "HIGH CHORDS": "high_chords", "BROAD CHORDS": "broad_chords",
    "HARMONIC CHORDS": "harmonic_chords", "AM": "am",
    "CHIRPS VELOCITY": "chirps_velocity", "CHIRPS FREQUENCY": "chirps_frequency",
    "WHITENOISES": "white_noises", "NARROW NOISES": "narrow_noises",
    "BROAD NOISES": "broad_noises", "DECOMPOSITIONS": "decompositions",
    "COMPLEX SOUNDS": "complex_sounds",
}

_COMPLEX_SUFFIX = re.compile(r"_(inverted)?(50|70)dB$")


def _floats(cell: str) -> List[float]:
    return [float(x) for x in re.split(r"[,\-]", cell) if x.strip()]


def read_sound_table(csv_path) -> List[dict]:
    """Parse ``Sound Parameters.csv`` (tab-separated, category filled down)."""
    rows, cat = [], None
    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            if not any(r):
                continue
            cat = r[0].strip() or cat
            name = NAME_FIXES.get(r[1].strip(), r[1].strip())
            rows.append({
                "category_raw": cat,
                "name": name,
                "intensity_db": _floats(r[2]),
                "frequency_khz": _floats(r[3]),
                "duration_ms_csv": float(r[4]),
                "index_cc": int(r[5]),
                "index_cn_ic": int(r[6]),
                "index_ac": int(r[7]),
            })
    return rows


def _direction(category: str, row: dict) -> Optional[str]:
    if category.startswith("chirps"):
        f = row["frequency_khz"]
        return "up" if f[-1] > f[0] else "down"
    if category in ("ramps", "white_noises", "narrow_noises", "decompositions"):
        i = row["intensity_db"]
        if len(i) == 2 and i[0] != i[1]:
            return "up" if i[1] > i[0] else "down"
    return None


def derive_stim_meta(row: dict, duration_ms: float) -> dict:
    """The ``stim_meta`` dict of one sound (see README_Gosselin_2025)."""
    category = CATEGORY_SLUGS[row["category_raw"]]
    name = row["name"]
    meta = {
        "name": name,
        "category": category,
        "intensity_db": list(row["intensity_db"]),
        "frequency_khz": list(row["frequency_khz"]),
        "duration_ms": float(duration_ms),
        "duration_ms_csv": row["duration_ms_csv"],
        "index_cc": row["index_cc"],
        "index_cn_ic": row["index_cn_ic"],
        "index_ac": row["index_ac"],
        "level_db": (row["intensity_db"][0]
                     if len(row["intensity_db"]) == 1 and category != "am" else None),
        "reversed": "inverted" in name,
        "direction": _direction(category, row),
        "source_clip": None,
        "decomposition_role": None,
        "am_rate_hz": None,
        "carrier": None,
    }
    if category == "complex_sounds":
        meta["source_clip"] = _COMPLEX_SUFFIX.sub("", name)
        if meta["source_clip"] in DECOMPOSED_CLIPS and not meta["reversed"]:
            meta["decomposition_role"] = "original"
    elif category == "decompositions":
        meta["source_clip"] = name.split("_", 1)[0]
        meta["decomposition_role"] = ("reconstructed" if name.endswith("_reconstructed")
                                      else "component")
    elif category == "am":
        m = re.match(r"AM_(\d+)Hz_(.+)$", name)
        meta["am_rate_hz"] = float(m.group(1))
        rest = m.group(2)
        meta["carrier"] = ("white_noise" if rest == "WN"
                           else "chord" if "+" in rest else "tone")
    return meta


def load_wav_int16(wav_dir, name: str) -> np.ndarray:
    """One sound as int16 samples at the native 192 kHz."""
    x, fs = sf.read(Path(wav_dir) / f"{name}.wav", dtype="int16")
    if fs != NATIVE_FS:
        raise ValueError(f"{name}.wav: expected {NATIVE_FS} Hz, got {fs}")
    return x if x.ndim == 1 else x[:, 0]


def build_stim_table(csv_path, wav_dir) -> List[dict]:
    """stim_meta for every CSV row, in CSV order; durations from the wavs."""
    out = []
    for row in read_sound_table(csv_path):
        n = sf.info(str(Path(wav_dir) / f"{row['name']}.wav")).frames
        out.append(derive_stim_meta(row, 1000.0 * n / NATIVE_FS))
    return out


def trial_waveform(wav: np.ndarray, n_bins: int, hop: int) -> np.ndarray:
    """Zero buffer of ``n_bins * hop`` samples with the sound at +200 ms.

    Samples stay in int16 units (absolute level kept). A sound that would
    overrun the buffer is cropped (none does in the release).
    """
    buf = np.zeros(n_bins * hop, dtype=np.float64)
    n = min(len(wav), len(buf) - ONSET_SAMPLES)
    buf[ONSET_SAMPLES:ONSET_SAMPLES + n] = wav[:n]
    return buf


def spec_floor_db(window_ms: float, fs: int = NATIVE_FS) -> float:
    """dB level of int16 quantisation noise (variance 1/12) through the
    Hamming window used by :func:`logspectrogram`: ``10 log10(sum(w^2) / 12)``.

    Silence and anything quieter than the 16-bit quantisation noise of the
    files sit at this floor, for every sound alike.
    """
    win = np.hamming(max(int(round(window_ms * fs / 1000.0)), 4))
    return float(10.0 * np.log10(np.sum(win ** 2) / 12.0))


def logspec_trial(wav: np.ndarray, n_bins: int, dt_ms: float, fmin: float, fmax: float,
                  bins_per_octave: int, window_ms: Optional[float] = None) -> np.ndarray:
    """(F, n_bins) log-spaced spectrogram (dB re 1 int16 step) of the trial
    buffer, clipped below at :func:`spec_floor_db`."""
    hop = int(round(dt_ms * NATIVE_FS / 1000.0))
    window_ms = 2.0 * dt_ms if window_ms is None else window_ms
    S_db, _ = logspectrogram(trial_waveform(wav, n_bins, hop), NATIVE_FS, dt_ms=dt_ms,
                             fmin=fmin, fmax=fmax, bins_per_octave=bins_per_octave,
                             window_ms=window_ms)
    S_db = np.maximum(S_db[:, :n_bins], spec_floor_db(window_ms))
    return S_db.astype(np.float32)


_GROUPS_JSON = Path(__file__).with_name("_gosselin2025_stim_groups.json")


def find_gain_copies(wav_dir, names: Sequence[str], thresh: float = 0.999) -> List[List[str]]:
    """Groups of sounds that are the same waveform up to a scalar gain
    (equal length and |Pearson r| > ``thresh``)."""
    wavs = {n: load_wav_int16(wav_dir, n).astype(np.float64) for n in names}
    parent = {n: n for n in names}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n

    by_len: Dict[int, List[str]] = {}
    for n in names:
        by_len.setdefault(len(wavs[n]), []).append(n)
    for group in by_len.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                x, y = wavs[a], wavs[b]
                if x.std() > 0 and y.std() > 0 and abs(np.corrcoef(x, y)[0, 1]) > thresh:
                    parent[root(b)] = root(a)
    out: Dict[str, List[str]] = {}
    for n in names:
        out.setdefault(root(n), []).append(n)
    return sorted(sorted(g) for g in out.values() if len(g) > 1)


def load_gain_copies() -> List[List[str]]:
    """The gain-copy groups of the release, shipped with the loader."""
    with open(_GROUPS_JSON) as f:
        return json.load(f)["gain_copies"]
