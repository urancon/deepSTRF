"""Helpers shared by the trial-based mouse datasets (Gosselin 2025, Bagur 2025).

Both releases play int16 wav files at 192 kHz inside a fixed trial window and
record repeated trials. Shared here: the trial waveform buffer, the
absolute-floor log spectrogram, the finder of sounds that are the same
waveform up to a gain, the stratified split draw, overlap-weighted re-binning
onto a coarser grid, and md5-checked downloads.

Internal to :mod:`deepSTRF.datasets.audio`.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from deepSTRF.datasets.audio._logspectrogram import logspectrogram

KEYS = ("train", "val", "test")


# ------------------------------------------------------------------ stimuli
def trial_waveform(wav: np.ndarray, n_bins: int, hop: int, onset_samples: int) -> np.ndarray:
    """Zero buffer of ``n_bins * hop`` samples with the sound starting at
    ``onset_samples``. Samples stay in int16 units (absolute level kept). A
    sound that would overrun the buffer is cropped."""
    buf = np.zeros(n_bins * hop, dtype=np.float64)
    n = max(min(len(wav), len(buf) - onset_samples), 0)
    buf[onset_samples:onset_samples + n] = wav[:n]
    return buf


def spec_floor_db(window_ms: float, fs: int = 192000) -> float:
    """dB level of int16 quantisation noise (variance 1/12) through the
    Hamming window used by :func:`logspectrogram`: ``10 log10(sum(w^2) / 12)``.

    Silence and anything quieter than the 16-bit quantisation noise of the
    files sit at this floor, for every sound alike.
    """
    win = np.hamming(max(int(round(window_ms * fs / 1000.0)), 4))
    return float(10.0 * np.log10(np.sum(win ** 2) / 12.0))


def logspec_trial(wav: np.ndarray, n_bins: int, dt_ms: float, fmin: float, fmax: float,
                  bins_per_octave: int, onset_samples: int, window_ms: Optional[float] = None,
                  fs: int = 192000) -> np.ndarray:
    """(F, n_bins) log-spaced spectrogram (dB re 1 int16 step) of the trial
    buffer, clipped below at :func:`spec_floor_db`."""
    hop = int(round(dt_ms * fs / 1000.0))
    window_ms = 2.0 * dt_ms if window_ms is None else window_ms
    S_db, _ = logspectrogram(trial_waveform(wav, n_bins, hop, onset_samples), fs, dt_ms=dt_ms,
                             fmin=fmin, fmax=fmax, bins_per_octave=bins_per_octave,
                             window_ms=window_ms)
    S_db = np.maximum(S_db[:, :n_bins], spec_floor_db(window_ms, fs))
    return S_db.astype(np.float32)


def find_gain_copies(wavs: Mapping[str, np.ndarray], thresh: float = 0.999) -> List[List[str]]:
    """Groups of sounds that are the same waveform up to a scalar gain (sign
    included): equal length and |Pearson r| > ``thresh``. Each group sorted,
    groups sorted; singletons left out."""
    names = list(wavs)
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
                x, y = np.asarray(wavs[a], np.float64), np.asarray(wavs[b], np.float64)
                if x.std() > 0 and y.std() > 0 and abs(np.corrcoef(x, y)[0, 1]) > thresh:
                    parent[root(b)] = root(a)
    out: Dict[str, List[str]] = {}
    for n in names:
        out.setdefault(root(n), []).append(n)
    return sorted(sorted(g) for g in out.values() if len(g) > 1)


# ------------------------------------------------------------------- splits
def draw_split(units: Sequence[str], strata: Sequence[str], seed, fractions: Sequence[float]
               ) -> Dict[str, List[int]]:
    """Assign whole units to train / val / test, stratum by stratum; each unit,
    in a seeded shuffle, goes to the set furthest below its target share.
    Returns the stimulus indices of each set."""
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be 3 shares summing to 1, got {fractions}")
    rng = random.Random(seed)
    size: Dict[str, int] = {}
    stratum_of: Dict[str, str] = {}
    for u, st in zip(units, strata):
        size[u] = size.get(u, 0) + 1
        stratum_of[u] = st
    assign: Dict[str, str] = {}
    for st in sorted(set(strata)):
        todo = sorted(u for u in size if stratum_of[u] == st)
        rng.shuffle(todo)
        total = sum(size[u] for u in todo)
        count = dict.fromkeys(KEYS, 0)
        for u in todo:
            k = max(KEYS, key=lambda k: fractions[KEYS.index(k)] * total - count[k])
            assign[u] = k
            count[k] += size[u]
    out = {k: [] for k in KEYS}
    for i, u in enumerate(units):
        out[assign[u]].append(i)
    return out


# --------------------------------------------------------------- re-binning
def overlap_weights(n_in: int, in_ms: float, out_ms: float, n_out: int,
                    offset_ms: float = 0.0) -> np.ndarray:
    """(n_out, n_in) weights averaging input bins of ``in_ms`` (bin j covers
    [j in_ms, (j+1) in_ms)) into output bins of ``out_ms``; output bin k covers
    [offset_ms + k out_ms, offset_ms + (k+1) out_ms). Rows sum to 1."""
    lo = np.arange(n_in, dtype=np.float64) * in_ms
    W = np.zeros((n_out, n_in))
    for k in range(n_out):
        b, e = offset_ms + k * out_ms, offset_ms + (k + 1) * out_ms
        W[k] = np.clip(np.minimum(lo + in_ms, e) - np.maximum(lo, b), 0.0, None)
    s = W.sum(axis=1, keepdims=True)
    if np.any(s == 0):
        raise ValueError("an output bin has no overlap with the input grid")
    return W / s


# ---------------------------------------------------------------- downloads
_STAMPS = ".md5_checked.json"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def md5_intact(root: Path, name: str, md5: str,
               hasher: Optional[Callable[[Path], str]] = None) -> bool:
    """md5 check of ``root/name``, remembered by size + mtime (in
    ``root/.md5_checked.json``) so a big file is hashed once."""
    hasher = md5_of if hasher is None else hasher
    root = Path(root)
    dest = root / name
    if not dest.exists():
        return False
    st = dest.stat()
    key = [st.st_size, st.st_mtime_ns, md5]
    stamps_path = root / _STAMPS
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    if stamps.get(name) == key:
        return True
    if hasher(dest) != md5:
        return False
    stamps[name] = key
    stamps_path.write_text(json.dumps(stamps))
    return True


def fetch_checked(record: int, name: str, root: Path, md5: str,
                  download: Optional[Callable] = None,
                  hasher: Optional[Callable[[Path], str]] = None) -> None:
    """Make ``root/name`` an intact copy of Zenodo ``record``'s file: skip it
    if intact, otherwise delete it, download it and check it again (IOError,
    file deleted, on a second mismatch)."""
    if download is None:
        from deepSTRF.utils.data_download import zenodo_download as download
    root = Path(root)
    if md5_intact(root, name, md5, hasher):
        return
    dest = root / name
    if dest.exists():
        dest.unlink()                             # stream_download keeps existing files
    download(record, name, dest)
    if not md5_intact(root, name, md5, hasher):
        dest.unlink()
        raise IOError(f"{name}: md5 mismatch after download; deleted, retry the download")
