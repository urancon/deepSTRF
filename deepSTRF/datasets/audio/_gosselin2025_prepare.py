"""Download and slim cache for Gosselin 2025 (Zenodo 17271813).

Cache layout (``<path>/_cache`` by default):

- ``{CN,IC}_spikes.npz``: per-neuron offsets (int64, N+1), within-neuron flat
  index over (sound, repeat, 1 ms bin) in CSV sound order (uint32), spike
  count (uint8). Re-binned exactly to any allowed ``dt_ms`` at load.
- ``AC.npy``: float16 (N, S, R, 24) in CSV sound order, NaN = missing repeat.
- ``CC_ihc.npy``: float32 (n_ihc, 900, S) hair-cell mean rates; ``CC_groups.npy``.
- ``<key>.json``: manifest with ``version`` and shapes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from deepSTRF.datasets.audio._gosselin2025_stims import ZENODO_RECORD, read_sound_table
from deepSTRF.datasets.audio._gosselin2025_cochlear import ihc_groups_from_spont, ihc_means
from deepSTRF.utils.data_download import unzip, zenodo_download

CACHE_VERSION = 1

#: md5 of every release file (Zenodo 17271813; arrays identical to 14421103).
MD5 = {
    "Dataset_AC.npy": "ac9653ecaf47a9d66055428e93a34667",
    "Dataset_IC.npy": "fed30d8460d7b00e95c689f37ce27f98",
    "Dataset_CN.npy": "62aca6528b8be6ad5c17ccede56113fb",
    "Dataset_CC.npy": "1440b6879a1ead5f42af867a6e4827e9",
    "Soundset.zip": "2f24a196c18a887d8100885fedb5b4db",
    "Info_CN.csv": "e39a821019ca4845ed8e71c7bd9fb282",
    "Info_AC.csv": "fa60bf5a070f5abfe25d978dd7a5d7f1",
    "Info_IC.csv": "3f6277dc416b958cdd7aeac6a35b9fb8",
    "Sound Parameters.csv": "dbc5e8bb9f15ce2a2a911005cbb5c675",
}
_CHUNK = 64


def required_files(areas: Sequence[str], cochlear: bool, fibres: bool) -> List[str]:
    """Release files needed to load ``areas`` (and the auditory-nerve model)."""
    files = ["Sound Parameters.csv", "Soundset.zip"]
    for a in areas:
        files += [f"Info_{a}.csv", f"Dataset_{a}.npy"]
    if cochlear or fibres:
        files.append("Dataset_CC.npy")
    return files


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def download_gosselin2025(path, areas: Sequence[str] = ("CN", "IC", "AC"),
                          cochlear: bool = True) -> Path:
    """Fetch the files needed for ``areas`` (+ the auditory-nerve model) into
    ``path``, check their md5, unzip the sound set. Files already present and
    intact are skipped."""
    root = Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name in required_files(areas, cochlear, False):
        dest = root / name
        if dest.exists() and _md5(dest) == MD5[name]:
            continue
        zenodo_download(ZENODO_RECORD, name, dest)
        got = _md5(dest)
        if got != MD5[name]:
            dest.unlink()
            raise IOError(f"{name}: md5 {got} != {MD5[name]}; deleted, retry the download")
    wav_dir(root)
    return root


def wav_dir(root) -> Path:
    """``<root>/soundset/Soundset``, unzipping ``Soundset.zip`` if needed."""
    root = Path(root)
    d = root / "soundset" / "Soundset"
    if not d.is_dir():
        unzip(root / "Soundset.zip", root / "soundset")
    return d


def _manifest(cache: Path, key: str) -> Path:
    return cache / f"{key}.json"


def is_prepared(cache, key: str) -> bool:
    p = _manifest(Path(cache), key)
    if not p.exists():
        return False
    return json.loads(p.read_text()).get("version") == CACHE_VERSION


def _write_manifest(cache: Path, key: str, **fields) -> None:
    _manifest(cache, key).write_text(json.dumps({"version": CACHE_VERSION, **fields}))


def _order(root: Path, area: str) -> np.ndarray:
    col = "index_ac" if area == "AC" else "index_cn_ic"
    return np.array([r[col] for r in read_sound_table(root / "Sound Parameters.csv")])


def prepare_area(root, cache, area: str) -> None:
    """Build the cache of one recorded area (CN, IC or AC), in CSV sound order."""
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    X = np.load(root / f"Dataset_{area}.npy", mmap_mode="r")
    order = _order(root, area)
    N, S = X.shape[0], len(order)
    shape = [N, S, X.shape[3], X.shape[1]]
    if area == "AC":
        out = np.lib.format.open_memmap(cache / "AC.npy", mode="w+", dtype=np.float16,
                                        shape=tuple(shape))
        for i in range(0, N, _CHUNK):
            a = np.asarray(X[i:i + _CHUNK])[:, :, order, :]          # (n, T, S, R)
            out[i:i + _CHUNK] = a.transpose(0, 2, 3, 1).astype(np.float16)
        out.flush()
        del out
        _write_manifest(cache, area, shape=shape)
        return
    offsets, idx, cnt = [0], [], []
    for i in range(0, N, _CHUNK):
        a = np.asarray(X[i:i + _CHUNK])[:, :, order, :]              # (n, T, S, R)
        a = a.transpose(0, 2, 3, 1).reshape(a.shape[0], -1)          # (n, S*R*T)
        if a.max() > 255 or a.min() < 0:
            raise ValueError(f"{area}: spike counts outside 0..255")
        rows, cols = np.nonzero(a)
        idx.append(cols.astype(np.uint32))
        cnt.append(a[rows, cols].astype(np.uint8))
        offsets += list(offsets[-1] + np.cumsum(np.bincount(rows, minlength=a.shape[0])))
    np.savez(cache / f"{area}_spikes.npz", offsets=np.array(offsets, np.int64),
             idx=np.concatenate(idx), counts=np.concatenate(cnt))
    _write_manifest(cache, area, shape=shape)


def load_ephys(cache, area: str, dt_ms: int) -> np.ndarray:
    """(N, S, R, 900 // dt_ms) float32 spike counts per bin, CSV sound order."""
    cache = Path(cache)
    N, S, R, T0 = json.loads(_manifest(cache, area).read_text())["shape"]
    z = np.load(cache / f"{area}_spikes.npz")
    off, idx, cnt = z["offsets"], z["idx"].astype(np.int64), z["counts"].astype(np.float32)
    Tb = T0 // dt_ms
    out = np.zeros((N, S * R * Tb), np.float32)
    for n in range(N):
        sl = slice(off[n], off[n + 1])
        sr, t = np.divmod(idx[sl], T0)
        out[n] = np.bincount(sr * Tb + t // dt_ms, weights=cnt[sl], minlength=S * R * Tb)
    return out.reshape(N, S, R, Tb)


def load_ac(cache) -> np.ndarray:
    """(N, S, R, 24) float32 deconvolved responses, CSV sound order, NaN kept."""
    return np.load(Path(cache) / "AC.npy").astype(np.float32)


def load_sessions(root, area: str) -> np.ndarray:
    """Recording session of each neuron, in release neuron order."""
    with open(Path(root) / f"Info_{area}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["Neuron Index"]))
    return np.array([int(r["Recording Session"]) for r in rows])


def prepare_cochlear(root, cache) -> None:
    """Cache the auditory-nerve model as hair-cell means, CSV sound order."""
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    order = np.array([r["index_cc"] for r in read_sound_table(root / "Sound Parameters.csv")])
    X = np.load(root / "Dataset_CC.npy", mmap_mode="r")
    cc = np.asarray(X)[:, :, order]
    groups = ihc_groups_from_spont(cc[:, :200, :].mean(axis=(1, 2)))
    np.save(cache / "CC_ihc.npy", ihc_means(cc, groups))
    np.save(cache / "CC_groups.npy", groups)
    _write_manifest(cache, "CC", n_fibres=int(cc.shape[0]), n_ihc=int(groups.max() + 1))


def load_cochlear_ihc(cache):
    """((n_ihc, 900, S) float32 hair-cell rates, (n_fibres,) hair cell of each fibre)."""
    cache = Path(cache)
    return np.load(cache / "CC_ihc.npy"), np.load(cache / "CC_groups.npy")
