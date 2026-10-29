"""Download, raw-count recovery and slim cache for Bagur 2025 (Zenodo 13941450).

Release layout: 2-photon ``Data_{AC,TH,IC}.mat`` are MATLAB v7.3 (HDF5), seen
by h5py transposed as (R, S, T, N); electrophysiology ``Data_{ICE,THE}.mat``
and ``Data_CC.mat`` are v5, (N, T, S, R). Sounds are in data order already.

The electrophysiology is shipped baseline-subtracted per trial (each trial's
pre-stimulus mean removed). Spike counts are recovered exactly: a trial's
minimum is minus its baseline, since every trial has empty 5 ms bins.

Cache layout (``<path>/_cache`` by default), one key per population
(``"AC-2p"``, ``"IC-ephys"``, ...):

- 2p: ``<key>.npy`` float16 (N, S, R, 47); ``twop_t.npy`` the 47 frame times (s);
- ephys: ``<key>_spikes.npz`` per-neuron offsets, flat index over (sound,
  repeat, 5 ms bin) and counts (uint8);
- ``<key>.json``: manifest with ``version`` and shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from deepSTRF.datasets.audio import _trial_audio as _ta
from deepSTRF.datasets.audio._bagur2025_stims import DATA_FILES, ZENODO_RECORD
from deepSTRF.utils.data_download import unzip, zenodo_download

CACHE_VERSION = 1
_CHUNK = 256

#: md5 of the release files used by the loader (Zenodo API, record 13941450).
MD5 = {
    "Data_AC.mat": "9b3a2300187cb377652c8c77c34764b5",
    "Data_TH.mat": "86f15280c72a6e2071536f49d6fe23ee",
    "Data_IC.mat": "b98a934e4f5f8315e140196f087bd7ce",
    "Data_ICE.mat": "16fb7035799a6dea63041db5745ea2bf",
    "Data_THE.mat": "0abc293afe24254e83a13cf3a356c64c",
    "Data_CC.mat": "67ff22b1832e10492c4f16aeff094a77",
    "AnatInfo_AC.mat": "7742830a606a615de46f1000616d334f",
    "AnatInfo_ICE.mat": "ab65e7ab6888cb64d86c555ccb1788fa",
    "Code.zip": "e16c68a608e1d89f90fbfc24795f64be",
    "Sounds.zip": "dcec827cab5b5374b895579ff1d23492",
}
_ANAT_FILES = {("AC", "2p"): "AnatInfo_AC.mat", ("IC", "ephys"): "AnatInfo_ICE.mat"}


def key_of(area: str, method: str) -> str:
    return f"{area}-{method}"


def required_files(areas: Sequence[str], method: str, cochlear: bool,
                   prepared: Sequence[str] = (), have_wavs: bool = False,
                   have_code: bool = False) -> List[str]:
    """Release files needed to load ``areas`` recorded with ``method``.

    ``prepared`` lists the cache keys already built (their data files are not
    needed; ``"CC"`` for the cochlear model); ``have_wavs`` / ``have_code``: the
    sound set / code archive is already unzipped. Anatomy files are always
    read at load time (they are small)."""
    files = [] if have_code else ["Code.zip"]
    files += [] if have_wavs else ["Sounds.zip"]
    for a in areas:
        if key_of(a, method) not in prepared:
            files.append(DATA_FILES[(a, method)])
        if (a, method) in _ANAT_FILES:
            files.append(_ANAT_FILES[(a, method)])
    if cochlear and "CC" not in prepared:
        files.append("Data_CC.mat")
    return files


def code_dir(root) -> Path:
    """``<root>/code`` (holds ``Code/Utilities/SoundInfo.mat``), unzipped if needed."""
    root = Path(root)
    if not (root / "code" / "Code").is_dir():
        unzip(root / "Code.zip", root / "code")
    return root / "code"


def wav_dir(root) -> Path:
    """``<root>/sounds/Sounds``, unzipping ``Sounds.zip`` if needed."""
    root = Path(root)
    d = root / "sounds" / "Sounds"
    if not d.is_dir():
        unzip(root / "Sounds.zip", root / "sounds")
    return d


def download_bagur2025(path, areas: Union[str, Sequence[str]] = ("AC", "TH", "IC"), method: str = "2p",
                       cochlear: bool = False, prepared: Sequence[str] = ()) -> Path:
    """Fetch the files needed for ``areas`` x ``method`` (+ the cochlear model)
    into ``path``, md5-checked (each file hashed once, then remembered; a
    corrupt one is fetched again), and unzip the code and sound archives.
    Data files whose cache is built (``prepared``) are not fetched."""
    areas = (areas,) if isinstance(areas, str) else tuple(areas)
    bad = [a for a in areas if (a, method) not in DATA_FILES]
    if bad:
        raise ValueError(f"no {method!r} recording for area(s) {bad} (2p: AC, TH, IC; ephys: IC, TH)")
    root = Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    have_wavs = (root / "sounds" / "Sounds").is_dir()
    have_code = (root / "code" / "Code").is_dir()
    for name in required_files(areas, method, cochlear, prepared, have_wavs, have_code):
        _ta.fetch_checked(ZENODO_RECORD, name, root, MD5[name], download=zenodo_download)
    code_dir(root)
    wav_dir(root)
    return root


# -------------------------------------------------------------------- cache
def _manifest(cache: Path, key: str) -> Path:
    return Path(cache) / f"{key}.json"


def is_prepared(cache, key: str) -> bool:
    p = _manifest(cache, key)
    return p.exists() and json.loads(p.read_text()).get("version") == CACHE_VERSION


def _write_manifest(cache: Path, key: str, **fields) -> None:
    _manifest(cache, key).write_text(json.dumps({"version": CACHE_VERSION, **fields}))


def recover_counts(x: np.ndarray, time_axis: int) -> np.ndarray:
    """Spike counts from baseline-subtracted trials: subtract each trial's
    minimum over ``time_axis``. Raises ``ValueError`` if the result is not
    integer (within 1e-3) or exceeds 255."""
    c = x - x.min(axis=time_axis, keepdims=True)
    r = np.round(c)
    if np.abs(c - r).max() > 1e-3:
        raise ValueError("electrophysiology trials do not recover integer spike counts "
                         f"(max residual {np.abs(c - r).max():.3g})")
    if r.max() > 255:
        raise ValueError("spike counts above 255 in one 5 ms bin")
    return r.astype(np.uint8)


def prepare_population(root, cache, area: str, method: str) -> None:
    """Build the slim cache of one population (see the module docstring)."""
    import h5py
    from scipy.io import loadmat
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    key, path = key_of(area, method), root / DATA_FILES[(area, method)]
    if method == "2p":
        with h5py.File(path, "r") as h:
            D = h["Data"]                                          # (R, S, T, N)
            R, S, T, N = D.shape
            np.save(cache / "twop_t.npy", np.asarray(h["t"][()], np.float64).ravel())
            out = np.lib.format.open_memmap(cache / f"{key}.npy", mode="w+", dtype=np.float16,
                                            shape=(N, S, R, T))
            for i in range(0, N, _CHUNK):
                a = np.asarray(D[:, :, :, i:i + _CHUNK])           # (R, S, T, n)
                out[i:i + _CHUNK] = a.transpose(3, 1, 0, 2).astype(np.float16)
            out.flush()
            del out
        _write_manifest(cache, key, shape=[N, S, R, T])
        return
    X = loadmat(path, variable_names=["Data"])["Data"]             # (N, T, S, R)
    N, T, S, R = X.shape
    offsets, idx, cnt = [0], [], []
    for i in range(0, N, _CHUNK):
        c = recover_counts(X[i:i + _CHUNK], time_axis=1).transpose(0, 2, 3, 1)   # (n, S, R, T)
        c = c.reshape(c.shape[0], -1)
        rows, cols = np.nonzero(c)
        idx.append(cols.astype(np.uint32))
        cnt.append(c[rows, cols])
        offsets += list(offsets[-1] + np.cumsum(np.bincount(rows, minlength=c.shape[0])))
    np.savez(cache / f"{key}_spikes.npz", offsets=np.array(offsets, np.int64),
             idx=np.concatenate(idx), counts=np.concatenate(cnt))
    _write_manifest(cache, key, shape=[N, S, R, T])


def load_population(cache, area: str, method: str, bin_factor: int = 1) -> np.ndarray:
    """(N, S, R, T) float32: 2p values as released, or ephys spike counts summed
    over ``bin_factor`` consecutive 5 ms bins (T = 300 // bin_factor)."""
    cache, key = Path(cache), key_of(area, method)
    if method == "2p":
        return np.load(cache / f"{key}.npy").astype(np.float32)
    N, S, R, T0 = json.loads(_manifest(cache, key).read_text())["shape"]
    z = np.load(cache / f"{key}_spikes.npz")
    off, idx, cnt = z["offsets"], z["idx"].astype(np.int64), z["counts"].astype(np.float32)
    Tb = T0 // bin_factor
    out = np.zeros((N, S * R * Tb), np.float32)
    for n in range(N):
        sl = slice(off[n], off[n + 1])
        sr, t = np.divmod(idx[sl], T0)
        out[n] = np.bincount(sr * Tb + t // bin_factor, weights=cnt[sl], minlength=S * R * Tb)
    return out.reshape(N, S, R, Tb)


def load_twop_time(cache) -> np.ndarray:
    """The 47 released 2-photon frame times (s, relative to sound onset)."""
    return np.load(Path(cache) / "twop_t.npy")


def load_anatomy(root, area: str, method: str) -> Optional[dict]:
    """Per-unit anatomy, or None where the release has none.

    AC 2p: ``subfield`` (AI, DP, VPAF, SRAF, AAF, or None when unassigned),
    ``x_um`` / ``y_um`` (position on the brain surface after aligning the mice
    on the tonotopic map) and ``depth_um``. IC ephys: ``subdivision`` (IC-C,
    IC-D, IC-L, "outside IC", or None without histology) and Allen-atlas
    ``ap`` / ``ml`` / ``dv`` as released (NaN without histology)."""
    from scipy.io import loadmat
    f = _ANAT_FILES.get((area, method))
    if f is None:
        return None
    m = loadmat(Path(root) / f, squeeze_me=True, struct_as_record=False)
    if area == "AC":
        a, names = m["AnatInfo_AC"], [str(v) for v in m["RegionNames_AC"]]
        return {"subfield": [names[int(k) - 1] if int(k) > 0 else None for k in a.area],
                "x_um": np.asarray(a.x, float), "y_um": np.asarray(a.y, float),
                "depth_um": np.asarray(a.z, float)}
    a, names = m["AnatInfo_IC"], [str(v) for v in m["RegionNames_IC"]]
    st = np.asarray(a.Struct, int)
    none = st == 0
    sub = [None if k == 0 else (names[k - 1] if k <= len(names) else "outside IC") for k in st]
    return {"subdivision": sub, **{k: np.where(none, np.nan, np.asarray(getattr(a, v), float))
                                   for k, v in (("ap", "AP"), ("ml", "ML"), ("dv", "DV"))}}


# ----------------------------------------------------------- cochlear model
# Established on the release (2026-10-29): 657 fibres (the paper's methods say
# 590), 15 stochastic repeats, each trial baseline-subtracted, so spontaneous
# rates (and with them any hair-cell grouping) are lost. Fibres 0..~580 are
# ordered tonotopically (best frequency on the 70 dB tones rising from 5 to
# 37.4 kHz); the last ~75 respond weakly with an erratic tone best frequency,
# consistent with characteristic frequencies above the highest tone (the
# model spans 5-50 kHz). ``cochlear_cf_khz`` is therefore a smooth fit of best
# frequency on fibre index (robust, strongly driven fibres only), extrapolated
# over the tail: an APPROXIMATE label. The paper's fibres-per-hair-cell rule
# gives 590 fibres for its 40 hair cells; no hair-cell count on the same grid
# gives 657, so the grouping is not reconstructed.

def _approx_cf(bf: np.ndarray, peak: np.ndarray) -> np.ndarray:
    """Quadratic fit of log2(best frequency) on fibre index (fibre density
    varies along the cochlea), on strongly driven fibres, dropping > 0.5 octave
    outliers; linear if the quadratic is not monotonic. Release: 5.1-51 kHz,
    94.5% of fibres 0-579 within 0.25 octave of their tone best frequency."""
    k = np.arange(len(bf), dtype=np.float64)
    y = np.log2(bf)
    for deg in (2, 1):
        use = peak >= 0.5 * np.median(peak)
        for _ in range(3):
            p = np.polyfit(k[use], y[use], deg)
            use &= np.abs(y - np.polyval(p, k)) <= 0.5
        cf = 2.0 ** np.polyval(p, k)
        if np.all(np.diff(cf) > 0):
            return cf
    return cf


def prepare_cochlear(root, cache, stim_meta) -> None:
    """Cache the cochlear model averaged over its 15 repeats (float32
    (F, 300, S)), each fibre's best frequency on the 70 dB tones and the
    approximate characteristic frequency (see the notes above)."""
    from scipy.io import loadmat
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    cc = loadmat(root / "Data_CC.mat", variable_names=["Data"])["Data"].mean(axis=-1)
    cc = cc.astype(np.float32)                           # (F, 300, S)
    tones = [s for s, m in enumerate(stim_meta) if m.get("group") == "pure_70dB"]
    khz = np.array([stim_meta[s]["frequency_khz"][0] for s in tones])
    ev = cc[:, 100:200][:, :, tones].mean(axis=1)        # 0..500 ms after onset, (F, n_tones)
    bf = khz[ev.argmax(axis=1)]
    np.save(cache / "CC_mean.npy", cc)
    np.save(cache / "CC_bf_khz.npy", bf)
    np.save(cache / "CC_cf_khz.npy", _approx_cf(bf, ev.max(axis=1)))
    _write_manifest(cache, "CC", n_fibres=int(cc.shape[0]))


def load_cochlear(cache):
    """((F, 300, S) mean rates, (F,) best frequency on 70 dB tones, (F,)
    approximate characteristic frequency), kHz."""
    cache = Path(cache)
    return (np.load(cache / "CC_mean.npy"), np.load(cache / "CC_bf_khz.npy"),
            np.load(cache / "CC_cf_khz.npy"))
