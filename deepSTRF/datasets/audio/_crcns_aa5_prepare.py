"""CRCNS-AA5 one-time "slimming" step: site archives -> compact cache.

The CRCNS-AA5 release (Robotka, Gahr & Theunissen 2022) ships one
``<site>.tar.gz`` per recording site, holding one pickle per spike-sorted
unit. Each pickle duplicates the stimulus waveforms and, above all, the cage
microphone recording of every trial (``micWav``, 25 kHz), which is why the
full release is ~163 GB. The information the loader needs — spike times,
trial bookkeeping, unit metadata and one copy of each stimulus — is a tiny
fraction of that.

:func:`prepare_aa5` streams each archive **without extracting it to disk**,
unpickles one unit at a time (peak memory ~ the largest single unit pickle,
~1-2 GB), and writes, per site::

    <dest>/<bird>/<site>/
        units.json       per-unit metadata (unitInfo dict)
        playbacks.json   one entry per unique playback (trialInd): stimulus,
                         absolute start time, mic-derived alignment metrics
        spikes.npz       ragged spike times: one record per (unit, trial) with
                         its [start, stop) slice into a flat spike_times array
        manifest.json    cache version, source archive, counts (written last:
                         its presence marks the site as complete)

and, shared by all sites, ``<dest>/stimuli/<file>.wav`` (the played stimulus
over the full -0.5 .. +4.5 s response window, lossless int16 at its native
rate) plus ``<dest>/stimuli/index.json``.

The mic recording is **not** stored. Before discarding it, each playback is
cross-correlated with its stimulus to measure where the sound actually was
(``mic_offset_ms``) and how loud the off-stimulus sound was; the loader uses
these to drop the ~0.7% of playbacks whose logged onset is wrong.

The pickles were written with pandas 1.x; :class:`_AA5Unpickler` bridges the
two pandas internals that changed since (see its docstring), so they load
under pandas 1.5 as well as recent pandas.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import io
import json
import os
import pickle
import re
import shutil
import tarfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

CACHE_VERSION = 1
WINDOW_S = (-0.5, 4.5)               # response window re. logged stimulus onset
AA5_BIRDS: Tuple[str, ...] = ("ZF4F", "ZF5M", "ZF6M", "ZF7F")

# Full CRCNS-AA5 release: site -> archive size in bytes (from the CRCNS file
# listing). Used to check local copies and to drive ``download_aa5``.
AA5_SITES: Dict[str, int] = {
    "ZF4F_2t_190611_160843": 2834343757, "ZF4F_2t_190612_101201": 6289758495,
    "ZF4F_2t_190612_134337": 4343768853, "ZF4F_2t_190613_101015": 5583252055,
    "ZF4F_2t_190613_142654": 1265287545, "ZF4F_3t_190613_150928": 6976698853,
    "ZF4F_4t_190614_094759": 17784,      "ZF4F_5t_190615_114154": 3545990920,
    "ZF4F_5t_190616_102456": 1186039399, "ZF4F_6t_190616_142033": 3720594094,
    "ZF4F_6_5_190618_110932": 44537328,  "ZF4F_7t_190619_095313": 1038330954,
    "ZF4F_8t_190620_101233": 1095157918, "ZF4F_8_5t_190621_100019": 401205399,
    "ZF5M_10t_190702_095313": 5404360582, "ZF5M_10_5t_190703_133641": 4333966397,
    "ZF5M_11t_190704_100420": 6046038540, "ZF5M_11t_190704_121850": 5129143152,
    "ZF5M_11_5t_190705_102710": 3894983502, "ZF5M_12t_190707_102304": 2762856931,
    "ZF5M_6t_190625_102256": 1560644227, "ZF5M_6_5t_190625_135634": 1043340439,
    "ZF5M_7t_190625_154417": 3177488339, "ZF5M_7t_190626_101721": 679657201,
    "ZF5M_8t_190627_132428": 6784216809, "ZF5M_8t_190628_100257": 1906422425,
    "ZF5M_8t_190628_124320": 2196821200, "ZF5M_8_5t_190629_095922": 3254100649,
    "ZF5M_9t_190630_100846": 2450297699, "ZF5M_9t_190701_102100": 509520809,
    "ZF5M_9_5t_190701_125653": 2388985498,
    "ZF6M_10_5t_190808_102150": 1517344326, "ZF6M_2t_190715_172820": 1691012416,
    "ZF6M_2t_190716_085010": 3785365627, "ZF6M_6_5t_190719_094702": 4728993310,
    "ZF6M_7_5t_190723_100723": 6288992330, "ZF6M_7_5t_190723_141539": 4397202722,
    "ZF6M_8t_190725_105432": 5904987989, "ZF6M_9_5t_190805_102553": 1686476050,
    "ZF6M_9_5t_190805_130046": 2240722109,
    "ZF7F_3t_190716_130715": 1963384862, "ZF7F_4t_190717_094723": 1170276452,
    "ZF7F_4_5t_190718_091732": 1522836076, "ZF7F_5t_190720_152938": 4848100887,
    "ZF7F_5t_190720_182644": 143087879,  "ZF7F_5_5t_190722_121506": 6052265435,
    "ZF7F_6t_190724_104215": 5912130814, "ZF7F_7t_190806_115742": 6668616338,
    "ZF7F_7_5t_190807_105305": 6076169670, "ZF7F_8_5t_190809_133540": 5329502454,
}

# <bird>_<depth>_<yymmdd>_<hhmmss>; depth in microdrive turns, "6_5t" = 6.5
# (one archive, ZF4F_6_5_190618_110932, omits the trailing "t").
_SITE_RE = re.compile(r"^(?P<bird>ZF\d[MF])_(?P<depth>\d+(?:_5)?)t?_(?P<date>\d{6})_(?P<time>\d{6})$")
_UNIT_RE = re.compile(r"goodPlayback-e(?P<electrode>\d+)-c(?P<cluster>\d+)\.pkl$")
TURN_UM = 250.0                      # 1 microdrive turn = 250 um (Robotka et al. 2023, STAR Methods)


def parse_site(site: str) -> dict:
    """Split a site name into bird / depth (turns and um) / date / time."""
    m = _SITE_RE.match(site)
    if m is None:
        raise ValueError(f"Unrecognised CRCNS-AA5 site name: {site!r}")
    depth = float(m["depth"].replace("_", "."))
    return dict(bird=m["bird"], depth_turns=depth, depth_um=depth * TURN_UM,
                date=m["date"], time=m["time"])


# ---------------------------------------------------------------------------
# Unpickling
# ---------------------------------------------------------------------------

try:
    importlib.import_module("pandas.core.indexes.numeric")
    _NEED_INDEX_SHIM = False
except ImportError:            # pandas >= 2
    _NEED_INDEX_SHIM = True


def _new_block_compat(values, placement, *args, **kwargs):
    """``pandas.core.internals.blocks.new_block`` accepting a raw placement.

    Some AA5 pickles were written by a pandas version that pickled each
    DataFrame block's placement as a plain ``slice`` / array; recent pandas
    requires a ``BlockPlacement``.
    """
    from pandas._libs.internals import BlockPlacement
    from pandas.core.internals.blocks import new_block
    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)
    return new_block(values, placement, *args, **kwargs)


class _AA5Unpickler(pickle.Unpickler):
    """Unpickler for the pandas-1.x pickles of the AA5 release.

    Two pandas internals changed since the release was written:

    - ``Int64Index`` / ``Float64Index`` lived in
      ``pandas.core.indexes.numeric``, removed in pandas 2; plain ``pd.Index``
      reconstructs them faithfully;
    - blocks were rebuilt with ``new_block(values, placement=<slice>)``;
      recent pandas wants a ``BlockPlacement`` (:func:`_new_block_compat`).
    """

    def find_class(self, module, name):
        if _NEED_INDEX_SHIM and module == "pandas.core.indexes.numeric":
            import pandas as pd
            return pd.Index
        if module == "pandas.core.internals.blocks" and name == "new_block":
            return _new_block_compat
        return super().find_class(module, name)


def load_unit_pickle(fh) -> Tuple[dict, "pd.DataFrame", "pd.DataFrame"]:
    """Read (unitInfo, dfAbsTime, dfRelTime) from an open AA5 unit pickle.

    Note that unpickling executes code from the file: only use this on the
    CRCNS-AA5 archives themselves.
    """
    # one Unpickler per object, like the release notebook's three pk.load():
    # with protocol >= 4 memo indices are implicit, so a shared memo breaks
    return tuple(_AA5Unpickler(fh).load() for _ in range(3))


# ---------------------------------------------------------------------------
# Mic-based alignment check
# ---------------------------------------------------------------------------

_MIC_FS = 25000
_HOP = 125                               # 5 ms at 25 kHz
_LAG_RANGE_S = (-0.5, 4.0)
_mel_cache: dict = {}


def _mel(x: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio
    if "mel" not in _mel_cache:
        _mel_cache["mel"] = torchaudio.transforms.MelSpectrogram(
            sample_rate=_MIC_FS, n_fft=512, win_length=250, hop_length=_HOP,
            n_mels=48, f_min=250.0, f_max=10000.0, power=2.0)
    with torch.no_grad():
        return _mel_cache["mel"](torch.as_tensor(x, dtype=torch.float32)).numpy()


def _to_mic_rate(stim: np.ndarray, fs_stim: int) -> np.ndarray:
    if fs_stim == _MIC_FS:
        return stim
    from scipy.signal import resample_poly
    g = np.gcd(_MIC_FS, fs_stim)
    return resample_poly(stim, _MIC_FS // g, fs_stim // g)


def mic_alignment(stim: np.ndarray, fs_stim: int, mic: np.ndarray) -> dict:
    """Locate the played stimulus inside one trial's mic recording.

    Both signals cover the same -0.5 .. +4.5 s window. Compressed mel
    spectrograms (mic: per-band median removed, to cancel the stationary hum)
    are cross-correlated over lags -0.5 .. +4 s in 5 ms steps; the peak is
    then refined to the sample by waveform cross-correlation within +-10 ms.

    Returns ``mic_offset_ms`` (where the sound is, re. the logged onset;
    positive = later than logged), ``mic_peak`` (normalised spectral
    correlation at the best lag, in [-1, 1]), ``mic_second`` (best value
    more than 50 ms away — ``peak / second`` is a confidence ratio), and the
    loudest 20 ms of mic signal before / after the stimulus relative to the
    mic level during it (``mic_pre_peak_db`` / ``mic_post_peak_db``; high
    values flag vocalisations or other loud sounds in the response window).
    """
    stim = _to_mic_rate(np.asarray(stim, np.float64), fs_stim)
    mic = np.asarray(mic, np.float64)
    n = min(len(stim), len(mic))
    stim, mic = stim[:n], mic[:n]

    S = _mel(stim) ** 0.3
    M = _mel(mic) ** 0.3
    M = np.maximum(M - np.median(M, axis=1, keepdims=True), 0.0)
    S = S - S.mean(axis=1, keepdims=True)
    M = M - M.mean(axis=1, keepdims=True)
    T = S.shape[1]
    L = 1 << int(np.ceil(np.log2(2 * T)))
    xc = np.fft.irfft((np.fft.rfft(M, L, axis=1) * np.conj(np.fft.rfft(S, L, axis=1))).sum(0), L)
    lo, hi = (int(round(v * _MIC_FS / _HOP)) for v in _LAG_RANGE_S)
    lags = np.arange(lo, hi + 1)
    vals = xc[lags % L] / (np.linalg.norm(S) * np.linalg.norm(M) + 1e-12)
    i = int(np.argmax(vals))
    coarse = int(lags[i]) * _HOP
    far = np.abs(lags - lags[i]) > 10
    second = float(vals[far].max()) if far.any() else float("nan")

    # sample-level refinement, +-10 ms around the coarse lag
    L2 = 1 << int(np.ceil(np.log2(2 * n)))
    wx = np.fft.irfft(np.fft.rfft(mic, L2) * np.conj(np.fft.rfft(stim, L2)), L2)
    fine = np.arange(coarse - 250, coarse + 251)
    lag = int(fine[np.argmax(np.abs(wx[fine % L2]))])

    # off-stimulus loudness, measured at the corrected position
    nz = np.nonzero(np.abs(stim) > 1e-6 * (np.abs(stim).max() + 1e-12))[0]
    on, off = (nz[0] + lag, nz[-1] + lag) if len(nz) else (0, n - 1)
    on, off = int(np.clip(on, 0, n - 1)), int(np.clip(off, 0, n - 1))
    act = mic[on:off + 1]
    act_db = 10 * np.log10(np.mean(act ** 2) + 1e-20) if len(act) else np.nan
    frame = int(0.02 * _MIC_FS)

    def peak_db(x):
        if len(x) < frame:
            return float("nan")
        fr = x[: len(x) // frame * frame].reshape(-1, frame)
        return float(10 * np.log10(np.max(np.mean(fr ** 2, 1)) + 1e-20) - act_db)

    return dict(mic_offset_ms=1e3 * lag / _MIC_FS, mic_peak=float(vals[i]),
                mic_second=second,
                mic_pre_peak_db=peak_db(mic[: max(on - frame // 2, 0)]),
                mic_post_peak_db=peak_db(mic[min(off + int(0.1 * _MIC_FS), n):]))


# ---------------------------------------------------------------------------
# Sources: tar.gz archives or already-extracted site folders
# ---------------------------------------------------------------------------

def _find_sources(src: Path) -> Dict[str, Path]:
    """Map site name -> archive (``<site>.tar.gz``) or extracted site dir."""
    out: Dict[str, Path] = {}
    for p in sorted(src.rglob("*.tar.gz")):
        name = p.name[: -len(".tar.gz")]
        if _SITE_RE.match(name) and not p.name.startswith("._"):
            out[name] = p
    for p in sorted(src.rglob("PlaybackPkl")):
        name = p.parent.name
        if p.is_dir() and _SITE_RE.match(name) and name not in out:
            out[name] = p.parent
    return out


def _iter_unit_pickles(source: Path) -> Iterator[Tuple[str, "io.BufferedIOBase"]]:
    """Yield (pickle file name, readable stream), one unit at a time."""
    if source.is_dir():
        for p in sorted((source / "PlaybackPkl").glob("goodPlayback-*.pkl")):
            with open(p, "rb") as fh:
                yield p.name, fh
        return
    with tarfile.open(source, mode="r|gz") as tf:
        for m in tf:
            base = m.name.rsplit("/", 1)[-1]
            if m.isfile() and base.endswith(".pkl") and not base.startswith("._"):
                yield base, tf.extractfile(m)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, default=_json_default, indent=1)


def _as_int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _as_float(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def is_site_prepared(dest: Union[str, Path], site: str) -> bool:
    bird = parse_site(site)["bird"]
    mf = Path(dest) / bird / site / "manifest.json"
    if not mf.exists():
        return False
    try:
        return json.loads(mf.read_text()).get("cache_version") == CACHE_VERSION
    except (OSError, json.JSONDecodeError):
        return False


def _store_stimulus(stim_dir: Path, index: dict, file: str, wav: np.ndarray, fs: int) -> str:
    """Write the stimulus once (lossless int16 when integer-valued). Returns sha1."""
    import soundfile as sf
    w = np.asarray(wav, np.float64)
    integer = np.allclose(w, np.round(w)) and np.abs(w).max() <= 32767
    data = np.round(w).astype(np.int16) if integer else w.astype(np.float32)
    sha1 = hashlib.sha1(data.tobytes()).hexdigest()
    if file in index:
        if index[file]["sha1"] != sha1:
            warnings.warn(f"CRCNS-AA5: stimulus {file!r} differs between sites "
                          f"(sha1 {index[file]['sha1'][:8]} vs {sha1[:8]}); keeping the first copy.")
        return sha1
    sf.write(stim_dir / file, data, fs, subtype="PCM_16" if integer else "FLOAT")
    index[file] = dict(fs=int(fs), n_samples=int(len(data)), sha1=sha1,
                       t_start_s=WINDOW_S[0], integer_pcm=bool(integer))
    return sha1


def prepare_site(source: Union[str, Path], dest: Union[str, Path], *,
                 mic_alignment_check: bool = True, overwrite: bool = False,
                 progress: bool = True) -> Path:
    """Slim ONE site (archive or extracted folder) into ``dest``. Idempotent.

    Writes into a temporary folder and renames it into place at the end, so
    an interrupted run never leaves a half-written site behind.
    """
    import pandas as pd  # noqa: F401  (needed by the unpickler)

    source, dest = Path(source), Path(dest)
    site = source.name[: -len(".tar.gz")] if source.name.endswith(".tar.gz") else source.name
    info_site = parse_site(site)
    out = dest / info_site["bird"] / site
    if not overwrite and is_site_prepared(dest, site):
        return out

    stim_dir = dest / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)
    index_path = stim_dir / "index.json"
    stim_index = json.loads(index_path.read_text()) if index_path.exists() else {}

    tmp = out.with_name(f".{site}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    units: List[dict] = []
    playbacks: Dict[int, dict] = {}
    rec_unit, rec_trial, rec_start, rec_stop = [], [], [], []
    spikes: List[np.ndarray] = []
    n_spk = 0

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(desc=f"AA5 {site}", unit="unit")
        except ImportError:
            bar = None

    for fname, fh in _iter_unit_pickles(source):
        info, dfabs, dfrel = load_unit_pickle(fh)
        m = _UNIT_RE.search(fname)
        u_idx = len(units)
        units.append(dict(
            cell_id=f"{site}_{fname[len('goodPlayback-'):-len('.pkl')]}",
            file=fname, bird=info.get("Bird", info_site["bird"]), site=site,
            electrode=_as_int(info.get("Electrode")) or (int(m["electrode"]) if m else None),
            cluster=int(m["cluster"]) if m else None,
            sort=_as_int(info.get("Sort")),
            spike_snr=_as_float(info.get("SpikeSNR")),
            rate_threshold=_as_float(info.get("RateThreshold")),
            kde_bw=_as_float(info.get("KDE_BW")),
            n_stim=_as_int(info.get("nStim")) or 0,
            spike_shape=np.asarray(info.get("SpikeShape", []), float).round(4).tolist(),
            spike_std=np.asarray(info.get("SpikeStd", []), float).round(4).tolist(),
        ))
        for _, row in dfrel.iterrows():
            fs = int(round(1.0 / (row.tStim[1] - row.tStim[0])))
            sha1 = _store_stimulus(stim_dir, stim_index, row.file, row.stimWav, fs)
            for k, (ti, sp) in enumerate(zip(row.trialInd, row.spikeTimes)):
                ti = int(ti)
                sp = np.asarray(sp, np.float64).ravel()
                rec_unit.append(u_idx)
                rec_trial.append(ti)
                rec_start.append(n_spk)
                n_spk += len(sp)
                rec_stop.append(n_spk)
                spikes.append(sp)
                if ti not in playbacks:
                    a = dfabs.loc[ti] if ti in dfabs.index else None
                    pb = dict(trial=ti, file=row.file, call_type=row.call_type,
                              relation_short=row.relation_short, stim_sha1=sha1,
                              start_time=_as_float(a["start_time"]) if a is not None else None,
                              duration=_as_float(a["duration"]) if a is not None else None)
                    if mic_alignment_check:
                        try:
                            pb.update(mic_alignment(row.stimWav, fs, row.micWav[k]))
                        except Exception as e:  # pragma: no cover - defensive
                            warnings.warn(f"CRCNS-AA5 mic alignment failed for {site} trial {ti}: {e}")
                    playbacks[ti] = pb
        del info, dfabs, dfrel
        if bar is not None:
            bar.update(1)
            bar.set_postfix(playbacks=len(playbacks))
    if bar is not None:
        bar.close()

    # inter-playback timing: time to the next playback onset (can fall inside
    # the +4.5 s response window, as ISIs are 3-6 s after the stimulus ends)
    starts = sorted((pb["start_time"], t) for t, pb in playbacks.items() if pb["start_time"] is not None)
    for (s0, t0), (s1, _t1) in zip(starts[:-1], starts[1:]):
        playbacks[t0]["next_onset_s"] = s1 - s0

    _write_json(tmp / "units.json", units)
    _write_json(tmp / "playbacks.json", [playbacks[t] for t in sorted(playbacks)])
    np.savez_compressed(
        tmp / "spikes.npz",
        unit=np.asarray(rec_unit, np.int32), trial=np.asarray(rec_trial, np.int32),
        start=np.asarray(rec_start, np.int64), stop=np.asarray(rec_stop, np.int64),
        spike_times=np.concatenate(spikes) if spikes else np.zeros(0))
    _write_json(index_path, stim_index)
    from deepSTRF import __version__
    _write_json(tmp / "manifest.json", dict(
        cache_version=CACHE_VERSION, deepstrf_version=__version__, site=site,
        **info_site, source=source.name,
        source_bytes=source.stat().st_size if source.is_file() else None,
        n_units=len(units), n_playbacks=len(playbacks), n_trial_records=len(rec_unit),
        window_s=list(WINDOW_S), mic_alignment_check=mic_alignment_check,
        created=_dt.datetime.now().isoformat(timespec="seconds")))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return out


def prepare_aa5(src: Union[str, Path], dest: Union[str, Path], *,
                sites: Optional[Iterable[str]] = None,
                birds: Optional[Iterable[str]] = None,
                mic_alignment_check: bool = True,
                delete_sources: bool = False,
                overwrite: bool = False,
                progress: bool = True) -> List[str]:
    """Slim every CRCNS-AA5 site found under ``src`` into the cache ``dest``.

    Works with whatever subset of the release is present: archives
    (``<site>.tar.gz``, at any depth under ``src``) and/or already-extracted
    ``<site>/PlaybackPkl/`` folders. Sites already in the cache are skipped,
    so the call can be re-run after downloading more archives.

    Parameters
    ----------
    src, dest : path-like
        Folder holding the raw release / folder for the slim cache (the two
        may be the same).
    sites, birds : iterable of str, optional
        Restrict to these site names and/or birds (``AA5_BIRDS``).
    mic_alignment_check : bool, default True
        Measure each playback's actual onset from the mic before discarding
        it (~5-10 ms per playback).
    delete_sources : bool, default False
        Delete each archive / extracted folder after its site is cached.
    overwrite : bool, default False
        Rebuild sites even if already cached.

    Returns
    -------
    list of str
        Names of the sites now in the cache (prepared or already there).
    """
    src, dest = Path(src).expanduser(), Path(dest).expanduser()
    found = _find_sources(src)
    if sites is not None:
        wanted = set(sites)
        unknown = wanted - set(AA5_SITES)
        if unknown:
            raise ValueError(f"Unknown CRCNS-AA5 site(s): {sorted(unknown)}")
        found = {k: v for k, v in found.items() if k in wanted}
    if birds is not None:
        bset = set(birds)
        found = {k: v for k, v in found.items() if parse_site(k)["bird"] in bset}
    done = []
    for site, source in found.items():
        if source.is_file() and source.stat().st_size != AA5_SITES.get(site, source.stat().st_size):
            warnings.warn(f"CRCNS-AA5: {source.name} is {source.stat().st_size} bytes, expected "
                          f"{AA5_SITES[site]} — incomplete download? Skipping.")
            continue
        prepare_site(source, dest, mic_alignment_check=mic_alignment_check,
                     overwrite=overwrite, progress=progress)
        done.append(site)
        if delete_sources and source.resolve() != (dest / parse_site(site)["bird"] / site).resolve():
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
    return sorted(set(done) | {s for s in AA5_SITES if is_site_prepared(dest, s)})


def download_aa5(dest: Union[str, Path], *,
                 sites: Optional[Iterable[str]] = None,
                 birds: Optional[Iterable[str]] = None,
                 keep_archives: bool = False,
                 mic_alignment_check: bool = True,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 progress: bool = True) -> List[str]:
    """Download CRCNS-AA5 site archives one at a time and slim each into ``dest``.

    Each archive (0.02-7 GB) is downloaded to ``dest/_archives/<bird>/``,
    slimmed, then deleted unless ``keep_archives=True`` — so the disk
    footprint never exceeds one archive plus the cache. Sites already cached
    are skipped. Needs a free CRCNS account (``$CRCNS_USERNAME`` /
    ``$CRCNS_PASSWORD``).

    Each archive's path inside the dataset is looked up in the CRCNS file
    list (:func:`~deepSTRF.utils.data_download.crcns_file_list`), and its
    size checked against ``AA5_SITES``.
    """
    from deepSTRF.utils.data_download import crcns_download, crcns_file_list

    dest = Path(dest).expanduser()
    todo = list(AA5_SITES)
    if sites is not None:
        wanted = set(sites)
        unknown = wanted - set(AA5_SITES)
        if unknown:
            raise ValueError(f"Unknown CRCNS-AA5 site(s): {sorted(unknown)}")
        todo = [s for s in todo if s in wanted]
    if birds is not None:
        todo = [s for s in todo if parse_site(s)["bird"] in set(birds)]
    todo = [s for s in todo if not is_site_prepared(dest, s)]
    if not todo:
        return sorted(s for s in AA5_SITES if is_site_prepared(dest, s))

    remote = {Path(p).name: (p, size) for p, size in
              crcns_file_list("aa-5", username=username, password=password).items()}
    for site in todo:
        entry = remote.get(f"{site}.tar.gz")
        if entry is None:
            raise RuntimeError(f"CRCNS-AA5: {site}.tar.gz is not in the aa-5 file list.")
        rel, size = entry
        if size != AA5_SITES[site]:
            warnings.warn(f"CRCNS-AA5: {rel} is listed at {size} bytes, expected "
                          f"{AA5_SITES[site]}; downloading anyway.")
        archive = dest / "_archives" / parse_site(site)["bird"] / f"{site}.tar.gz"
        if not archive.exists():
            crcns_download(f"aa-5/{rel}", archive,
                           username=username, password=password, progress=progress)
        prepare_site(archive, dest, mic_alignment_check=mic_alignment_check, progress=progress)
        if not keep_archives:
            archive.unlink()
    return sorted(s for s in AA5_SITES if is_site_prepared(dest, s))
