"""Prepare step for :class:`~deepSTRF.datasets.audio.yang2025.Yang2025Dataset`.

Turns the raw Zenodo release of Yang & Long (2025) — budgerigar anterior arcopallium
(AAC) single units recorded while the birds vocalise, with a piezo skull microphone
on the same 30 kHz clock — into a slim cache:

    <dest>/index.json                       episodes (with splits) + per-bird units
    <dest>/<bird>/spikes.npz                spike_times (s, session clock), unit (int)
    <dest>/<bird>/audio/<bout_id>_<i>.flac  int16 piezo audio around each episode

Only the vocal episodes (annotated warble bouts and calls) plus a margin are kept,
so the 3.4 GB of raw spikes + audio shrink to a few hundred MB.

Reference: Yang, Z. & Long, M. A. (2025). Convergent vocal representations in
parrot and human forebrain motor networks. Nature 640, 427-434.
doi:10.1038/s41586-025-08695-8. Data: Zenodo 10.5281/zenodo.14057061 (CC BY 4.0).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

BIRDS = ("Ti81", "Li145", "Bl122", "Or61")
ZENODO_RECORD = 14057061
AUDIO_FS = 30000
CACHE_VERSION = 2
RAW_MEMBERS = (
    "bgI.mat",
    *(f"{b}_ChronicLeftAAC/{f}" for b in BIRDS for f in ("sua.mat", "audioCh3_HP.flac")),
)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- episodes
def build_episodes(sy_time_win: np.ndarray, warble_flag: np.ndarray,
                   bouts: Sequence[Tuple[float, float]], *, bird: str,
                   min_duration_s: float = 0.005,
                   splices_s: Sequence[float] = ()) -> List[dict]:
    """One dict per vocal episode: each warble bout, and each call element.

    ``sy_time_win`` ``(n, 2)`` are the annotated elements (s), ``warble_flag``
    marks the warble ones (the others are calls), ``bouts`` the warble bouts'
    ``(start, end)``. Episodes shorter than ``min_duration_s`` are dropped; an
    episode crossing a splice of the recording (``splices_s``) is cut there into
    two episodes sharing one ``bout_id``. Sorted by onset.
    """
    sy = np.atleast_2d(np.asarray(sy_time_win, dtype=float))
    flag = np.asarray(warble_flag).astype(bool)
    raw = [("warble", f"{bird}_w{k:03d}", float(a), float(b)) for k, (a, b) in enumerate(bouts)]
    raw += [("call", f"{bird}_c{k:03d}", float(a), float(b))
            for k, (a, b) in enumerate(sy[~flag])]
    eps = []
    for typ, bid, on, off in raw:
        cuts = [on] + [s for s in sorted(splices_s) if on < s < off] + [off]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= min_duration_s:
                eps.append(dict(bird=bird, episode_type=typ, bout_id=bid, onset_s=a, offset_s=b))
    eps.sort(key=lambda e: e["onset_s"])
    return eps


def _split_units(episodes: List[dict], min_gap_s: float) -> List[str]:
    """Split-unit key of each episode: episodes of one bird that share a
    ``bout_id`` or lie less than ``min_gap_s`` apart form one unit (keyed by
    its first ``bout_id``), so that no item's pre-roll reaches into an episode
    of another split."""
    parent = list(range(len(episodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    first_of = {}
    for i, e in enumerate(episodes):
        union(first_of.setdefault((e["bird"], e["bout_id"]), i), i)
    by_bird: Dict[str, List[int]] = {}
    for i, e in enumerate(episodes):
        by_bird.setdefault(e["bird"], []).append(i)
    for idx in by_bird.values():
        idx.sort(key=lambda i: episodes[i]["onset_s"])
        reach, last = -np.inf, None
        for i in idx:
            if last is not None and episodes[i]["onset_s"] - reach < min_gap_s:
                union(last, i)
            if episodes[i]["offset_s"] >= reach:
                reach, last = episodes[i]["offset_s"], i
    roots = [find(i) for i in range(len(episodes))]
    key_of = {}
    for i in sorted(range(len(episodes)), key=lambda i: (episodes[i]["bird"], episodes[i]["onset_s"])):
        key_of.setdefault(roots[i], episodes[i]["bout_id"])
    return [key_of[r] for r in roots]


def assign_splits(episodes: List[dict], fractions: Sequence[float] = (0.70, 0.15, 0.15),
                  seed: int = 0, min_gap_s: float = 1.0) -> None:
    """Write a ``'split'`` (train / val / test) into every episode dict, in place.

    Split units are whole bouts, merged with every episode of the same bird
    less than ``min_gap_s`` away (default 1 s: the authors' bout criterion, and
    longer than any pre-roll). Per bird x unit type (``warble`` if the unit
    holds any warble, else ``call``), units are dealt out: in a seeded random
    order, test then val are filled up to their share of the stratum's
    duration, skipping any unit that would push the split above 1.25 x its
    share — so an outsized bout always falls to train. A stratum with at least
    3 units gets at least one in every split.
    """
    rng = np.random.default_rng(seed)
    unit_of = _split_units(episodes, min_gap_s)
    is_warble: Dict[str, bool] = {}
    for e, u in zip(episodes, unit_of):
        is_warble[u] = is_warble.get(u, False) or e["episode_type"] == "warble"
    strata: Dict[Tuple[str, str], Dict[str, float]] = {}
    for e, u in zip(episodes, unit_of):
        dur = strata.setdefault((e["bird"], "warble" if is_warble[u] else "call"), {})
        dur[u] = dur.get(u, 0.0) + e["offset_s"] - e["onset_s"]
    split_of: Dict[str, str] = {}
    for key in sorted(strata):
        dur = strata[key]
        units = sorted(dur)
        order = [units[i] for i in rng.permutation(len(units))]
        total = sum(dur.values())
        free = list(order)
        for split, frac in (("test", fractions[2]), ("val", fractions[1])):
            target, got, picked = frac * total, 0.0, []
            for u in list(free):
                if got >= target:
                    break
                if got + dur[u] <= 1.25 * target:
                    picked.append(u)
                    got += dur[u]
            if not picked and len(units) >= 3 and len(free) > 1:
                picked = [min(free, key=lambda u: dur[u])]
            for u in picked:
                free.remove(u)
                split_of[u] = split
        for u in free:
            split_of[u] = "train"
    for e, u in zip(episodes, unit_of):
        e["split"] = split_of[u]


# --------------------------------------------------------------------------- raw files
def _splices_from_ks_dir(ks_dir: str) -> List[float]:
    """Splice times (s, on the concatenated audio) from a ``..._cutA-B_C-D`` folder name."""
    m = re.search(r"cut((?:\d+-\d+_?)+)", ks_dir)
    if not m:
        return []
    segs = [tuple(map(float, s.split("-"))) for s in m.group(1).strip("_").split("_")]
    return list(np.cumsum([b - a for a, b in segs])[:-1])


def _load_bgi(raw: Path) -> Dict[str, object]:
    import scipy.io as sio
    bg = sio.loadmat(raw / "bgI.mat", squeeze_me=True, struct_as_record=False)["bgI"]
    return {str(b.birdId): b for b in np.atleast_1d(bg)}


def _load_units(raw: Path, bird: str):
    import scipy.io as sio
    sua = sio.loadmat(raw / f"{bird}_ChronicLeftAAC" / "sua.mat", squeeze_me=True,
                      struct_as_record=False, variable_names=["sua"])["sua"]
    return list(np.atleast_1d(sua))


def prepare_yang2025(raw: PathLike, dest: PathLike, *, margin_s: float = 1.0) -> Path:
    """Build the slim cache in ``dest`` from the raw release in ``raw``.

    ``raw`` holds ``bgI.mat`` and the four ``<bird>_ChronicLeftAAC/`` folders
    (``sua.mat`` + ``audioCh3_HP.flac``), as written by :func:`download_yang2025`.
    Each episode's audio is kept over ``[onset - margin_s, offset + margin_s]``,
    clipped to the recording segment it lies in; spikes are kept inside those
    windows. Returns ``dest``.
    """
    import soundfile as sf

    raw, dest = Path(raw).expanduser(), Path(dest).expanduser()
    bgi = _load_bgi(raw)
    episodes, units = [], {}
    windows: Dict[str, List[Tuple[float, float]]] = {}
    for bird in BIRDS:
        b = bgi[bird]
        audio_path = raw / f"{bird}_ChronicLeftAAC" / "audioCh3_HP.flac"
        info = sf.info(str(audio_path))
        if info.samplerate != AUDIO_FS:
            raise ValueError(f"{audio_path}: expected {AUDIO_FS} Hz, got {info.samplerate}")
        session_s = info.frames / AUDIO_FS
        splices = _splices_from_ks_dir(str(b.myKsDir))
        bounds = [0.0] + splices + [session_s]
        bouts = [(float(w.start), float(w.end)) for w in np.atleast_1d(b.warbleInfo)]
        eps = build_episodes(b.syTimeWin, b.warbleSyFlag, bouts, bird=bird, splices_s=splices)

        (dest / bird / "audio").mkdir(parents=True, exist_ok=True)
        windows[bird] = []
        n_of: Dict[str, int] = {}
        for e in eps:
            seg = max(i for i in range(len(bounds) - 1) if bounds[i] <= e["onset_s"])
            t0 = max(e["onset_s"] - margin_s, bounds[seg])
            t1 = min(e["offset_s"] + margin_s, bounds[seg + 1])
            i0, i1 = int(round(t0 * AUDIO_FS)), int(round(t1 * AUDIO_FS))
            wav, _ = sf.read(str(audio_path), start=i0, stop=i1, dtype="int16")
            i = n_of[e["bout_id"]] = n_of.get(e["bout_id"], -1) + 1
            rel = f"{bird}/audio/{e['bout_id']}_{i}.flac"
            sf.write(str(dest / rel), wav, AUDIO_FS, subtype="PCM_16")
            e.update(audio_file=rel, audio_t0_s=i0 / AUDIO_FS, audio_t1_s=i1 / AUDIO_FS)
            windows[bird].append((i0 / AUDIO_FS, i1 / AUDIO_FS))
        episodes += eps

        sua = _load_units(raw, bird)
        win = []                                    # union of the (possibly overlapping) windows
        for a, z in sorted(windows[bird]):
            if win and a <= win[-1][1]:
                win[-1][1] = max(win[-1][1], z)
            else:
                win.append([a, z])
        win = np.asarray(win)
        times, unit_idx = [], []
        for n, u in enumerate(sua):
            st = np.atleast_1d(np.asarray(u.spikeTimes, dtype=float))
            j = np.searchsorted(win[:, 0], st, side="right") - 1
            keep = (j >= 0) & (st < win[np.clip(j, 0, None), 1])
            times.append(st[keep])
            unit_idx.append(np.full(keep.sum(), n, np.int32))
        np.savez(dest / bird / "spikes.npz", spike_times=np.concatenate(times),
                 unit=np.concatenate(unit_idx))
        units[bird] = [dict(cluster_id=int(u.clusterID), shank=int(u.shank),
                            max_channel=int(u.maxChannelTemplate),
                            coordinates=[float(c) for c in np.atleast_1d(u.coordinates)],
                            fr=float(u.fr)) for u in sua]

    assign_splits(episodes)
    index = dict(version=CACHE_VERSION, audio_fs=AUDIO_FS, birds=list(BIRDS),
                 source="Yang & Long 2025, Nature 640:427; Zenodo 10.5281/zenodo.14057061 (CC BY 4.0)",
                 episodes=episodes, units=units)
    (dest / "index.json").write_text(json.dumps(index, indent=1))
    return dest


def download_yang2025(raw_dest: PathLike, progress: bool = True) -> Path:
    """Fetch the needed members of the 15.7 GB ``Data.zip`` (about 3.3 GB) into ``raw_dest``."""
    from deepSTRF.utils.data_download import zenodo_zip_members_download
    zenodo_zip_members_download(ZENODO_RECORD, "Data.zip", RAW_MEMBERS, raw_dest,
                                progress=progress)
    return Path(raw_dest).expanduser()
