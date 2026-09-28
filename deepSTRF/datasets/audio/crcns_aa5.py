"""CRCNS-AA5 — awake, freely behaving zebra finches hearing their whole vocal repertoire.

Reference
---------
Robotka H., Thomas L., Yu K., Wood W., Elie J.E., Gahr M. & Theunissen F.E.
(2023). "Sparse ensemble neural code for a complete vocal repertoire."
*Cell Reports* 42(2): 112034. doi:10.1016/j.celrep.2023.112034

Data: Robotka H., Gahr M. & Theunissen F.E. (2022). "Simultaneous
extracellular recordings of avian auditory neurons in freely behaving zebra
finches presented with all the repertoire of vocalizations used by this
species for vocal communication." CRCNS.org. doi:10.6080/K0TT4P5Q

Extracellular units recorded with chronic 16-channel arrays in the auditory
pallium (Field L, CLM, NCM, CMM) of 4 freely behaving adult zebra finches
(ZF4F, ZF5M, ZF6M, ZF7F), in response to 110 calls spanning the 10 call types
of the repertoire, 20 songs plus spectrally / temporally filtered versions,
and (at some sites) 10 random ripples. Each stimulus was played ~10 times per
site; per-unit trial counts are lower because periods with motion artifacts
were excluded unit by unit.

The raw release is ~163 GB of per-unit pickles; this loader reads the compact
cache written by :func:`prepare_aa5` (a one-time streaming pass, ~10 s per GB
of archive). See the dataset README in the deepSTRF docs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._crcns_aa5_prepare import (  # noqa: F401  (re-exported)
    AA5_BIRDS,
    AA5_SITES,
    WINDOW_S,
    download_aa5,
    is_site_prepared,
    parse_site,
    prepare_aa5,
)
from deepSTRF.utils.data import hanning_smooth
from deepSTRF.utils.data_download import default_cache_dir

STIM_CLASSES = ("call", "song", "song_sfilt", "song_tfilt", "ripple")

# <Vocalizer>_<code>_<CallType>_<rendition>_fs25k.wav, e.g.
# "BlaBla0506_MAF_Te_8-1-9_fs25k.wav". The vocalizer ID and call types follow
# the Elie & Theunissen (2016) repertoire library. The 3-letter code is not
# documented: its first letter matches the vocalizer's sex (M / F / U) and its
# second its age (A = adult, C = chick; the juvenile Be / LT calls always
# carry C) — we expose those two as *inferred* fields and keep the raw code.
_CALL_RE = re.compile(r"^(?P<vocalizer>[A-Za-z]{6}\d{4})_(?P<code>[A-Z]{3})_(?P<call_type>[A-Za-z]{2})_"
                      r"(?P<rendition>[^_]+)_fs25k\.wav$")
_SONG_RE = re.compile(r"^stim(?P<song>\d+)(?:_(?P<filt>[st]filt))?\.wav$")
_RIPPLE_RE = re.compile(r"^randripple(?P<ripple>\d+)\.wav$")

CALL_TYPES = {
    "Ag": "Wsst / aggressive", "Be": "begging", "DC": "distance call", "Di": "distress",
    "LT": "long tonal", "Ne": "nest", "So": "song", "Te": "tet", "Th": "thuk", "Wh": "whine",
}


def _stim_info(name: str) -> dict:
    """Parse a stimulus file name into its ``stim_meta`` fields."""
    info = dict(name=name, stim_class=None, call_type=None, vocalizer=None,
                vocalizer_code=None, vocalizer_sex=None, vocalizer_age=None,
                rendition=None, song_id=None, ripple_id=None)
    m = _CALL_RE.match(name)
    if m:
        code = m["code"]
        info.update(stim_class="call", call_type=m["call_type"], vocalizer=m["vocalizer"],
                    vocalizer_code=code,
                    vocalizer_sex={"M": "M", "F": "F"}.get(code[0]),
                    vocalizer_age={"A": "adult", "C": "chick"}.get(code[1]),
                    rendition=m["rendition"])
        return info
    m = _SONG_RE.match(name)
    if m:
        info.update(stim_class="song" if m["filt"] is None else f"song_{m['filt']}",
                    call_type="So", song_id=int(m["song"]))
        return info
    m = _RIPPLE_RE.match(name)
    if m:
        info.update(stim_class="ripple", ripple_id=int(m["ripple"]))
        return info
    info["stim_class"] = "unknown"
    return info


class CRCNSAA5Dataset(AudioNeuralDataset):
    """PyTorch dataset for CRCNS-AA5 (Robotka, Gahr & Theunissen).

    Awake, freely behaving zebra finches; chronic extracellular recordings in
    the auditory pallium; the full vocal repertoire as stimuli. Population-
    and batch-compatible. Works with any subset of the 50 recording sites.

    Every stimulus is presented in the release's fixed **5 s response window,
    -0.5 s to +4.5 s around the logged onset** — pre- and post-stimulus
    silence included — so ``stims[s]`` and every ``responses[s][n]`` share
    ``T = 5000 / dt_ms`` bins, with the sound starting at bin
    ``0.5 s / dt``.

    Notes
    -----
    Follows the standard deepSTRF data paradigm (see
    ``docs/_source/md/data_paradigm.md``). AA5-specific points:

    - **Repeats** are the unit's *kept* trials (motion-artifact periods were
      excluded unit by unit by the authors), so ``R`` varies per
      (stim, unit), typically 2-10. ``self.trial_ids[s][n]`` lists the
      site-wide playback index of each row, so units recorded
      simultaneously can be re-aligned trial by trial.
    - **Onset check.** The cage-mic recording (not stored) was used during
      :func:`prepare_aa5` to locate each playback. Playbacks whose sound was
      confidently *not* at the logged onset (~0.9%, up to 2.5 s off) are
      dropped by default (``drop_misaligned``).
    - **Next playback.** In ~2% of playbacks the next stimulus starts before
      +4.5 s; those trials are dropped by default
      (``drop_next_playback_overlap``).
    - **Levels.** Playback level was deliberately varied per stimulus to match
      natural loudness (Robotka et al. 2023), and the wav amplitudes carry
      it: stimuli are never normalised per stimulus.
    - ``self.playback_meta[site]`` holds the per-playback table: stimulus,
      absolute start time, ``mic_offset_ms`` / ``mic_peak`` / ``mic_second``
      (onset check), ``mic_pre_peak_db`` / ``mic_post_peak_db`` (loudest
      20 ms of mic signal before / after the sound, in dB re. the mic level
      during the sound — relative, so unreliable at quiet-mic sites such as
      ZF6M_10_5t), ``next_onset_s``, and the resulting flags
      ``misaligned``, ``next_playback_overlap`` and ``dropped``.

    ``stim_meta`` keys: ``name`` (wav file name, the canonical id),
    ``stim_class`` (one of ``STIM_CLASSES``), ``call_type`` (2-letter code,
    see ``CALL_TYPES``; ``'So'`` for songs, ``None`` for ripples),
    ``vocalizer``, ``vocalizer_code`` (raw 3-letter code),
    ``vocalizer_sex`` / ``vocalizer_age`` (inferred from that code),
    ``rendition``, ``song_id``, ``ripple_id``, ``native_fs``,
    ``sound_onset_s`` / ``sound_offset_s`` (first / last non-zero sample,
    re. the logged onset), ``duration_s`` and ``level_db`` (RMS over the
    sound, dB re. int16 full scale).

    ``nrn_meta`` keys: ``cell_id``, ``animal_id`` (bird), ``sex``, ``site``,
    ``depth_turns`` / ``depth_um`` (microdrive depth; 1 turn = 250 um),
    ``recording_date``, ``electrode``, ``cluster``, ``sort_id``,
    ``spike_snr`` (the authors' spike-waveform SNR; their single-unit
    threshold is 5 — not to be confused with the response ``'snr'`` written
    by :meth:`compute_neuron_quality`), ``rate_threshold`` / ``kde_bw``
    (authors' artifact-exclusion parameters), ``auditory_z`` (mean / SD over
    all the unit's trials of the firing-rate change 0-500 ms after vs
    500-0 ms before onset — the effect size behind the paper's "auditory"
    t-test, recomputed here; negative = inhibited by sound), ``n_trials``,
    ``spike_shape`` / ``spike_std`` (mean spike waveform, 40 samples).
    No per-unit brain-area label is released.
    """

    def __init__(self,
                 path: Optional[Union[str, Path]] = None,
                 animals: Union[str, Sequence[str]] = "all",
                 sites: Optional[Sequence[str]] = None,
                 stimuli: Sequence[str] = STIM_CLASSES,
                 dt_ms: float = 5.0,
                 smooth: bool = True,
                 n_mels: int = 32,
                 compression: str = "cubic",
                 window_ms: float = 10.0,
                 audio_fs: int = 25000,
                 return_waveform: bool = False,
                 min_trials: int = 1,
                 drop_misaligned: bool = True,
                 misalign_tol_ms: float = 25.0,
                 drop_next_playback_overlap: bool = True,
                 raw_path: Optional[Union[str, Path]] = None,
                 download: bool = False,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Parameters
        ----------
        path : path-like, optional
            Folder of the slim cache written by :func:`prepare_aa5`.
            Defaults to ``default_cache_dir('AA5')``.
        animals : 'all' or sequence of str
            Birds to load (subset of ``AA5_BIRDS``).
        sites : sequence of str, optional
            Restrict to these recording sites (names as in ``AA5_SITES``).
        stimuli : sequence of str
            Stimulus classes to keep, subset of ``STIM_CLASSES``.
        dt_ms : float, default 5.0
            Time-bin width in ms. Must divide the 5000 ms window. The default
            keeps the full release in ~0.8 GB of responses (1 ms: ~3.7 GB).
        smooth : bool, default True
            Smooth each trial with a 21 ms Hanning window (Hsu, Borst &
            Theunissen 2004; same kernel as the other CRCNS-AA loaders).
        n_mels, compression, window_ms
            Mel spectrogram: number of bands, compression (``'cubic'``,
            ``'log1p'`` or ``'none'``), analysis window length. Frames are
            causal: frame ``t`` summarises the ``window_ms`` of audio ending
            at the end of bin ``t``.
        audio_fs : int, default 25000
            Common sample rate: calls are 25 kHz, songs and ripples 32 kHz and
            are resampled so that every stimulus gets the same mel filterbank.
            ``audio_fs * dt_ms / 1000`` must be an integer.
        return_waveform : bool, default False
            Return ``(1, T * hop)`` waveforms at ``audio_fs`` (in int16
            full-scale units) instead of mel spectrograms.
        min_trials : int, default 1
            (stim, unit) pairs with fewer valid trials become NaN sentinels.
        drop_misaligned : bool, default True
            Drop playbacks whose sound the mic check confidently located more
            than ``misalign_tol_ms`` away from the logged onset.
        drop_next_playback_overlap : bool, default True
            Drop playbacks whose window (up to +4.5 s) already contains the
            onset of the next playback (~2%). Dropping the trial, rather than
            NaN-ing its tail, follows the data paradigm: any NaN inside
            ``responses[s][n]`` marks the whole (stim, unit) pair as missing.
        raw_path : path-like, optional
            Folder holding (part of) the raw release (``<site>.tar.gz``
            archives or extracted site folders). Any site found there and not
            yet cached is slimmed into ``path`` first.
        download : bool, default False
            Fetch missing sites from CRCNS (one archive at a time, slimmed
            then deleted; needs ``$CRCNS_USERNAME`` / ``$CRCNS_PASSWORD``).
        """
        path = Path(default_cache_dir("AA5") if path is None else path).expanduser()
        birds = AA5_BIRDS if animals == "all" else tuple(animals)
        unknown = set(birds) - set(AA5_BIRDS)
        if unknown:
            raise ValueError(f"Unknown AA5 bird(s) {sorted(unknown)}; choose from {AA5_BIRDS}")
        bad_cls = set(stimuli) - set(STIM_CLASSES)
        if bad_cls:
            raise ValueError(f"Unknown stimulus class(es) {sorted(bad_cls)}; choose from {STIM_CLASSES}")
        if raw_path is not None:
            prepare_aa5(raw_path, path, sites=sites, birds=birds)
        if download:
            download_aa5(path, sites=sites, birds=birds, username=username, password=password)

        super().__init__(str(path), dt_ms)
        n_bins = 1000.0 * (WINDOW_S[1] - WINDOW_S[0]) / dt_ms
        if abs(n_bins - round(n_bins)) > 1e-6:
            raise ValueError(f"dt_ms={dt_ms} must divide the {1000 * (WINDOW_S[1] - WINDOW_S[0]):.0f} ms window")
        hop = audio_fs * dt_ms / 1000.0
        if abs(hop - round(hop)) > 1e-6:
            raise ValueError(f"audio_fs * dt_ms / 1000 = {hop} must be an integer")
        self.T = int(round(n_bins))
        self.onset_bin = int(round(-WINDOW_S[0] * 1000.0 / dt_ms))
        self.species = "zebra finch"
        self.hearing_range_hz = (250.0, 8000.0)
        self.F = int(n_mels)
        self.compression = compression
        self.window_ms = float(window_ms)
        self.return_waveform = bool(return_waveform)
        self.audio_fs = int(audio_fs)
        self.animals = birds
        self.stim_types = tuple(stimuli)

        # ---------------------------------------------------------- sites
        site_dirs = []
        for bird in birds:
            for d in sorted((path / bird).glob("*/manifest.json")):
                site = d.parent.name
                if sites is None or site in set(sites):
                    site_dirs.append(d.parent)
        if not site_dirs:
            raise FileNotFoundError(
                f"No prepared CRCNS-AA5 site under {path} for birds={birds}"
                + (f", sites={list(sites)}" if sites is not None else "")
                + ". Run prepare_aa5(<raw release folder>, path) or pass raw_path= / download=True.")
        self.sites = [d.name for d in site_dirs]

        # ---------------------------------------------------------- stimuli
        stim_index = json.loads((path / "stimuli" / "index.json").read_text())
        needed = set()
        site_data = []
        for d in site_dirs:
            pbs = json.loads((d / "playbacks.json").read_text())
            pbs = [p for p in pbs if _stim_info(p["file"])["stim_class"] in self.stim_types]
            site_data.append((d, pbs))
            needed.update(p["file"] for p in pbs)
        stim_names = sorted(needed, key=lambda f: (STIM_CLASSES.index(_stim_info(f)["stim_class"])
                                                    if _stim_info(f)["stim_class"] in STIM_CLASSES else 99, f))
        s_of = {f: i for i, f in enumerate(stim_names)}
        self.stims, self.stim_meta = [], []
        for f in stim_names:
            wav, meta = self._load_stim(path / "stimuli" / f, stim_index[f])
            self.stims.append(wav)
            self.stim_meta.append(meta)

        # ---------------------------------------------------------- responses
        nan_sentinel = torch.full((1, 1), float("nan"))
        columns = []                 # per unit: {s: (tensor, trial_ids)}
        self.nrn_meta = []
        self.playback_meta: Dict[str, List[dict]] = {}
        for d, pbs in site_data:
            site = d.name
            site_info = parse_site(site)
            units = json.loads((d / "units.json").read_text())
            with np.load(d / "spikes.npz") as npz:             # read each array once
                z = {k: npz[k] for k in npz.files}
            st, starts, stops = z["spike_times"], z["start"], z["stop"]
            pb_by_trial = {p["trial"]: p for p in pbs}
            for p in pbs:
                peak = p.get("mic_peak") or 0.0
                conf = peak > 0.2 and peak / max(p.get("mic_second") or 1e-3, 1e-3) > 1.5
                p["misaligned"] = bool(conf and abs(p.get("mic_offset_ms", 0.0)) > misalign_tol_ms)
                nxt = p.get("next_onset_s")
                p["next_playback_overlap"] = bool(nxt is not None and nxt < WINDOW_S[1])
                p["dropped"] = bool((drop_misaligned and p["misaligned"]) or
                                    (drop_next_playback_overlap and p["next_playback_overlap"]))
            self.playback_meta[site] = pbs

            # bin every trial record of the site at once: (n_records, T) counts
            n_rec = len(starts)
            rec_of_spike = np.repeat(np.arange(n_rec), stops - starts)
            b = np.floor((st - WINDOW_S[0]) * 1000.0 / dt_ms).astype(np.int64)
            ok = (b >= 0) & (b < self.T)
            counts = np.zeros((n_rec, self.T), np.float32)
            np.add.at(counts, (rec_of_spike[ok], b[ok]), 1.0)
            # auditory effect size, over ALL of the unit's trials (any stim class):
            # rate change 0-500 ms after vs 500-0 ms before onset
            pre = np.bincount(rec_of_spike[(st >= -0.5) & (st < 0.0)], minlength=n_rec)
            post = np.bincount(rec_of_spike[(st >= 0.0) & (st < 0.5)], minlength=n_rec)
            dz = (post - pre) / 0.5

            for ui, u in enumerate(units):
                sel = np.nonzero(z["unit"] == ui)[0]
                if len(sel) == 0:
                    continue                                   # empty unit (nStim = 0)
                rows: Dict[int, List[int]] = {}
                for k in sel:
                    p = pb_by_trial.get(int(z["trial"][k]))
                    if p is None or p["dropped"]:
                        continue
                    rows.setdefault(s_of[p["file"]], []).append(int(k))
                col = {}
                for s, ks in rows.items():
                    if len(ks) < min_trials:
                        continue
                    ks.sort(key=lambda k: int(z["trial"][k]))
                    R = torch.from_numpy(counts[ks])
                    if smooth:
                        R = hanning_smooth(R, window_ms=21.0, dt_ms=dt_ms)
                    col[s] = (R, [int(z["trial"][k]) for k in ks])
                if not col:
                    continue
                dd = dz[sel]
                columns.append(col)
                self.nrn_meta.append(dict(
                    cell_id=u["cell_id"], animal_id=u["bird"], sex=u["bird"][-1], site=site,
                    depth_turns=site_info["depth_turns"], depth_um=site_info["depth_um"],
                    recording_date=site_info["date"], electrode=u["electrode"],
                    cluster=u["cluster"], sort_id=u["sort"], spike_snr=u["spike_snr"],
                    rate_threshold=u["rate_threshold"], kde_bw=u["kde_bw"],
                    auditory_z=float(dd.mean() / dd.std()) if len(dd) > 1 and dd.std() > 0 else float("nan"),
                    n_trials=int(sum(len(v[1]) for v in col.values())),
                    spike_shape=u["spike_shape"], spike_std=u["spike_std"]))

        self.N_neurons = len(self.nrn_meta)
        self.responses = [[columns[n][s][0] if s in columns[n] else nan_sentinel
                           for n in range(self.N_neurons)] for s in range(len(self.stims))]
        self.trial_ids = [[columns[n][s][1] if s in columns[n] else []
                           for n in range(self.N_neurons)] for s in range(len(self.stims))]
        self.validate()

    # ------------------------------------------------------------------
    def _load_stim(self, wav_path: Path, idx: dict):
        w, sr = sf.read(str(wav_path), dtype="float64", always_2d=False)
        if idx.get("integer_pcm", True):
            w = w * 32768.0                                    # back to the release's int16 units
        w = torch.tensor(w / 32768.0, dtype=torch.float32)     # int16 full-scale units
        meta = _stim_info(wav_path.name)
        nz = torch.nonzero(w.abs() > 0).flatten()
        on = (nz[0].item() / sr + WINDOW_S[0]) if len(nz) else float("nan")
        off = (nz[-1].item() / sr + WINDOW_S[0]) if len(nz) else float("nan")
        level = 20 * np.log10(float(w[w.abs() > 0].pow(2).mean().sqrt()) + 1e-12) if len(nz) else float("nan")
        meta.update(native_fs=int(sr), sound_onset_s=on, sound_offset_s=off,
                    duration_s=off - on, level_db=level)

        if sr != self.audio_fs:
            w = torchaudio.functional.resample(w, sr, self.audio_fs)
        hop = int(round(self.audio_fs * self.dt / 1000.0))
        n = self.T * hop
        w = torch.nn.functional.pad(w, (0, max(0, n - w.shape[-1])))[:n]
        if self.return_waveform:
            return w.unsqueeze(0).contiguous(), meta

        n_fft = max(int(round(self.window_ms * 1e-3 * self.audio_fs)), hop)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=self.audio_fs, n_fft=n_fft,
                                                   hop_length=hop, n_mels=self.F, center=False)
        # causal framing: left-pad so frame t covers the n_fft samples ending
        # at the end of bin t, i.e. [(t+1)*hop - n_fft, (t+1)*hop)
        spec = mel(torch.nn.functional.pad(w, (n_fft - hop, 0)))[..., : self.T]
        if self.compression == "cubic":
            spec = spec.pow(1.0 / 3)
        elif self.compression == "log1p":
            spec = torch.log1p(spec)
        elif self.compression != "none":
            raise ValueError(f"Unknown compression {self.compression!r}")
        return spec.unsqueeze(0), meta

    # ------------------------------------------------------------------
    def select_call_type(self, call_type: str) -> List[int]:
        """Restrict to stimuli of one call type (e.g. ``'DC'``); returns their indices."""
        return self.select_stims_by_attr("call_type", call_type)

    def select_stim_class(self, stim_class: str) -> List[int]:
        """Restrict to one stimulus class (see ``STIM_CLASSES``); returns their indices."""
        return self.select_stims_by_attr("stim_class", stim_class)
