"""Bagur 2025 — mouse inferior colliculus, auditory thalamus and auditory cortex.

Reference
---------
Bagur S, Bourg J, Kempf A, Tarpin T, Bergaoui K, Guo Y, Ceballo S,
Schwenkgrub J, Verdier A, Puel JL, Bourien J, Bathellier B (2025). "A spatial
code for temporal information is necessary for efficient sensory learning."
*Science Advances* 11(2): eadr6214. https://doi.org/10.1126/sciadv.adr6214

Data: Zenodo record 13941450.

Awake, head-fixed mice heard 140 sounds (tones, intensity ramps, AM sounds,
frequency sweeps, a blank), 13-15 times each. Five populations: two-photon
calcium imaging of auditory cortex (AC) somata, of thalamic (TH) axon boutons
in AC and of dorsal inferior colliculus (IC) somata (deconvolved GCaMP6s,
~31.7 ms frames), and silicon-probe recordings in central IC and in TH (spike
counts, 5 ms bins). The release also holds a cochlear model driven by the same
sounds, exposed here as an alternative stimulus representation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._logspectrogram import n_bands_for
from deepSTRF.datasets.audio import _trial_audio as _ta
from deepSTRF.datasets.audio import _bagur2025_stims as bs
from deepSTRF.datasets.audio import _bagur2025_prepare as bp
from deepSTRF.utils.data_download import default_cache_dir

#: Behavioural audiogram, 60 dB SPL limits: Koay G, Heffner RS, Heffner HE
#: (2002), Hear Res 171:111-118 (control mice, F1 C57BL/6J x C3H/HeJ — not
#: pure C57BL/6J). Informational only.
BAGUR_HEARING_RANGE_HZ = (2300.0, 86000.0)

EPHYS_DT_CHOICES = (5, 10, 20, 25, 50, 100)
SPEC_BACKENDS = ("logspec", "cochlear_model")
_UNIT_TYPE = {("AC", "2p"): "soma ROI", ("IC", "2p"): "soma ROI", ("TH", "2p"): "axon bouton",
              ("IC", "ephys"): "single unit", ("TH", "ephys"): "single unit"}
_RECORDING_TYPE = {"2p": "2p-calcium", "ephys": "extracellular-silicon-probe"}


def _best_freq(evoked: np.ndarray, freqs: Sequence[float]) -> Optional[float]:
    """Frequency of the tone with the largest evoked response, or None when no
    tone evokes a response above baseline."""
    ev = np.nan_to_num(np.asarray(evoked, dtype=np.float64), nan=-np.inf)
    i = int(np.argmax(ev))
    return float(freqs[i]) if ev[i] > 0 else None


def _num(v) -> Optional[float]:
    return None if not np.isfinite(v) else float(v)


class Bagur2025Dataset(AudioNeuralDataset):
    """Mouse IC / TH / AC responses to 140 sounds, imaging or spikes.

    One instance loads one recording method (one time base) and any of its
    populations: ``method="2p"`` with ``areas`` among ``"AC"``, ``"TH"``
    (thalamic axon boutons imaged in AC) and ``"IC"``; ``method="ephys"`` with
    ``"IC"`` and ``"TH"``. Each trial runs from 500 ms before the sound onset
    to +1 s; the stimulus is the matching trial buffer.

    See the Bagur 2025 dataset page of the deepSTRF docs for every
    ``stim_meta`` / ``nrn_meta`` key and the known caveats.

    Examples
    --------
    >>> ds = Bagur2025Dataset(areas="IC", method="ephys", download=True)   # doctest: +SKIP
    >>> split = ds.group_split(seed=0)
    """

    def __init__(self, path: Optional[str] = None, *,
                 areas: Union[str, Iterable[str]],
                 method: str,
                 dt_ms: Optional[float] = None,
                 return_waveform: bool = False,
                 audio_fs: int = bs.NATIVE_FS,
                 spec_backend: str = "logspec",
                 fmin: float = 2000.0,
                 fmax: float = 64000.0,
                 bins_per_octave: int = 6,
                 window_ms: Optional[float] = None,
                 smooth: bool = False,
                 cache_dir: Optional[str] = None,
                 prepare: bool = True,
                 download: bool = False):
        """
        Parameters
        ----------
        path : str, optional
            Folder holding the release files. Defaults to the platformdirs
            cache (``$DEEPSTRF_DATA_DIR`` overrides).
        areas : str or iterable of str
            Required. ``"AC"``, ``"TH"``, ``"IC"`` (2p) or ``"IC"``, ``"TH"``
            (ephys); populations are loaded in the order given.
        method : {"2p", "ephys"}
            Required. Two-photon imaging (deconvolved, signed, ~31.7 ms frames)
            or electrophysiology (raw spike counts, 5 ms bins).
        dt_ms : float, optional
            Ephys: 5 (default, native), 10, 20, 25, 50 or 100 ms, so the onset
            and the 1.5 s window fall on bin edges. 2p: the frame period only.
        return_waveform : bool, default False
            Hand out the raw ``(1, T_audio)`` trial waveform at 192 kHz (int16 /
            32768, absolute level kept) instead of a spectrogram.
        audio_fs : int, default 192000
            Waveform sample rate. Only the native rate is supported.
        spec_backend : {"logspec", "cochlear_model"}, default "logspec"
            ``"logspec"``: log-spaced Hamming spectrogram in dB with a fixed
            absolute floor (16-bit quantisation noise), so level differences
            between sounds are kept. ``"cochlear_model"``: the release's
            cochlear-model fibre rates (mean of its 15 repeats) as the
            ``(1, 657, T)`` input.
        fmin, fmax, bins_per_octave : float, float, int
            Spectrogram bands (logspec only): 2-64 kHz at 6 / octave = 31.
        window_ms : float, optional
            Spectrogram analysis window; default ``2 * dt``.
        smooth : bool, default False
            Ephys: smooth each response with a 21 ms Hanning window. Not
            allowed for 2p, which the authors already smoothed.
        cache_dir : str, optional
            Where the slim cache lives. Defaults to ``<path>/_cache``.
        prepare : bool, default True
            Build a missing cache from the release files.
        download : bool, default False
            Fetch the files the call needs from Zenodo, md5-checked.
        """
        asked = (areas,) if isinstance(areas, str) else tuple(areas)
        self._check_args(asked, method, dt_ms, return_waveform, audio_fs, spec_backend, smooth)
        areas = tuple(dict.fromkeys(asked))
        root = Path(path).expanduser() if path is not None else default_cache_dir("Bagur2025")
        cache = Path(cache_dir).expanduser() if cache_dir else root / "_cache"
        cochlear = spec_backend == "cochlear_model"
        keys = tuple(bp.key_of(a, method) for a in areas) + (("CC",) if cochlear else ())
        prepared = tuple(k for k in keys if bp.is_prepared(cache, k))
        if download:
            bp.download_bagur2025(root, areas, method, cochlear=cochlear, prepared=prepared)
        missing = [f for f in bp.required_files(
            areas, method, cochlear, prepared, have_wavs=(root / "sounds" / "Sounds").is_dir(),
            have_code=(root / "code" / "Code").is_dir()) if not (root / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Bagur2025: missing {missing} in {root}. Pass download=True, or fetch them "
                f"from https://doi.org/10.5281/zenodo.{bs.ZENODO_RECORD}.")
        stim_meta = bs.build_stim_table(bp.code_dir(root), bp.wav_dir(root))
        for a in areas:
            if bp.key_of(a, method) not in prepared:
                if not prepare:
                    raise FileNotFoundError(f"cache for {a}-{method} missing in {cache}; "
                                            "pass prepare=True")
                bp.prepare_population(root, cache, a, method)
        if cochlear and "CC" not in prepared:
            if not prepare:
                raise FileNotFoundError(f"cochlear-model cache missing in {cache}; pass prepare=True")
            bp.prepare_cochlear(root, cache, stim_meta)

        if method == "2p":
            self.twop_frame_times_s = bp.load_twop_time(cache)
            period = float(np.diff(self.twop_frame_times_s).mean()) * 1000.0
            dt = round(period * bs.NATIVE_FS / 1000.0) * 1000.0 / bs.NATIVE_FS
            self._buffer_start_ms = float(self.twop_frame_times_s[0]) * 1000.0 - dt / 2.0
            T = len(self.twop_frame_times_s)
        else:
            dt = float(bs.EPHYS_DT_MS if dt_ms is None else dt_ms)
            self._buffer_start_ms = -bs.ONSET_MS
            T = int(bs.EPHYS_BINS * bs.EPHYS_DT_MS / dt)
        super().__init__(str(root), dt)
        self.stim_meta = stim_meta
        self.areas, self.method = areas, method
        self.species = "house mouse (Mus musculus), C57BL/6J"
        self.hearing_range_hz = BAGUR_HEARING_RANGE_HZ
        self.recording_type = _RECORDING_TYPE[method]
        self.onset_ms = bs.ONSET_MS
        self.zenodo_record = bs.ZENODO_RECORD
        self.return_waveform = bool(return_waveform)
        self.spec_backend = spec_backend
        if method == "2p":
            self.indicator = "GCaMP6s"
            self.response_units = "deconvolved"

        self._load_responses(root, cache, areas, method, T)
        self._build_stims(cache, bp.wav_dir(root), T, fmin, fmax, bins_per_octave, window_ms)
        if smooth:
            self.smooth_responses(window_ms=21.0)
        units, _ = bs.split_units(self.stim_meta, bs.load_duplicate_groups())
        for m, u in zip(self.stim_meta, units):
            m["split_group"] = u
        self.validate()

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _check_args(areas, method, dt_ms, return_waveform, audio_fs, spec_backend, smooth):
        if method not in bs.METHODS:
            raise ValueError(f"method must be one of {bs.METHODS} (got {method!r}).")
        bad = [a for a in areas if a not in bs.AREAS]
        if not areas or bad:
            raise ValueError(f"unknown area(s) {bad}; choose from {bs.AREAS}")
        if method == "ephys" and "AC" in areas:
            raise ValueError("AC was only imaged (method='2p'); electrophysiology covers IC and TH.")
        if method == "2p":
            if smooth:
                raise ValueError("smooth=True is for spike counts: the 2p responses are already "
                                 "smoothed by the authors (Gaussian, sigma = 31 ms).")
            if dt_ms is not None:
                raise ValueError("2p data come at the imaging frame period: leave dt_ms=None.")
        elif dt_ms is not None and float(dt_ms) not in EPHYS_DT_CHOICES:
            raise ValueError(f"ephys dt_ms must be one of {EPHYS_DT_CHOICES} ms so that the onset "
                             f"and the 1.5 s window fall on bin edges (got {dt_ms}).")
        if audio_fs != bs.NATIVE_FS:
            raise ValueError(f"audio_fs must be the native {bs.NATIVE_FS} (got {audio_fs}).")
        if spec_backend not in SPEC_BACKENDS:
            raise ValueError(f"spec_backend must be one of {SPEC_BACKENDS} (got {spec_backend!r}).")
        if return_waveform and spec_backend == "cochlear_model":
            raise ValueError("return_waveform=True and spec_backend='cochlear_model' exclude each "
                             "other: the cochlear model replaces the waveform as the input.")

    # ------------------------------------------------------------- responses
    def _bin_masks(self, T):
        starts = self._buffer_start_ms + np.arange(T) * self.dt          # bin start, ms from onset
        pre = starts + self.dt <= 0.0
        evoked = (starts >= 0.0) & (starts < 500.0)
        return pre, evoked

    def _load_responses(self, root, cache, areas, method, T):
        self.responses = [[] for _ in self.stim_meta]
        self.nrn_meta = []
        pt70 = [s for s, m in enumerate(self.stim_meta) if m["group"] == "pure_70dB"]
        pt70_khz = [self.stim_meta[s]["frequency_khz"][0] for s in pt70]
        factor = 1 if method == "2p" else int(round(self.dt / bs.EPHYS_DT_MS))
        pre, evoked = self._bin_masks(T)
        for area in areas:
            arr = bp.load_population(cache, area, method, factor)        # (N, S, R, T)
            anat = bp.load_anatomy(root, area, method)
            t_all = torch.from_numpy(arr)
            psth = arr.mean(axis=2)                                        # (N, S, T)
            base = psth[:, :, pre].mean(axis=2)
            ev = psth[:, :, evoked].mean(axis=2) - base
            for n in range(arr.shape[0]):
                for s in range(arr.shape[1]):
                    self.responses[s].append(t_all[n, s])
                meta = {
                    "cell_id": f"{area}-{method}_{n:05d}",
                    "area": area,
                    "method": method,
                    "recording_type": _RECORDING_TYPE[method],
                    "unit_type": _UNIT_TYPE[(area, method)],
                    "index_in_release": n,
                    "best_freq_khz": _best_freq(ev[n, pt70], pt70_khz) if pt70 else None,
                    # 2p traces are baseline-subtracted per trial: no spontaneous rate
                    "spont_rate": None if method == "2p" else float(base[n].mean() * 1000.0 / self.dt),
                }
                if anat is not None:
                    for k, v in anat.items():
                        meta[k] = v[n] if isinstance(v, list) else _num(v[n])
                self.nrn_meta.append(meta)
        self.N_neurons = len(self.nrn_meta)

    # ---------------------------------------------------------------- stimuli
    def _onset_samples(self) -> int:
        return int(round(-self._buffer_start_ms * bs.NATIVE_FS / 1000.0))

    def _build_stims(self, cache, wdir, T, fmin, fmax, bpo, window_ms):
        self.stims = []
        onset = self._onset_samples()
        if self.spec_backend == "cochlear_model":
            cc, self.cochlear_bf_khz, self.cochlear_cf_khz = bp.load_cochlear(cache)
            # model rates on the 5 ms grid from -500 ms, averaged onto this instance's bins
            W = _ta.overlap_weights(cc.shape[1], bs.EPHYS_DT_MS, self.dt, T,
                                    offset_ms=self._buffer_start_ms + bs.ONSET_MS)
            for s in range(len(self.stim_meta)):
                self.stims.append(torch.from_numpy((cc[:, :, s] @ W.T).astype(np.float32))[None])
            self.F = int(cc.shape[0])
            return
        if self.return_waveform:
            self.audio_fs = bs.NATIVE_FS
            self.F = n_bands_for(fmin, fmax, bpo)            # target width for a wav2spec front end
            hop = int(round(self.dt * bs.NATIVE_FS / 1000.0))
            for m in self.stim_meta:
                buf = _ta.trial_waveform(bs.load_wav_int16(wdir, m["name"]), T, hop, onset) / 32768.0
                self.stims.append(torch.from_numpy(buf.astype(np.float32))[None])
            return
        for m in self.stim_meta:
            spec = bs.logspec_trial(bs.load_wav_int16(wdir, m["name"]), T, self.dt, fmin, fmax,
                                    bpo, window_ms, onset_samples=onset)
            self.stims.append(torch.from_numpy(spec)[None])
        self.F = int(self.stims[0].shape[1])

    # ----------------------------------------------------------------- split
    def group_split(self, seed: Optional[int] = None,
                    fractions: Sequence[float] = (0.7, 0.1, 0.2)) -> Dict[str, List[int]]:
        """Split the 140 sounds into train / val / test without leakage.

        Sounds that are the same waveform up to a gain stay together: the
        60 / 70 / 80 dB copies of each tone, the 60 / 80 dB copies of each
        chirp, and the four 500 ms chirps that the release also ships under a
        second name. Everything else is a separate sound, including the
        time-reversed (down vs up) ramps and chirps. Units are drawn category
        by category, each to the set furthest below its target share; the
        blank (silence) always goes to train. The draw covers all 140 sounds
        whatever is loaded, so a seed gives the same split for every
        population.

        Parameters
        ----------
        seed : int, optional
            Seed of the draw. ``None`` draws a new random split at each call.
        fractions : sequence of 3 floats, default (0.7, 0.1, 0.2)
            Target train / val / test shares, in sounds.

        Returns
        -------
        dict
            ``{'train', 'val', 'test'}``: absolute stimulus indices. Use
            :meth:`stim_to_iter_indices` to build a ``Subset``.
        """
        units, strata = bs.split_units(self.stim_meta, bs.load_duplicate_groups())
        keep = [i for i, m in enumerate(self.stim_meta) if m["category"] != "blank"]
        sp = _ta.draw_split([units[i] for i in keep], [strata[i] for i in keep], seed, fractions)
        out = {k: sorted(keep[j] for j in v) for k, v in sp.items()}
        out["train"] = sorted(out["train"] + [i for i in range(len(units)) if i not in keep])
        return out
