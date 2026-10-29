"""Gosselin 2025 — mouse cochlear nucleus, inferior colliculus and auditory cortex.

Reference
---------
Gosselin E, Bagur S, Jamali S, Puel J-L, Bourien J, Bathellier B (2025).
"Sound feature representations decorrelate across the mouse auditory
pathway." *PLOS Biology* 23(10): e3003452.
https://doi.org/10.1371/journal.pbio.3003452

Data: Zenodo record 17271813 (CC-BY-4.0).

Awake, head-fixed, passively listening mice heard the same 307 sounds,
12 times each: cochlear nucleus (CN) and inferior colliculus (IC) recorded
with Neuropixels (spike counts at 1 ms), auditory cortex (AC) imaged with
two-photon calcium microscopy (deconvolved GCaMP6s, 43.5 ms bins, separate
mice). The release also holds a biophysical auditory-nerve model driven by
the same sounds, exposed here as an alternative stimulus representation.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._logspectrogram import n_bands_for
from deepSTRF.datasets.audio import _gosselin2025_stims as gs
from deepSTRF.datasets.audio import _gosselin2025_cochlear as gc
from deepSTRF.datasets.audio import _gosselin2025_prepare as gp
from deepSTRF.datasets.audio import _gosselin2025_split as gsplit
from deepSTRF.utils.data_download import default_cache_dir

#: Behavioural audiogram, 60 dB SPL limits: Koay G, Heffner RS, Heffner HE
#: (2002), Hear Res 171:111-118 (control mice, F1 C57BL/6J x C3H/HeJ — not
#: pure C57BL/6J). Informational only.
GOSSELIN_HEARING_RANGE_HZ = (2300.0, 86000.0)

AREAS = ("CN", "IC", "AC")
EPHYS_DT_CHOICES = (1, 2, 4, 5, 10, 20, 25, 50, 100)
SPEC_BACKENDS = ("logspec", "cochlear_model")
COCHLEAR_CHANNELS = ("ihc", "fibres")
COCHLEAR_TRANSFORMS = (None, "log1p")


def _best_freq(evoked: np.ndarray, freqs: Sequence[float]) -> Optional[float]:
    """Frequency of the tone with the largest evoked response, or None when no
    tone evokes a response above baseline (a neuron silent or suppressed by
    every tone has no best frequency)."""
    ev = np.nan_to_num(np.asarray(evoked, dtype=np.float64), nan=-np.inf)
    i = int(np.argmax(ev))
    return float(freqs[i]) if ev[i] > 0 else None


class Gosselin2025Dataset(AudioNeuralDataset):
    """Mouse CN / IC (Neuropixels) or AC (2-photon) responses to 307 sounds.

    Every neuron heard all 307 sounds: pure tones, intensity ramps, chords,
    amplitude-modulated sounds, frequency sweeps, noises, 15 natural-like
    clips (forward and time-reversed, at 50 and 70 dB) and decompositions of
    four of them. Each trial runs from 200 ms before the sound onset to
    +700 ms (CN / IC) or +844 ms (AC); the stimulus is the matching trial
    buffer with the sound at +200 ms.

    See ``docs/_source/md/README_Gosselin_2025.md`` for every ``stim_meta`` /
    ``nrn_meta`` key and the known caveats: CN / IC units fade across repeats
    and the zeros are shipped as released; AC values are signed deconvolved
    calcium (pair them with ``mse_loss``).

    Examples
    --------
    >>> ds = Gosselin2025Dataset(areas=("IC",), download=True)          # doctest: +SKIP
    >>> split = ds.group_split(seed=0)
    """

    def __init__(self, path: Optional[str] = None, *,
                 areas: Union[str, Iterable[str]],
                 dt_ms: Optional[float] = None,
                 return_waveform: bool = False,
                 audio_fs: int = gs.NATIVE_FS,
                 spec_backend: str = "logspec",
                 fmin: float = 2000.0,
                 fmax: float = 64000.0,
                 bins_per_octave: int = 6,
                 window_ms: Optional[float] = None,
                 cochlear_channels: str = "ihc",
                 cochlear_transform: Optional[str] = None,
                 smooth: bool = False,
                 cache_dir: Optional[str] = None,
                 prepare: bool = True,
                 download: bool = False):
        """
        Parameters
        ----------
        path : str, optional
            Folder holding the release files (``Dataset_*.npy``, ``Info_*.csv``,
            ``Sound Parameters.csv``, ``Soundset.zip``). Defaults to the
            platformdirs cache (``$DEEPSTRF_DATA_DIR`` overrides).
        areas : str or iterable of str
            Required. Any of ``"CN"`` and ``"IC"``, or ``"AC"`` alone: the
            2-photon cortex data has its own time base and cannot be mixed
            with spikes. A single area can be given as a plain string. There
            is no default: CN + IC at the native 1 ms take ~26 GB of RAM.
        dt_ms : float, optional
            Response bin width. CN / IC: one of 1, 2, 4, 5, 10, 20, 25, 50,
            100 ms, so the 200 ms onset and the 900 ms window fall on bin
            edges; default 1 ms, the native resolution (float32 responses:
            ~19 GB for CN and ~7 GB for IC at 1 ms, ~3.8 / 1.5 GB at 5 ms).
            AC: 43.5 ms only (the imaging bins; the default).
        return_waveform : bool, default False
            Hand out the raw ``(1, T_audio)`` trial waveform at 192 kHz
            (int16 / 32768, absolute level kept) instead of a spectrogram.
        audio_fs : int, default 192000
            Waveform sample rate. Only the native rate is supported.
        spec_backend : {"logspec", "cochlear_model"}, default "logspec"
            ``"logspec"``: log-spaced Hamming spectrogram in dB with a fixed
            absolute floor (16-bit quantisation noise), so level differences
            between sounds are kept. ``"cochlear_model"``: the release's
            auditory-nerve model rates as the ``(1, F, T)`` input.
        fmin, fmax, bins_per_octave : float, float, int
            Spectrogram bands (logspec only): 2-64 kHz at 6 / octave = 31.
        window_ms : float, optional
            Spectrogram analysis window; default ``2 * dt_ms``.
        cochlear_channels : {"ihc", "fibres"}, default "ihc"
            ``"ihc"``: 51 channels, the mean rate of each inner hair cell's
            fibres. ``"fibres"``: all 691 model fibres (reads
            ``Dataset_CC.npy`` memory-mapped; it is not cached).
        cochlear_transform : {None, "log1p"}, default None
            None keeps the model's linear rates (spikes/s), as the authors
            use them; ``"log1p"`` compresses them.
        smooth : bool, default False
            CN / IC: smooth each response with a 21 ms Hanning window
            (:meth:`smooth_responses`). Not allowed for AC, which the authors
            already smoothed (Gaussian, sigma = 31 ms).
        cache_dir : str, optional
            Where the slim cache lives. Defaults to ``<path>/_cache``.
        prepare : bool, default True
            Build a missing cache from the release files (~40 s for all).
        download : bool, default False
            Fetch the files needed for ``areas`` (and the auditory-nerve model
            when ``spec_backend="cochlear_model"``) from Zenodo, md5-checked.
        """
        asked = (areas,) if isinstance(areas, str) else tuple(areas)
        bad = [a for a in asked if a not in AREAS]
        if bad:
            raise ValueError(f"unknown area(s) {bad}; choose from {AREAS}")
        areas = tuple(a for a in AREAS if a in asked)          # canonical order CN, IC, AC
        self._check_args(areas, dt_ms, return_waveform, audio_fs, spec_backend,
                         cochlear_channels, cochlear_transform, smooth)
        is_ac = areas == ("AC",)
        dt = (gs.AC_DT_MS if is_ac else 1.0) if dt_ms is None else float(dt_ms)

        root = Path(path).expanduser() if path is not None else default_cache_dir("Gosselin2025")
        super().__init__(str(root), dt)
        cochlear = spec_backend == "cochlear_model"
        fibres = cochlear and cochlear_channels == "fibres"
        cache = Path(cache_dir).expanduser() if cache_dir else root / "_cache"
        keys = areas + (("CC",) if cochlear else ())
        prepared = tuple(k for k in keys if gp.is_prepared(cache, k))   # their arrays: not needed
        if download:
            gp.download_gosselin2025(root, areas, cochlear=cochlear, prepared=prepared)
        have_wavs = (root / "soundset" / "Soundset").is_dir()
        missing = [f for f in gp.required_files(areas, cochlear, fibres, prepared, have_wavs)
                   if not (root / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Gosselin2025: missing {missing} in {root}. Pass download=True, or fetch them "
                f"from https://doi.org/10.5281/zenodo.{gs.ZENODO_RECORD}.")
        for key in keys:
            if key not in prepared:
                if not prepare:
                    raise FileNotFoundError(f"cache for {key} missing in {cache}; pass prepare=True")
                if key == "CC":
                    gp.prepare_cochlear(root, cache)
                else:
                    gp.prepare_area(root, cache, key)

        self.areas = areas
        self.species = "house mouse (Mus musculus), C57BL/6J"
        self.hearing_range_hz = GOSSELIN_HEARING_RANGE_HZ
        self.onset_ms = gs.ONSET_MS
        self.zenodo_record = gs.ZENODO_RECORD
        self.return_waveform = bool(return_waveform)
        self.spec_backend = spec_backend
        if is_ac:
            self.recording_type = "2p-calcium"
            self.indicator = "GCaMP6s"
            self.response_units = "deconvolved"
        else:
            self.recording_type = "extracellular-neuropixels"

        wdir = gp.wav_dir(root)
        self.stim_meta = gs.build_stim_table(root / "Sound Parameters.csv", wdir)
        T = gs.AC_BINS if is_ac else gs.EPHYS_BINS // int(dt)

        self._load_responses(root, cache, areas, None if is_ac else int(dt))
        self._build_stims(root, cache, wdir, T, dt, fmin, fmax, bins_per_octave, window_ms,
                          cochlear_channels, cochlear_transform)
        if smooth:
            self.smooth_responses(window_ms=21.0)
        units, _ = gsplit.split_units(self.stim_meta, gs.load_gain_copies())
        for m, u in zip(self.stim_meta, units):
            m["split_group"] = u
        self.validate()

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _check_args(areas, dt_ms, return_waveform, audio_fs, spec_backend,
                    cochlear_channels, cochlear_transform, smooth):
        if not areas:
            raise ValueError(f"unknown area(s); choose from {AREAS}")
        if "AC" in areas and len(areas) > 1:
            raise ValueError("AC (2-photon, 43.5 ms bins) cannot be loaded together with CN/IC "
                             "(1 ms spikes); make one instance per time base.")
        if areas == ("AC",):
            if smooth:
                raise ValueError("smooth=True is for CN / IC spikes: AC is already smoothed "
                                 "by the authors (Gaussian, sigma = 31 ms).")
            if dt_ms is not None and float(dt_ms) != gs.AC_DT_MS:
                raise ValueError(f"AC is binned at the imaging rate: dt_ms must be "
                                 f"{gs.AC_DT_MS} (got {dt_ms}).")
        elif dt_ms is not None and float(dt_ms) not in EPHYS_DT_CHOICES:
            raise ValueError(f"CN/IC dt_ms must be one of {EPHYS_DT_CHOICES} ms so that the "
                             f"200 ms onset and the 900 ms window fall on bin edges (got {dt_ms}).")
        if audio_fs != gs.NATIVE_FS:
            raise ValueError(f"audio_fs must be the native {gs.NATIVE_FS} (got {audio_fs}).")
        if spec_backend not in SPEC_BACKENDS:
            raise ValueError(f"spec_backend must be one of {SPEC_BACKENDS} (got {spec_backend!r}).")
        if return_waveform and spec_backend == "cochlear_model":
            raise ValueError("return_waveform=True and spec_backend='cochlear_model' exclude each "
                             "other: the cochlear model replaces the waveform as the input.")
        if cochlear_channels not in COCHLEAR_CHANNELS:
            raise ValueError(f"cochlear_channels must be one of {COCHLEAR_CHANNELS}.")
        if cochlear_transform not in COCHLEAR_TRANSFORMS:
            raise ValueError(f"cochlear_transform must be one of {COCHLEAR_TRANSFORMS}.")

    # ------------------------------------------------------------- responses
    def _load_responses(self, root, cache, areas, dt_int):
        self.responses = [[] for _ in self.stim_meta]
        self.nrn_meta = []
        pt70 = [s for s, m in enumerate(self.stim_meta)
                if m["category"] == "pure_tones" and m["level_db"] == 70.0]
        for area in areas:
            arr = gp.load_ac(cache) if area == "AC" else gp.load_ephys(cache, area, dt_int)
            self._append_area(arr, area, gp.load_sessions(root, area), pt70)
        self.N_neurons = len(self.nrn_meta)

    def _bin_masks(self, T):
        starts = np.arange(T) * self.dt
        pre = starts + self.dt <= gs.ONSET_MS
        evoked = (starts >= gs.ONSET_MS) & (starts < gs.ONSET_MS + 500.0)
        return pre, evoked

    def _append_area(self, arr: np.ndarray, area: str, sessions: np.ndarray, pt70: List[int]):
        N, S, R, T = arr.shape
        t_all = torch.from_numpy(arr)
        if area == "AC":
            # Missing repeats are whole-trial NaN. Nearly every pair has its valid
            # repeats first, so most responses stay views into ``arr``.
            valid = ~np.isnan(arr).any(axis=3)                           # (N, S, R)
            n_valid = valid.sum(axis=2)
            prefix = valid.cumsum(axis=2)[np.arange(N)[:, None], np.arange(S)[None, :],
                                          np.maximum(n_valid - 1, 0)] == n_valid
        pre, evoked = self._bin_masks(T)
        if area == "AC":
            with warnings.catch_warnings():                              # pairs with no repeat
                warnings.simplefilter("ignore", RuntimeWarning)
                psth = np.nanmean(arr, axis=2)                           # (N, S, T)
        else:
            psth = arr.mean(axis=2)          # spikes have no NaN; nanmean would copy arr
        base = psth[:, :, pre].mean(axis=2)                              # (N, S)
        ev = psth[:, :, evoked].mean(axis=2) - base
        scale = 1.0 if area == "AC" else 1000.0 / self.dt               # counts/bin -> sp/s
        pt70_khz = [self.stim_meta[s]["frequency_khz"][0] for s in pt70]
        for n in range(N):
            for s in range(S):
                if area != "AC":
                    r = t_all[n, s]
                elif n_valid[n, s] == 0:
                    r = torch.full((1, 1), float("nan"))
                elif prefix[n, s]:
                    r = t_all[n, s, :n_valid[n, s]]
                else:
                    r = t_all[n, s][torch.from_numpy(valid[n, s])]
                self.responses[s].append(r)
            bf = _best_freq(ev[n, pt70], pt70_khz) if pt70 else None
            self.nrn_meta.append({
                "cell_id": f"{area}_{n:04d}",
                "area": area,
                "session": int(sessions[n]),
                "session_id": f"{area}-s{int(sessions[n])}",
                "index_in_release": n,
                "best_freq_khz": bf,
                # AC traces are baseline-subtracted per trial: no spontaneous rate
                "spont_rate": None if area == "AC" else float(base[n].mean() * scale),
            })

    # ---------------------------------------------------------------- stimuli
    def _build_stims(self, root, cache, wdir, T, dt, fmin, fmax, bpo, window_ms,
                     cochlear_channels, cochlear_transform):
        hop = int(round(dt * gs.NATIVE_FS / 1000.0))
        self.stims = []
        if self.spec_backend == "cochlear_model":
            self._build_cochlear(root, cache, T, dt, cochlear_channels, cochlear_transform)
            return
        if self.return_waveform:
            self.audio_fs = gs.NATIVE_FS
            self.F = n_bands_for(fmin, fmax, bpo)       # target width for a wav2spec front end
            for m in self.stim_meta:
                buf = gs.trial_waveform(gs.load_wav_int16(wdir, m["name"]), T, hop) / 32768.0
                self.stims.append(torch.from_numpy(buf.astype(np.float32))[None])
            return
        for m in self.stim_meta:
            spec = gs.logspec_trial(gs.load_wav_int16(wdir, m["name"]), T, dt, fmin, fmax, bpo,
                                    window_ms)
            self.stims.append(torch.from_numpy(spec)[None])
        self.F = int(self.stims[0].shape[1])

    def _build_cochlear(self, root, cache, T, dt, channels, transform):
        ihc, groups = gp.load_cochlear_ihc(cache)                       # (G, 900, S), (fibres,)
        if channels == "fibres":
            order = np.array([m["index_cc"] for m in self.stim_meta])
            cc = np.load(root / "Dataset_CC.npy", mmap_mode="r")

            def per_stim(s):
                return np.asarray(cc[:, :, order[s]])                   # (fibres, 900)

            self.cochlear_ihc_of_fibre = groups
            self.cochlear_cf_khz = gc.approx_cf_khz(int(groups.max()) + 1)[groups]
        else:
            def per_stim(s):
                return ihc[:, :, s]

            self.cochlear_cf_khz = gc.approx_cf_khz(ihc.shape[0])
        self.cochlear_transform = transform
        for s in range(len(self.stim_meta)):
            x = gc.rebin_cochlear(per_stim(s), dt, T)
            if transform == "log1p":
                x = np.log1p(x)
            self.stims.append(torch.from_numpy(x.astype(np.float32))[None])
        self.F = int(self.stims[0].shape[1])

    # ----------------------------------------------------------------- split
    def group_split(self, seed: Optional[int] = None,
                    fractions: Sequence[float] = (0.7, 0.1, 0.2)) -> Dict[str, List[int]]:
        """Split the 307 sounds into train / val / test without leakage.

        Sounds that are the same waveform up to a gain (50 / 70 dB copies)
        stay together, and so does each decomposed clip's reconstruction with
        its components (the reconstruction is assembled from their audio).
        Everything else is a separate sound: tones, sweeps and ramps at other
        frequencies or directions, time-reversed clips (a reversed clip can be
        tested while its forward version is trained on), and an original
        clip vs its reconstruction (similar, but no shared audio). Units are
        drawn category by category (the decompositions form one stratum),
        each to the set furthest below its target share. The draw covers all
        307 sounds whatever ``areas`` is loaded, so a seed gives the same
        split for CN, IC and AC.

        Coarse at the edges: the decomposition stratum has only 4 units, one
        of 25 sounds (the dolphin call), which always goes to train; the
        other three decomposed clips reach val or test in 38-48% of seeds
        each (all in train at ``seed=0``). The 5 harmonic chords never get a
        validation sound.

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
        units, strata = gsplit.split_units(self.stim_meta, gs.load_gain_copies())
        return gsplit.draw_split(units, strata, seed, fractions)
