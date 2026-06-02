"""Alice EEG dataset adapted to the deepSTRF paradigm.

See ``docs/_source/md/README_Alice_EEG.md`` for context, citation, and the
benchmark comparison against Brodbeck et al. 2023 (eLife).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.utils.audio_io import load_wav
from deepSTRF.utils.data_download import default_cache_dir, stream_download, unzip


# UMd Digital Repository (DRUM) bitstream UUIDs for Brodbeck's restructured
# Alice EEG release (DOI 10.13016/pulf-lndn). Verified 2026-05-15 against
# https://drum.lib.umd.edu/handle/1903/27591. Anonymous downloads work over
# HTTPS — no credentials required. Total payload ~2.5 GiB.
_DRUM_BITSTREAMS = {
    "eeg.0.zip":   "264ca110-f7f4-4fd5-9b00-896102b841ad",
    "eeg.1.zip":   "bef532d8-cf74-4b9d-9b4c-5c1f81610ce9",
    "eeg.2.zip":   "25fc51ae-d1fa-4094-85af-65dd4cf30251",
    "stimuli.zip": "df241468-26ee-42df-b27f-3f438cfc5a3f",
}


def download_alice_eeg(dest: Optional[str] = None) -> str:
    """Download Brodbeck's restructured Alice EEG release from UMd DRUM.

    Idempotent: skips any zip that's already on disk and any subdirectory
    that's already unpacked. Returns the dataset directory.

    Parameters
    ----------
    dest : str, optional
        Defaults to the platformdirs cache (overridable via
        ``$DEEPSTRF_DATA_DIR``).

    Notes
    -----
    ~2.5 GiB total across four zips. Anonymous HTTPS; no auth.
    """
    dest_path = str(default_cache_dir("Alice_EEG") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)

    for filename, uuid in _DRUM_BITSTREAMS.items():
        zip_path = os.path.join(dest_path, filename)
        if not os.path.exists(zip_path):
            url = f"https://drum.lib.umd.edu/bitstreams/{uuid}/download"
            stream_download(url, zip_path)

        if filename.startswith("eeg."):
            target = os.path.join(dest_path, filename[:-len(".zip")])
            sentinel = glob.glob(os.path.join(target, "eeg", "S*",
                                              "S*_alice-raw.fif"))
        else:
            target = dest_path
            sentinel = glob.glob(os.path.join(dest_path, "stimuli", "*.wav"))
        if not sentinel:
            os.makedirs(target, exist_ok=True)
            unzip(zip_path, target)

    return dest_path


# -----------------------------------------------------------------------------
# Module-level helpers (pure; tested independently from the class).
# -----------------------------------------------------------------------------


def _discover_subjects(data_root: str) -> "dict[str, str]":
    """Map subject id -> absolute path of its ``Sxx_alice-raw.fif`` file.

    Brodbeck's restructure shards the per-subject EEG files across three
    sub-folders (``eeg.0/``, ``eeg.1/``, ``eeg.2/``) for upload-size reasons.
    Each shard contains an ``eeg/`` directory with one ``Sxx/`` per subject.
    """
    found: "dict[str, str]" = {}
    for shard in ("eeg.0", "eeg.1", "eeg.2"):
        shard_root = os.path.join(data_root, shard, "eeg")
        if not os.path.isdir(shard_root):
            continue
        for sub in sorted(os.listdir(shard_root)):
            fif = os.path.join(shard_root, sub, f"{sub}_alice-raw.fif")
            if os.path.exists(fif):
                found[sub] = fif
    return found


def _erb_filterbank(n_bands: int, sr: int, n_fft: int,
                    f_min: float = 80.0, f_max: Optional[float] = None) -> torch.Tensor:
    """Gaussian filterbank with ERB-spaced centers — gammatone approximation.

    Returns a ``(n_bands, n_fft // 2 + 1)`` matrix. Brodbeck 2023 uses
    Heeris's time-domain gammatone filterbank; we approximate it in the
    frequency domain with Gaussians centered on Glasberg & Moore (1990)
    ERB-scale frequencies. Spectrally equivalent to first order; matches
    the band centers and bandwidths used in the eelbrain figure 4 panels.
    """
    if f_max is None:
        f_max = sr / 2

    def hz_to_erb(hz: np.ndarray) -> np.ndarray:
        return 21.4 * np.log10(0.00437 * hz + 1.0)

    def erb_to_hz(erb: np.ndarray) -> np.ndarray:
        return (np.power(10.0, erb / 21.4) - 1.0) / 0.00437

    erb_centers = np.linspace(hz_to_erb(np.array(f_min)),
                              hz_to_erb(np.array(f_max)), n_bands)
    hz_centers = erb_to_hz(erb_centers)
    bandwidths = 24.7 * (0.00437 * hz_centers + 1.0)

    freqs = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_bands, freqs.size), dtype=np.float32)
    for b, (fc, bw) in enumerate(zip(hz_centers, bandwidths)):
        fb[b] = np.exp(-((freqs - fc) / bw) ** 2)
    return torch.from_numpy(fb)


def _heeris_gammatone(wav: torch.Tensor, sr: int, n_bands: int, dt_ms: float,
                      window_ms: float = 25.0,
                      f_min: float = 80.0,
                      f_max: Optional[float] = None) -> torch.Tensor:
    """Paper-faithful time-domain gammatone spectrogram (Heeris 2018).

    Calls :func:`gammatone.gtgram.gtgram` — the Heeris filterbank cited
    by Brodbeck et al. 2023 via eelbrain's ``gammatone_bank`` wrapper.
    Output shape ``(1, n_bands, T)``, low-frequency band at index 0
    (matches the Gaussian-approximation helper's convention; the raw
    ``gtgram`` returns high → low).

    The ``gammatone`` PyPI package is optional — it's listed under the
    ``[eeg]`` and ``[le]`` extras. Install via ``pip install
    deepSTRF[eeg]`` or ``pip install gammatone``.
    """
    try:
        import gammatone.gtgram as _gt
    except ImportError as exc:
        raise ImportError(
            "The Heeris gammatone backend requires the optional `gammatone` "
            "package. Install with `pip install gammatone` or "
            "`pip install deepSTRF[eeg]`."
        ) from exc

    wav_np = wav.squeeze(0).cpu().numpy().astype("float64")
    # gtgram's signature: gtgram(wave, fs, window_time, hop_time, channels, f_min, f_max=None)
    # NOTE: when f_max is None, gtgram defaults to fs/2 internally — we
    # pass it through explicitly so the audit-status of the band edges
    # is transparent in the returned tensor.
    #
    # Row convention: ``gtgram`` returns rows in INCREASING center-frequency
    # order (row 0 = f_min, row -1 = f_max). Counter-intuitively,
    # ``gammatone.filters.centre_freqs(...)`` returns the *same* centers in
    # *decreasing* order — different convention between the two
    # gammatone helpers. We pass gtgram's output straight through, matching
    # the Gaussian backend's "ERB band 0 = lowest frequency" convention.
    out = _gt.gtgram(
        wave=wav_np,
        fs=int(sr),
        window_time=float(window_ms) * 1e-3,
        hop_time=float(dt_ms) * 1e-3,
        channels=int(n_bands),
        f_min=float(f_min),
        f_max=(float(f_max) if f_max is not None else None),
    )                                                                  # (n_bands, T), row 0 = low
    out = np.log(out + 1e-8)
    return torch.from_numpy(out).to(dtype=torch.float32).unsqueeze(0)  # (1, n_bands, T)


def _gammatone_spectrogram(wav: torch.Tensor, sr: int,
                           n_bands: int, dt_ms: float,
                           n_fft: Optional[int] = None,
                           window_ms: Optional[float] = None,
                           f_min: float = 80.0,
                           f_max: Optional[float] = None,
                           backend: str = "gaussian") -> torch.Tensor:
    """Log-power ERB-band spectrogram.

    Two backends:

    - ``backend='gaussian'`` (default, back-compat): frequency-domain
      Gaussian filterbank with ERB-spaced centers, applied to a power
      STFT. Cheap; what deepSTRF has always done. The empirical audit
      at ``untracked/alice_eeg_spec_compare.py`` shows it differs
      visibly from the paper-faithful Heeris bank — lower dynamic
      range, less time-localized transients.
    - ``backend='heeris'`` (paper-faithful): time-domain gammatone
      filterbank from the ``gammatone`` PyPI package (Heeris 2018),
      same as Brodbeck et al. 2023 via eelbrain's ``gammatone_bank``.
      ``n_fft`` is ignored — Heeris owns the analysis-window logic
      via ``window_ms`` (default 25 ms).

    Parameters
    ----------
    wav : Tensor, shape ``(1, T_samples)``
        Mono waveform, float.
    sr : int
        Sample rate of ``wav``.
    n_bands : int
        Number of ERB bands.
    dt_ms : float
        Output time-bin width in ms; sets the STFT hop length.
    n_fft : int, optional
        Explicit FFT length (``backend='gaussian'`` only). Overrides
        ``window_ms`` if both are set. ``None`` falls back to
        ``window_ms`` (or the legacy default of 1024 if ``window_ms``
        is also ``None``).
    window_ms : float, optional
        FFT analysis-window length in ms. For ``backend='gaussian'``:
        ``n_fft = round(window_ms * 1e-3 * sr)``, floored at ``hop``.
        For ``backend='heeris'``: passed straight into ``gtgram`` as
        ``window_time`` (default 25 ms — Heeris's own convention).
    f_min, f_max : float, optional
        ERB-band edges in Hz. Default ``80.0`` and ``sr/2`` — matches
        Brodbeck 2023 Fig 4's lower edge but lets the upper edge run
        well past the speech-relevant range. For human-speech work,
        pass ``f_max=8000`` to drop the inaudible-for-speech bands.
    backend : {'gaussian', 'heeris'}, default 'gaussian'
        Spec-pipeline backend. See above. ``'heeris'`` requires the
        optional ``gammatone`` PyPI package (in the ``[eeg]`` extra).

    Returns
    -------
    Tensor, shape ``(1, n_bands, T)``
        Log power per band per frame, low-frequency band at index 0.
    """
    if backend not in ("gaussian", "heeris"):
        raise ValueError(
            f"backend must be 'gaussian' or 'heeris', got {backend!r}"
        )
    if backend == "heeris":
        return _heeris_gammatone(
            wav, sr=sr, n_bands=n_bands, dt_ms=dt_ms,
            window_ms=(window_ms if window_ms is not None else 25.0),
            f_min=f_min, f_max=f_max,
        )

    # --- backend == 'gaussian' (legacy default) -------------------------
    hop = max(1, int(round(sr * dt_ms / 1000.0)))
    if n_fft is None:
        if window_ms is not None:
            n_fft = max(int(round(window_ms * 1e-3 * sr)), hop)
        else:
            n_fft = 1024     # legacy default; preserves bit-identical specs
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop, power=2.0,
    )(wav)  # (1, n_fft//2+1, T)
    fb = _erb_filterbank(n_bands, sr, n_fft, f_min=f_min, f_max=f_max)
    out = torch.einsum("bf,cft->cbt", fb, spec)           # (1, n_bands, T)
    return torch.log(out + 1e-8)


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------


class AliceEEGDataset(AudioNeuralDataset):
    """PyTorch dataset for EEG from the Alice audiobook listening paradigm.

    33 human participants listened to the first chapter of *Alice in
    Wonderland* (~12.4 min) split into 12 audio segments, recorded with 61
    EEG channels per subject (10-20-like montage). Bad channels and bad
    artifact windows (marked in the source ``.fif`` metadata) are converted
    to NaN at the response level. Each subject heard each segment once
    (``R = 1``). deepSTRF consumes Brodbeck et al. 2023's restructured
    release (UMd PULFR ``10.13016/pulf-lndn``): per-subject MNE ``.fif``
    files plus 12 audio segments and a word-onset table. See
    ``docs/_source/md/README_Alice_EEG.md`` for the full dataset notes.

    The ``treat_subjects_as`` argument selects one of two layouts:

    - ``"neurons"`` (default): every ``(subject, channel)`` pair becomes a
      "neuron"; ``N = sum_s(n_channels_s)`` and ``R = 1`` everywhere. Bad
      channels carry the structural NaN sentinel. Use ``corrcoef`` / ``fve``.
    - ``"repeats"``: subjects are treated as repeats of a shared canonical
      per-channel EEG response; ``N = n_montage_channels`` (e.g. 61) and
      ``R = n_subjects``. Bad ``(channel, subject)`` combinations become NaN
      repeat slabs. Useful for inter-subject reliability (ISC-style) via
      ``normalized_corrcoef(method='schoppe')`` — but note this is
      *inter-subject* reliability, not trial reliability, so the iid-trial
      noise model the Schoppe correction assumes does not strictly hold;
      treat the resulting ceiling as a group-level sanity check.

    Notes
    -----
    Follows the standard deepSTRF data paradigm (see
    ``docs/_source/md/data_paradigm.md``). Alice-specific metadata:

    - ``stims`` are ``S = 12`` log-power ERB-band spectrograms ``(1, F, T_s)``
      (a gammatone approximation; see ``_gammatone_spectrogram``).
    - ``stim_meta`` dicts hold ``name``, ``type``, ``sample_rate``,
      ``n_samples`` and ``duration_s``.
    - ``nrn_meta`` dicts hold ``channel_id``, ``subject``, ``area`` and
      ``xyz`` in ``"neurons"`` mode; a channel-only entry in ``"repeats"``
      mode.

    The default ``spec_backend='gaussian'`` is a frequency-domain Gaussian
    approximation of Brodbeck 2023's time-domain gammatone (Heeris)
    filterbank — spectrally equivalent to first order but with lower dynamic
    range and less time-localized transients. ``spec_backend='heeris'``
    selects the paper-faithful bank (requires the optional ``gammatone``
    package in the ``[eeg]`` extra). The ``window_ms`` / ``fmin`` / ``fmax``
    constructor knobs control the FFT window and ERB-band edges; their
    defaults preserve the historical behaviour, so no existing fits change.

    References
    ----------
    Bhattasali et al. (2020). "The Alice Datasets: fMRI & EEG Observations of
    Natural Language Comprehension." LREC.

    Brennan et al. (2019). "Hierarchical structure guides rapid linguistic
    predictions during naturalistic listening." *PLOS ONE*.

    Brodbeck et al. (2023). Eelbrain methods paper. *eLife* (Tools &
    Resources).
    """

    def __init__(self, path: Optional[str] = None,
                 subjects: Optional[Sequence[str]] = None,
                 dt_ms: float = 10.0,
                 n_frequency_bands: int = 8,
                 treat_subjects_as: str = "neurons",
                 hp_freq_hz: Optional[float] = 1.0,
                 lp_freq_hz: Optional[float] = None,
                 window_ms: Optional[float] = None,
                 fmin: float = 80.0,
                 fmax: Optional[float] = None,
                 spec_backend: str = "gaussian",
                 download: bool = False,
                 return_waveform: bool = False,
                 audio_fs: int = 44100):
        """
        Parameters
        ----------
        path : str, optional
            Path to the Brodbeck-restructured Alice EEG data directory
            (containing ``eeg.0/``, ``eeg.1/``, ``eeg.2/``, ``stimuli/``).
            Defaults to the platformdirs cache
            (``$DEEPSTRF_DATA_DIR`` overrides).
        subjects : sequence of str, optional
            Subject ids (e.g. ``["S01", "S20"]``). If ``None``, all subjects
            discovered on disk are used.
        dt_ms : float, default 10.0
            Time-bin width in ms (100 Hz with the default — matches the
            eelbrain paper's analysis rate).
        n_frequency_bands : int, default 8
            ERB-band count for the gammatone-equivalent spectrogram.
            Matches Brodbeck 2023 Fig 4.
        treat_subjects_as : {"neurons", "repeats"}, default "neurons"
            See the class docstring.
        hp_freq_hz : float or None, default 1.0
            High-pass cutoff applied via ``raw.filter`` before downsampling
            and segmentation. The Brodbeck restructure ships data with a
            0.1 Hz HP, which leaves enough slow drift across the ~12 min
            recording that per-segment baselines vary by >1 SD — fatal for
            held-out fve. Brodbeck applies 1 Hz HP in the paper's analysis
            pipeline; we mirror that as the default. Pass ``None`` to skip.
        lp_freq_hz : float or None, default None
            Optional low-pass cutoff. Useful if you want to focus on the
            cortical-tracking band (< 40 Hz) or the envelope-tracking band
            (< 8 Hz).
        window_ms : float, optional
            FFT analysis-window length in ms for the stimulus
            spectrogram. ``None`` preserves the legacy ``n_fft=1024``
            default — at the audiobook sample rate (16 kHz) this gives
            a ~64 ms window; at 44.1 kHz, ~23 ms. Pass an explicit
            ``window_ms`` to override (e.g. ``25.0`` for the Kaldi
            convention). The spec pipeline is otherwise unchanged from
            the audit baseline — see the "Audit status" callout below
            before benchmarking against Brodbeck 2023.
        fmin, fmax : float, optional
            Lower and upper ERB-band edges in Hz. Default ``80.0`` and
            ``sr/2`` (Nyquist). For speech-tracking work, pass
            ``fmax=8000`` to drop bands above the speech-relevant range
            (matches Brodbeck 2023's published lower-band figure
            roughly; not empirically validated against the paper's
            actual filterbank — see "Audit status").
        spec_backend : {'gaussian', 'heeris'}, default 'gaussian'
            Spec-pipeline backend.

            - ``'gaussian'`` (back-compat): frequency-domain Gaussian
              ERB filterbank — the deepSTRF approximation that's been
              shipped to date.
            - ``'heeris'`` (paper-faithful, requires ``gammatone``
              PyPI package): time-domain Heeris filterbank, same as
              Brodbeck et al. 2023 via eelbrain's
              ``gammatone_bank``. See the empirical comparison at
              ``untracked/alice_eeg_spec_compare.py`` — Heeris has
              visibly sharper time-localization and broader dynamic
              range. Recommended when reproducing the paper.
        download : bool, default False
            If True and the data is missing under ``path``, fetch the four
            zips from the UMd DRUM mirror (~2.5 GiB total; anonymous HTTPS).
            Idempotent — skips any zip / unpacked subtree already present.
        """
        try:
            import mne  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AliceEEGDataset requires the optional `mne` dependency. "
                "Install with `pip install mne` or `pip install deepSTRF[eeg]`."
            ) from exc

        if path is None:
            path = str(default_cache_dir("Alice_EEG"))
        if download:
            download_alice_eeg(path)

        super().__init__(path, dt_ms)

        self.species = "human"
        self.behavioral_state = "passive-listening"
        self.F = int(n_frequency_bands)
        self.hp_freq_hz = hp_freq_hz
        self.lp_freq_hz = lp_freq_hz
        self.window_ms = float(window_ms) if window_ms is not None else None
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else None
        if spec_backend not in ("gaussian", "heeris"):
            raise ValueError(
                f"spec_backend must be 'gaussian' or 'heeris', got "
                f"{spec_backend!r}"
            )
        self.spec_backend = spec_backend
        self.hearing_range_hz = (20.0, 20000.0)   # human (informational)

        # Raw-waveform input mode (opt-in). The native stim is the in-loader
        # ERB/gammatone spectrogram; here we instead hand out the 44.1 kHz source
        # audiobook waveform (continuous audio, no silence flanks → aligns from
        # t=0) and let a model's wav2spec slot build the spectrogram. For the
        # paper-faithful Heeris backend, ``Gammatonegram`` reproduces it exactly.
        self.return_waveform = bool(return_waveform)
        self.audio_fs = int(audio_fs) if return_waveform else None

        if treat_subjects_as not in ("neurons", "repeats"):
            raise ValueError(
                f"treat_subjects_as must be 'neurons' or 'repeats', "
                f"got {treat_subjects_as!r}"
            )
        self.treat_subjects_as = treat_subjects_as

        # 1. discover subjects on disk and apply user filter ------------------
        available = _discover_subjects(path)
        if subjects is None:
            self.subjects = sorted(available)
        else:
            missing = [s for s in subjects if s not in available]
            if missing:
                raise FileNotFoundError(
                    f"Requested subjects not found under {path}: {missing}. "
                    f"Available: {sorted(available)}"
                )
            self.subjects = list(subjects)
        if not self.subjects:
            raise FileNotFoundError(
                f"No Alice EEG subjects found under {path}. Expected "
                f"`eeg.{{0,1,2}}/eeg/Sxx/Sxx_alice-raw.fif` layout."
            )

        # 2. load stimuli (12 .wav segments) → ERB-band log spectrograms -----
        self.stims, self.stim_meta, T_per_stim = self._load_stimuli(path)
        S = len(self.stims)
        # T_per_stim is the neural (spec) frame count per stim — NOT the waveform
        # length, which in return_waveform mode would mis-bin the EEG responses.
        wav_durations_s = [m["duration_s"] for m in self.stim_meta]

        # 3. load per-subject EEG, segment by stim onsets, downsample ---------
        per_subject = self._load_all_subjects(
            available, wav_durations_s, T_per_stim
        )

        # 4. assemble responses + nrn_meta in the chosen mode ----------
        if treat_subjects_as == "neurons":
            self.nrn_meta, self.responses = \
                self._assemble_neurons_mode(per_subject, S, T_per_stim)
        else:
            self.nrn_meta, self.responses = \
                self._assemble_repeats_mode(per_subject, S, T_per_stim)

        self.N_neurons = len(self.nrn_meta)

        self.validate()

    # ------------------------------------------------------------------
    # Stimulus loading
    # ------------------------------------------------------------------

    def _load_stimuli(self, path: str):
        stim_dir = os.path.join(path, "stimuli")
        wavs = sorted(
            glob.glob(os.path.join(stim_dir, "*.wav")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        if len(wavs) != 12:
            raise FileNotFoundError(
                f"Expected 12 audio segments in {stim_dir}, found {len(wavs)}. "
                f"Check that `stimuli.zip` has been unpacked."
            )

        stims, stim_meta, t_neural = [], [], []
        for wav_path in wavs:
            wav, sr = load_wav(wav_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            n_samples = int(wav.shape[-1])
            spec = _gammatone_spectrogram(
                wav, sr, self.F, self.dt,
                window_ms=self.window_ms, f_min=self.fmin, f_max=self.fmax,
                backend=self.spec_backend,
            )  # (1, F, T)
            t_neural.append(int(spec.shape[-1]))   # neural frame count (drives EEG binning)
            if self.return_waveform:
                # store the source waveform (resampled to audio_fs if needed),
                # grid-locked to T_neural*hop so it aligns with the spec frames
                # (= EEG response bins). Continuous audio → no offset.
                w = (wav if sr == self.audio_fs
                     else torchaudio.functional.resample(wav, sr, self.audio_fs))
                T_audio = spec.shape[-1] * self.hop
                if w.shape[-1] < T_audio:
                    w = torch.nn.functional.pad(w, (0, T_audio - w.shape[-1]))
                else:
                    w = w[..., :T_audio]
                stims.append(w.contiguous().float())
            else:
                stims.append(spec)
            stim_meta.append({
                "name": os.path.basename(wav_path),
                "type": "alice_chapter1",
                "sample_rate": float(sr),
                "n_samples": n_samples,
                "duration_s": n_samples / float(sr),
            })
        return stims, stim_meta, t_neural

    # ------------------------------------------------------------------
    # EEG loading
    # ------------------------------------------------------------------

    def _load_all_subjects(self, available_paths, wav_durations_s, T_per_stim):
        """Return a dict ``sub -> {'ch_names', 'xyz', 'segments'}``.

        ``segments`` is a list of length S; each element is a NumPy array of
        shape ``(n_channels_sub, T_per_stim[s])``. NaN where the channel is
        bad or the artifact-annotated window falls inside the segment.
        """
        import mne  # local import, already guarded in __init__
        mne.set_log_level("WARNING")

        out = {}
        target_fs = 1000.0 / self.dt
        for sub in self.subjects:
            raw = mne.io.read_raw_fif(available_paths[sub], preload=True,
                                      verbose=False)
            raw.pick(picks="eeg")  # drop stim/EOG/Aux channels

            # find segment onsets from the 12 numeric annotations '1'..'12'
            seg_onsets_s = self._extract_segment_onsets(raw, expected=12)

            # band-pass before downsampling — suppress slow DC drift that
            # makes per-segment baselines diverge across the recording.
            if self.hp_freq_hz is not None or self.lp_freq_hz is not None:
                raw.filter(l_freq=self.hp_freq_hz, h_freq=self.lp_freq_hz,
                           verbose=False)

            # downsample EEG (anti-aliasing handled by MNE) and rescale onsets
            raw.resample(target_fs, verbose=False)

            # mask bad channels & artifact windows with NaN
            data = raw.get_data()  # (n_chans, T) at target_fs
            bad_idx = [raw.ch_names.index(ch)
                       for ch in raw.info["bads"]
                       if ch in raw.ch_names]
            if bad_idx:
                data[bad_idx, :] = np.nan
            self._apply_bad_window_mask(data, raw, target_fs)

            # slice EEG segments aligned to the 12 audio segments
            segments = []
            for s, (onset_s, dur_s, T_s) in enumerate(
                    zip(seg_onsets_s, wav_durations_s, T_per_stim)):
                start = int(round(onset_s * target_fs))
                stop = start + T_s
                if stop > data.shape[1]:
                    # pad with NaN if the recording ended before the segment
                    pad = stop - data.shape[1]
                    chunk = np.concatenate(
                        [data[:, start:], np.full((data.shape[0], pad), np.nan)],
                        axis=1,
                    )
                else:
                    chunk = data[:, start:stop]
                segments.append(chunk.astype(np.float32))

            # montage xyz (subject-specific space; standardized to head coords)
            montage = raw.get_montage()
            xyz = {}
            if montage is not None:
                pos = montage.get_positions().get("ch_pos") or {}
                xyz = {ch: pos[ch] for ch in raw.ch_names if ch in pos}

            out[sub] = {
                "ch_names": list(raw.ch_names),
                "xyz": xyz,
                "segments": segments,
            }
        return out

    @staticmethod
    def _extract_segment_onsets(raw, expected: int = 12):
        """Return numeric-annotation onsets in seconds, ordered 1..expected.

        Brodbeck's restructure tags each audio segment onset with an
        annotation whose description is the segment index as a string
        ('1'..'12'). We sort by integer value so segments are returned in
        playback order regardless of file storage order.
        """
        anns = raw.annotations
        keep = [(int(d), float(o))
                for o, d in zip(anns.onset, anns.description)
                if str(d).isdigit() and 1 <= int(d) <= expected]
        keep.sort(key=lambda x: x[0])
        if len(keep) != expected:
            raise RuntimeError(
                f"Expected {expected} numeric segment annotations, "
                f"got {len(keep)} in {raw.filenames[0]!r}"
            )
        return [onset for _, onset in keep]

    @staticmethod
    def _apply_bad_window_mask(data: np.ndarray, raw, target_fs: float) -> None:
        """In-place NaN-mask any annotation whose description starts with 'BAD'."""
        for onset, dur, desc in zip(raw.annotations.onset,
                                    raw.annotations.duration,
                                    raw.annotations.description):
            if not str(desc).upper().startswith("BAD"):
                continue
            i0 = int(round(onset * target_fs))
            i1 = int(round((onset + max(dur, 0.0)) * target_fs))
            data[:, max(0, i0):max(0, i1)] = np.nan

    # ------------------------------------------------------------------
    # Assembly into the deepSTRF (S, list[N]) layout
    # ------------------------------------------------------------------

    def _assemble_neurons_mode(self, per_subject, S, T_per_stim):
        """Each (subject, channel) pair is one 'neuron'. R = 1 everywhere."""
        nrn_meta = []
        for sub in self.subjects:
            ch_names = per_subject[sub]["ch_names"]
            xyz_map = per_subject[sub]["xyz"]
            for ch in ch_names:
                nrn_meta.append({
                    "channel_id": ch,
                    "subject": sub,
                    "area": "EEG",
                    "xyz": xyz_map.get(ch, None),
                })

        # column offsets for vertical stacking of subjects' channels
        offsets, total = {}, 0
        for sub in self.subjects:
            offsets[sub] = total
            total += len(per_subject[sub]["ch_names"])

        responses: List[List[torch.Tensor]] = []
        for s in range(S):
            T_s = T_per_stim[s]
            pop_resps = [None] * total
            for sub in self.subjects:
                seg = per_subject[sub]["segments"][s]  # (n_chans_sub, T_s)
                start = offsets[sub]
                for ch_local, trace in enumerate(seg):
                    trace_t = torch.from_numpy(trace).unsqueeze(0)  # (1, T_s)
                    if torch.isnan(trace_t).all():
                        # structural-missingness sentinel per data paradigm §3.1
                        pop_resps[start + ch_local] = torch.full(
                            (1, 1), float("nan"))
                    else:
                        pop_resps[start + ch_local] = trace_t
            responses.append(pop_resps)
        return nrn_meta, responses

    def _assemble_repeats_mode(self, per_subject, S, T_per_stim):
        """Channels-as-neurons; subjects-as-repeats. N = montage channels,
        R = n_subjects. Bad/(channel, subject) cells become NaN slabs.

        Uses the union of channel names across selected subjects as the
        canonical N. Per-channel R-slots line up positionally with the
        order in ``self.subjects``.
        """
        # canonical channel list = union of channels across subjects
        canonical: List[str] = []
        seen = set()
        for sub in self.subjects:
            for ch in per_subject[sub]["ch_names"]:
                if ch not in seen:
                    seen.add(ch)
                    canonical.append(ch)
        N = len(canonical)
        R = len(self.subjects)

        # nrn_meta: one entry per canonical channel. xyz from the
        # first subject who has it (montages are subject-aligned in Brodbeck's
        # restructure, so this is well-defined).
        xyz_first = {}
        for sub in self.subjects:
            for ch, pos in per_subject[sub]["xyz"].items():
                xyz_first.setdefault(ch, pos)
        nrn_meta = [{
            "channel_id": ch,
            "subject": None,             # repeats mode: not per-neuron
            "area": "EEG",
            "xyz": xyz_first.get(ch),
        } for ch in canonical]

        # responses: list[S] of list[N] tensors (R, T_s)
        responses: List[List[torch.Tensor]] = []
        for s in range(S):
            T_s = T_per_stim[s]
            pop_resps: List[torch.Tensor] = []
            for n, ch in enumerate(canonical):
                slab = np.full((R, T_s), np.nan, dtype=np.float32)
                for r, sub in enumerate(self.subjects):
                    ch_names = per_subject[sub]["ch_names"]
                    if ch in ch_names:
                        idx = ch_names.index(ch)
                        slab[r, :] = per_subject[sub]["segments"][s][idx, :]
                if np.isnan(slab).all():
                    pop_resps.append(torch.full((1, 1), float("nan")))
                else:
                    pop_resps.append(torch.from_numpy(slab))
            responses.append(pop_resps)
        return nrn_meta, responses


if __name__ == "__main__":
    import sys
    DATA = os.path.join(
        os.path.dirname(__file__), "Alice_EEG", "data",
        "brodbeck_eelbrain_elife",
    )
    if not os.path.isdir(DATA):
        sys.exit(f"local data dir missing: {DATA}")
    ds = AliceEEGDataset(path=DATA, subjects=["S01"], dt_ms=10.0,
                           n_frequency_bands=8, treat_subjects_as="neurons")
    print(f"N_neurons: {ds.N_neurons}")
    print(f"S: {len(ds.stims)}, F: {ds.F}, dt: {ds.dt} ms")
    print(f"stim 0 shape: {ds.stims[0].shape}")
    print(f"response[0][0] shape: {ds.responses[0][0].shape}")
    print(f"sample nrn_meta[0]: {ds.nrn_meta[0]}")
    print(f"sample stim_meta[0]: {ds.stim_meta[0]}")
