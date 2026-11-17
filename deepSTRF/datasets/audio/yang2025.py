"""Yang & Long (2025) budgerigar AAC — a **decoding** dataset (neural activity → own voice).

Single units in the left anterior arcopallium (AAC) of 4 adult male budgerigars,
recorded with a 128-channel silicon probe while the freely moving birds produced
warble song and calls. The bird's own vocalisations come from a piezo skull
microphone sampled on the same 30 kHz clock as the spikes. AAC is a vocal-motor
area — activity leads the sound — so the natural model direction is decoding:
predict the spectrogram from the population activity (see
:class:`~deepSTRF.models.audio.decoding.AudioDecodingModel`).

Reference: Yang, Z. & Long, M. A. (2025). Convergent vocal representations in
parrot and human forebrain motor networks. Nature 640, 427-434.
doi:10.1038/s41586-025-08695-8. Data: Zenodo 10.5281/zenodo.14057061 (CC BY 4.0).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from deepSTRF.datasets.audio._yang2025_prepare import (
    BIRDS, CACHE_VERSION, download_yang2025, prepare_yang2025,
)
from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.utils.data_download import default_cache_dir

LOG_EPS = 1e-10


class Yang2025Dataset(AudioNeuralDataset):
    """Budgerigar AAC single units + the bird's own vocalisations (Yang & Long 2025).

    One item per vocal *episode* — a warble bout or a call — cut into
    pieces of at most ``max_episode_s`` (``None`` keeps episodes whole). Each
    item starts ``pre_roll_ms`` before its vocal onset so that a causal decoder
    has neural history to read; the 2nd+ pieces of a bout take that pre-roll
    from the end of the previous piece (same split, by construction).

    - ``stims[s]``: ``(1, F, T)`` causal log-mel spectrogram of the piezo audio
      (the decoding *target*), or ``(1, T * hop)`` raw audio with
      ``return_waveform=True``.
    - ``responses[s][n]``: ``(1, T)`` spike counts (R = 1, no repeats) for the
      item's bird; neurons of the other birds hold one shared all-NaN tensor.
      No motor-lead shift is applied: decoders learn their own lags.
    - ``stim_meta``: ``bird, episode_type ('warble'|'call'), bout_id, piece_idx,
      n_pieces, t_start_s, vocal_onset_s, vocal_offset_s, duration_s, split``.
      ``split`` is a fixed train / val / test assignment, per bird and
      stratified by episode type, that keeps every bout in one split.
    - ``nrn_meta``: ``bird, cluster_id, shank, max_channel, coordinates, fr``.

    Use ``select_pop_by_nrn_attr('bird', 'Ti81')`` for one bird (each bird is a
    separate population), and ``select_stims_by_attr('split', 'train')`` for a split.

    Parameters
    ----------
    path : path-like, optional
        Slim cache folder (written by :func:`prepare_yang2025`). Defaults to
        ``default_cache_dir('Yang2025')``.
    birds : str or sequence of str, default 'all'
        Subset of ``('Ti81', 'Li145', 'Bl122', 'Or61')``.
    dt_ms : float, default 5.0
    n_mels, fmin, fmax, window_ms : spectrogram settings (default 32 bands,
        500 Hz - 10 kHz, 10-ms causal window). The piezo carries the vocal
        energy at 1-8 kHz.
    compression : {'log', 'cubic', 'none'}, default 'log'
        ``'log'`` = natural log of mel power (+1e-10), what
        :mod:`deepSTRF.metrics.decoding` expects.
    pre_roll_ms : float, default 300.0
    max_episode_s : float or None, default 5.0
    return_waveform : bool, default False
    raw_path : path-like, optional
        Raw release folder to prepare the cache from, if the cache is missing.
    download : bool, default False
        If the cache is missing, fetch the needed ~3.3 GB of the Zenodo zip into
        ``default_cache_dir('Yang2025_raw')`` (or ``raw_path``) and prepare it.
    """

    def __init__(self,
                 path: Optional[Union[str, Path]] = None,
                 birds: Union[str, Sequence[str]] = "all",
                 dt_ms: float = 5.0,
                 n_mels: int = 32,
                 fmin: float = 500.0,
                 fmax: float = 10000.0,
                 window_ms: float = 10.0,
                 compression: str = "log",
                 pre_roll_ms: float = 300.0,
                 max_episode_s: Optional[float] = 5.0,
                 return_waveform: bool = False,
                 raw_path: Optional[Union[str, Path]] = None,
                 download: bool = False):
        path = Path(path).expanduser() if path is not None else default_cache_dir("Yang2025")
        if compression not in ("log", "cubic", "none"):
            raise ValueError(f"compression must be 'log', 'cubic' or 'none', got {compression!r}")
        if max_episode_s is not None and max_episode_s <= 0:
            raise ValueError(f"max_episode_s must be positive or None, got {max_episode_s}")
        if isinstance(birds, str):
            birds = None if birds == "all" else [birds]
        else:
            birds = [str(b) for b in birds]

        if not (path / "index.json").exists():
            unknown = sorted(set(birds or ()) - set(BIRDS))
            if unknown:                                   # fail before any download
                raise ValueError(f"Unknown bird(s) {unknown}; choose from {list(BIRDS)}")
            if raw_path is None and download:
                raw_path = download_yang2025(default_cache_dir("Yang2025_raw"))
            elif raw_path is not None and download:
                download_yang2025(raw_path)
            if raw_path is None:
                raise FileNotFoundError(
                    f"No Yang2025 cache at {path}. Pass download=True, or raw_path= to the "
                    "raw Zenodo release (prepare_yang2025(raw, path)).")
            prepare_yang2025(raw_path, path)

        index = json.loads((path / "index.json").read_text())
        if index.get("version") != CACHE_VERSION:
            raise ValueError(f"cache at {path} has version {index.get('version')}, expected "
                             f"{CACHE_VERSION}; delete it and prepare it again")
        available = list(index["birds"])
        birds = available if birds is None else birds
        unknown = sorted(set(birds) - set(available))
        if unknown:
            raise ValueError(f"Unknown bird(s) {unknown}; choose from {available}")

        super().__init__(str(path), dt_ms)
        self.audio_fs = int(index["audio_fs"])
        hop = self.audio_fs * dt_ms / 1000.0
        if abs(hop - round(hop)) > 1e-6:
            raise ValueError(f"audio_fs * dt_ms / 1000 = {hop} must be an integer")
        self._hop = int(round(hop))
        self.species = "budgerigar"
        self.birds = tuple(b for b in available if b in birds)
        self.F = int(n_mels)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.window_ms = float(window_ms)
        self.compression = compression
        self.pre_roll_ms = float(pre_roll_ms)
        self.max_episode_s = max_episode_s
        self.return_waveform = bool(return_waveform)
        n_fft = max(int(round(self.window_ms * 1e-3 * self.audio_fs)), self._hop)
        self._n_fft = n_fft
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_fs, n_fft=n_fft, hop_length=self._hop, n_mels=self.F,
            f_min=self.fmin, f_max=self.fmax, center=False)

        # ---------------------------------------------------------- neurons
        first_of = {}
        self.nrn_meta = []
        for bird in self.birds:
            first_of[bird] = len(self.nrn_meta)
            self.nrn_meta += [dict(bird=bird, **u) for u in index["units"][bird]]
        self.N_neurons = len(self.nrn_meta)

        # ---------------------------------------------------------- items
        nan_sentinel = torch.full((1, 1), float("nan"))
        spikes = {}
        for bird in self.birds:
            with np.load(path / bird / "spikes.npz") as z:
                spikes[bird] = (z["spike_times"], z["unit"])
        eps = sorted((e for e in index["episodes"] if e["bird"] in self.birds),
                     key=lambda e: (self.birds.index(e["bird"]), e["onset_s"]))
        self.stims, self.stim_meta, self.responses = [], [], []
        for e in eps:
            audio, _ = sf.read(str(path / e["audio_file"]), dtype="int16")
            audio = torch.from_numpy(audio.astype(np.float32) / 32768.0)
            length = e["offset_s"] - e["onset_s"]
            n_pieces = 1 if max_episode_s is None else max(1, math.ceil(length / max_episode_s - 1e-9))
            for k in range(n_pieces):
                p_on = e["onset_s"] + k * length / n_pieces
                p_off = e["onset_s"] + (k + 1) * length / n_pieces
                w0 = max(p_on - self.pre_roll_ms / 1000.0, e["audio_t0_s"])
                i0 = int(round((w0 - e["audio_t0_s"]) * self.audio_fs))
                t_start = e["audio_t0_s"] + i0 / self.audio_fs
                T = max(1, int(round((p_off - t_start) * 1000.0 / dt_ms)))
                self.stims.append(self._stim(audio, i0, T))
                self.stim_meta.append(dict(
                    bird=e["bird"], episode_type=e["episode_type"], bout_id=e["bout_id"],
                    piece_idx=k, n_pieces=n_pieces, t_start_s=t_start, vocal_onset_s=p_on,
                    vocal_offset_s=p_off, duration_s=T * dt_ms / 1000.0, split=e["split"]))
                self.responses.append(self._bin(spikes[e["bird"]], t_start, T, e["bird"],
                                                first_of, nan_sentinel))
        self.validate()

    # ------------------------------------------------------------------
    def _stim(self, audio: torch.Tensor, i0: int, T: int) -> torch.Tensor:
        n = T * self._hop
        w = audio[i0:i0 + n]
        w = torch.nn.functional.pad(w, (0, n - w.shape[-1]))
        if self.return_waveform:
            return w.unsqueeze(0).contiguous()
        # causal framing (as CRCNSAA5Dataset): frame t covers the n_fft samples
        # ending at the end of bin t; samples before the item start are taken
        # from the audio itself when available, zero otherwise
        left = self._n_fft - self._hop
        pre = audio[max(0, i0 - left):i0]
        x = torch.cat([torch.zeros(left - pre.shape[-1]), pre, w])
        with torch.no_grad():
            spec = self._mel(x)[..., :T]
        if self.compression == "log":
            spec = torch.log(spec + LOG_EPS)
        elif self.compression == "cubic":
            spec = spec.pow(1.0 / 3)
        return spec.unsqueeze(0)

    def _bin(self, bird_spikes, t_start: float, T: int, bird: str, first_of: dict,
             nan_sentinel: torch.Tensor) -> list:
        st, unit = bird_spikes
        b = np.floor((st - t_start) * 1000.0 / self.dt).astype(np.int64)
        ok = (b >= 0) & (b < T)
        n_units = sum(m["bird"] == bird for m in self.nrn_meta)
        counts = np.zeros((n_units, T), np.float32)
        np.add.at(counts, (unit[ok], b[ok]), 1.0)
        row = [nan_sentinel] * self.N_neurons
        for u in range(n_units):
            row[first_of[bird] + u] = torch.from_numpy(counts[u:u + 1])
        return row
