"""Causal log-mel spectrogram for the ``wav2spec`` slot."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMelSpectrogram(nn.Module):
    """Strictly-causal log-mel spectrogram, intended for the ``wav2spec``
    slot of an :class:`~deepSTRF.models.audio.AudioEncodingModel`.

    Defaults reproduce the **Rahman et al. 2019** cochleagram used by the
    NS1 dataset (PLoS Comp Biol 15(5): e1006618):

    - 10 ms Hanning window, 5 ms hop
    - 34 log-spaced channels, 500–22 627 Hz
    - amplitude (not power) spectrogram
    - threshold-clipped log compression

    Input shape  ``(B, 1, T_audio)``  — mono audio at ``audio_fs`` Hz.
    Output shape ``(B, 1, n_mels, T_neural)`` — log-mel spectrogram with an
    explicit ``C_in = 1`` channel axis so the rest of the deepSTRF pipeline
    (``prefiltering → core → readout``) is rank-compatible with spec-input
    models.

    Causality
    ---------
    Output frame ``t`` only sees audio samples in ``[0, (t+1) * hop)`` (i.e.
    no samples from neural bin ``t+1`` or later). Implementation: the input
    is left-padded by ``win - hop`` zeros, ``n_fft`` is locked to ``win`` so
    every STFT frame is exactly the windowed region (no internal zero-pad
    that would otherwise stretch the framing stride past ``win``), and the
    STFT runs with ``center=False``. Analysis frame ``t`` thus ends at audio
    sample ``(t+1) * hop - 1`` of the original waveform.

    Parameters
    ----------
    audio_fs : int
        Input audio sample rate in Hz.
    n_mels : int, default 34
        Number of mel bands. Default matches NS1's precomputed-spec layout.
    hop_ms : float, default 5.0
        Hop length in ms. Equals the dataset's neural ``dt_ms``. Determines
        ``hop = round(audio_fs * hop_ms / 1000)`` audio samples per output
        frame.
    win_ms : float, default 10.0
        STFT window length in ms. Must be ``>= hop_ms``. Default matches
        Rahman 2019.
    f_min : float, default 500.0
    f_max : float, optional
        Frequency limits for the mel filterbank (Hz). ``None`` → Nyquist.
        Default (500, 22627) Hz matches Rahman 2019. Note: ``audio_fs``
        must support ``f_max`` (i.e. ``audio_fs >= 2 * f_max``), otherwise
        ``f_max`` is clamped to Nyquist.
    magnitude : {'amplitude', 'power'}, default 'amplitude'
        Whether the STFT output is squared (power) or kept as magnitude
        (amplitude) before mel-weighting. Rahman 2019 uses amplitude.
    log_floor : float, default 1e-4
        Pre-log threshold: values below this are clamped up to this floor
        before the ``log``. Matches the "values below a low threshold were
        set to the threshold" step in Rahman 2019. Ignored when
        ``log_offset`` is not None.
    log_offset : float, optional
        Legacy log shape: ``log(mel + log_offset)``. When None (default),
        the Rahman threshold-clip form ``log(max(mel, log_floor))`` is
        used. Set to a positive number to recover ``log(mel + offset)``
        behaviour (the convention this module shipped with originally).
    mel_scale : {'htk', 'slaney'}, default 'htk'
        Mel-frequency conversion convention. Rahman 2019 uses voicebox's
        ``melbank.m`` (slightly different but in the same log family); the
        torchaudio ``'htk'`` curve is the closest off-the-shelf match.
    """

    def __init__(self, audio_fs: int, n_mels: int = 34,
                 hop_ms: float = 5.0, win_ms: float = 10.0,
                 f_min: float = 500.0, f_max: Optional[float] = 22627.0,
                 magnitude: str = "amplitude",
                 log_floor: float = 1e-4,
                 log_offset: Optional[float] = None,
                 mel_scale: str = "htk"):
        super().__init__()
        if audio_fs <= 0:
            raise ValueError(f"audio_fs must be positive (got {audio_fs})")
        if win_ms < hop_ms:
            raise ValueError(f"win_ms ({win_ms}) must be >= hop_ms ({hop_ms})")
        if magnitude not in ("amplitude", "power"):
            raise ValueError(
                f"magnitude must be 'amplitude' or 'power' (got {magnitude!r})"
            )

        self.audio_fs = int(audio_fs)
        self.n_mels = int(n_mels)
        self.out_channels = self.n_mels
        self.hop = max(1, int(round(audio_fs * hop_ms / 1000.0)))
        self.win = max(1, int(round(audio_fs * win_ms / 1000.0)))
        # n_fft locked to win so the STFT frame stride matches the windowed
        # region exactly (see the causality note in the class docstring).
        self.n_fft = self.win
        self.magnitude = magnitude
        self.log_floor = float(log_floor)
        self.log_offset = float(log_offset) if log_offset is not None else None
        self.f_min = float(f_min)
        nyquist = audio_fs / 2.0
        if f_max is None:
            self.f_max = nyquist
        else:
            self.f_max = min(float(f_max), nyquist)
        self.left_pad = self.win - self.hop  # >=0 by the check above

        # Hann window — register as buffer so .to(device) follows.
        self.register_buffer("_window",
                             torch.hann_window(self.win, periodic=False, dtype=torch.float32))

        # Mel filterbank — precomputed at construction; register as buffer.
        # torchaudio.functional.melscale_fbanks: (n_fft//2 + 1, n_mels) at the
        # n_fft frequency grid.
        import torchaudio.functional as taF
        fb = taF.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min, f_max=self.f_max,
            n_mels=self.n_mels, sample_rate=self.audio_fs,
            norm=None, mel_scale=mel_scale,
        )  # (n_fft//2+1, n_mels)
        self.register_buffer("_mel_fb", fb)

    def extra_repr(self) -> str:
        log_part = (f"log_offset={self.log_offset}" if self.log_offset is not None
                    else f"log_floor={self.log_floor}")
        return (f"audio_fs={self.audio_fs}, n_mels={self.n_mels}, hop={self.hop}, "
                f"win={self.win}, n_fft={self.n_fft}, magnitude={self.magnitude!r}, "
                f"f_range=({self.f_min}, {self.f_max}), {log_part}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T_audio)
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"CausalMelSpectrogram expects (B, 1, T_audio); got {tuple(x.shape)}"
            )
        # left-pad so that, with center=False, output frame t ends at audio
        # sample (t+1)*hop - 1 of the original waveform.
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))

        B = x.shape[0]
        # collapse mono channel for stft; restore later.
        stft = torch.stft(
            x.view(B, -1), n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self._window, center=False, return_complex=True,
        )  # (B, n_fft//2+1, T_neural)
        power = stft.real.pow(2) + stft.imag.pow(2)  # (B, n_fft//2+1, T_neural)
        if self.magnitude == "amplitude":
            spec = power.sqrt()
        else:  # 'power'
            spec = power
        mel = self._mel_fb.transpose(0, 1) @ spec     # (B, n_mels, T_neural)
        if self.log_offset is not None:
            log_mel = torch.log(mel + self.log_offset)
        else:
            log_mel = torch.log(mel.clamp(min=self.log_floor))
        return log_mel.unsqueeze(1)  # (B, 1, n_mels, T_neural) — explicit C_in axis
