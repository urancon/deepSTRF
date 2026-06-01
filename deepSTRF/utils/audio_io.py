"""Audio I/O helpers for the waveform-input branch of audio datasets."""
from typing import Optional, Tuple

import torch


def load_wav(path: str) -> Tuple[torch.Tensor, int]:
    """Load a WAV file as a float32 ``(channels, samples)`` tensor + sample rate.

    Uses ``soundfile`` (libsndfile) rather than ``torchaudio.load``: the latter
    routes through ``torchcodec`` on torchaudio >= 2.x, which needs FFmpeg
    shared libraries (``libavutil`` ...) that aren't present on bare CI runners
    or minimal installs. ``soundfile`` decodes PCM/float WAV directly with no
    FFmpeg dependency, and returns values bit-identical to
    ``torchaudio.load(path, normalize=True)`` for PCM wavs.

    Parameters
    ----------
    path : str
        Path to the WAV file.

    Returns
    -------
    (torch.Tensor, int)
        ``(wav, sample_rate)`` where ``wav`` is float32 of shape
        ``(channels, samples)`` (channels-first, matching ``torchaudio.load``).
    """
    import soundfile as sf

    data, fs = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    wav = torch.from_numpy(data.T).contiguous()                # (channels, samples)
    return wav, int(fs)


def load_resampled_mono_wav(path: str, target_fs: int,
                            target_length: Optional[int] = None) -> torch.Tensor:
    """Load a WAV file, downmix to mono, resample to ``target_fs``.

    Parameters
    ----------
    path : str
        Path to the WAV file.
    target_fs : int
        Target sample rate in Hz.
    target_length : int, optional
        If given, the returned tensor is right-cropped or right-padded with
        zeros so its last axis has exactly ``target_length`` samples. Useful
        when audio durations are very close to but not exactly the neural-data
        window (e.g. NS1's 5.000 s wavs vs the 4.995 s precomputed spec).

    Returns
    -------
    torch.Tensor
        Float32 mono waveform of shape ``(1, T)``. ``T == target_length`` if
        provided, else the natural resampled length.
    """
    import torchaudio  # only torchaudio.functional.resample (pure torch, no FFmpeg)

    wav, fs_in = load_wav(path)              # (channels, samples), float32
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # downmix to mono

    if fs_in != target_fs:
        wav = torchaudio.functional.resample(wav, fs_in, target_fs)

    if target_length is not None:
        T = wav.shape[-1]
        if T > target_length:
            wav = wav[..., :target_length]
        elif T < target_length:
            pad = torch.zeros(wav.shape[0], target_length - T,
                              dtype=wav.dtype, device=wav.device)
            wav = torch.cat([wav, pad], dim=-1)

    return wav.float()
