"""Audio I/O helpers for the waveform-input branch of audio datasets."""
from typing import Optional

import torch


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
    import torchaudio

    wav, fs_in = torchaudio.load(path)  # (channels, samples), float32
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
