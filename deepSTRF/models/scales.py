import torch
import numpy as np


def Hz_to_mel(f):
    """Convert a frequency (Hz) to the mel scale.

    Parameters
    ----------
    f : torch.Tensor or float
        Frequency in Hz.

    Returns
    -------
    torch.Tensor or float
        Frequency on the mel scale.
    """
    return 2595 * torch.log10(1 + f/700)


def mel_to_Hz(mel_freqs):
    """Convert mel-scale frequencies back to Hz (inverse of :func:`Hz_to_mel`).

    Parameters
    ----------
    mel_freqs : torch.Tensor or float
        Frequency on the mel scale.

    Returns
    -------
    torch.Tensor or float
        Frequency in Hz.
    """
    return 700 * (10 ** (mel_freqs/2595) - 1)


def Greenwood(x, animal='human'):
    """Greenwood cochlear position-to-frequency map.

    Parameters
    ----------
    x : torch.Tensor or float
        Normalised position along the cochlea in ``[0, 1]``.
    animal : {'human', 'mouse'}, default 'human'
        Species whose Greenwood coefficients to use.

    Returns
    -------
    torch.Tensor or float
        Characteristic frequency in Hz.

    Raises
    ------
    NotImplementedError
        If ``animal`` is not ``'human'`` or ``'mouse'``.
    """
    if animal == 'human':
        return 165.4 * (10 ** (2.1 * x) - 0.88)
    elif animal == 'mouse':
        return 712.6 * (10 ** (2.1 * x) + 0.40)
    else:
        raise NotImplementedError


def inverse_Greenwood(f, animal='human'):
    """Inverse Greenwood map: frequency (Hz) to normalised cochlear position.

    Parameters
    ----------
    f : torch.Tensor or float
        Characteristic frequency in Hz.
    animal : {'human', 'mouse'}, default 'human'
        Species whose Greenwood coefficients to use.

    Returns
    -------
    torch.Tensor or float
        Normalised position along the cochlea in ``[0, 1]``.

    Raises
    ------
    NotImplementedError
        If ``animal`` is not ``'human'`` or ``'mouse'``.
    """
    if animal == 'human':
        return torch.log10((f + 0.88) / 165.4) / 2.1
    elif animal == 'mouse':
        return torch.log10((f - 0.40) / 712.6) / 2.1
    else:
        raise NotImplementedError
