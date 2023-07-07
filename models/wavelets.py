import torch
import numpy as np


def gammatone(t, n=4, a=1, b=100, fc=500):
    """
    Computes the image of a tensor by a gammatone function of given characteristics.

    :param t: input tensor
    :param n: filter order; usually 4.
    :param a: filter amplitude; 1 by default.
    :param b: -3 dB filter bandwidth [Hz]
    :param fc: filter center frequency [Hz]
    :return:
    """
    return a * (t ** (n-1)) * torch.exp(- 2 * np.pi * b * t) * torch.cos(2 * np.pi * fc * t)


def sinc(t, b, fc):
    """
    Wavelet with a rectangular bandpass frequency response of center frequency fc and width b
    Indeed, the Fourier Transform of sinc is a rectangle; The first term was added to position the response correctly
    in frequency domain.

    :param t: input tensor
    :param b: filter bandwidth [Hz]
    :param fc: filter center frequency [Hz]
    :return:
    """
    return 2 * b * torch.cos(2 * np.pi * fc * t) * torch.sinc(b * t)


def sinc2(t, b, fc):
    """
    'Mel' wavelet, with a triangular bandpass frequency response of center frequency fc and width b
    Indeed, the Fourier Transform of sinc^2 is a triangle. The first term was added to position the response correctly
    in frequency domain.

    :param t: input tensor
    :param b: filter bandwidth [Hz]
    :param fc: filter center frequency [Hz]
    :return:
    """
    return b * torch.cos(2 * np.pi * fc * t) * (torch.sinc(b / 2 * t) ** 2)  # OK
