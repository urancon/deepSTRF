import torch
import numpy as np


def Hz_to_mel(f):
    return 2595 * torch.log10(1 + f/700)


def mel_to_Hz(mel_freqs):
    return 700 * (10 ** (mel_freqs/2595) - 1)


def Greenwood(x, animal='human'):
    if animal == 'human':
        return 165.4 * (10 ** (2.1 * x) - 0.88)
    elif animal == 'mouse':
        return 712.6 * (10 ** (2.1 * x) + 0.40)
    else:
        raise NotImplementedError


def inverse_Greenwood(f, animal='human'):
    if animal == 'human':
        return torch.log10((f + 0.88) / 165.4) / 2.1
    elif animal == 'mouse':
        return torch.log10((f - 0.40) / 712.6) / 2.1
    else:
        raise NotImplementedError
