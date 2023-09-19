import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from models.scales import mel_to_Hz, Hz_to_mel, ERB, Greenwood, inverse_Greenwood
from utils.filterbanks import FilterBank

import matplotlib.pyplot as plt


class FrequencyDependentHighPassExponential(nn.Module):
    """
    High-pass exponential filter with frequency dependent time constants.

    Independently filters each frequency band of an input spectrogram along temporal dimension with a parametrized
    exponential kernel:

        kernel = [...; -Cwa²; -Cwa; -Cw; +1]    with    C=1/(... + a² + a + 1)
                                                                so that the sum of the negative terms equals w

    This filter effectively computes the difference between the current value of the signal in each frequency band and
    an exponential average of its recent past.

    # TODO: talk about the flipped version of minisobel and ON-OFF responses

    The kernel is flat along frequency dimension, and we apply padding='same' to keep the same time dimension
    As a result, takes a 1-channel tensor as input, and returns a 2-channel tensor as output.
    input_spectrogram.shape = (B, 1, F, T)
    output_spectrogram.shape = (B, 2, F, T)

    """
    def __init__(self, init_a_vals, init_w_vals, kernel_size: int = 2, learnable: bool = True):
        """
        init_a_vals: a 1D vector of 'a' parameters (related to the time constant of the kernel's exponential). The
         higher the 'a', the higher the corresponding time constant of the exponential

        init_w_vals: a 1D vector of 'w' parameters (representing the weight given to the exponential average of the
         signal in its recent past)

        """
        super(FrequencyDependentHighPassExponential, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # make sure passed a and w are one-dimensional
        assert init_a_vals.shape == init_w_vals.shape
        assert len(init_a_vals.shape) == 1

        # general attributes
        self.F = len(init_a_vals)
        self.K = kernel_size

        # learnable parameters (one per freq.)
        self.a = init_a_vals if not learnable else Parameter(init_a_vals)
        self.w = init_w_vals if not learnable else Parameter(init_w_vals)

    def build_kernels(self):
        """
        Creates two parametrized kernels:
         - one for the ON response, highlighting onsets in the signal
         - one for the OFF response (offsets), which is the flipped version of the ON kernel

        """
        # normalization constant
        a_device = self.a.to(self.device)
        ones_device = torch.ones(self.K - 1).to(self.device)
        range_device = torch.arange(0, self.K - 1).to(self.device)

        C = 1 / (torch.outer(a_device, ones_device) ** range_device).sum(dim=1)
        C = C.to(self.device)

        # ON kernel begins with an exponential whose elements sum to -w, then finishes with +1
        kernel_ON = torch.ones(self.F, 1, self.K).to(self.device)

        C_device = C.to(self.device)
        w_device = self.w.to(self.device)

        kernel_ON[:, 0, 1:] = (-C_device * w_device).unsqueeze(-1) * (
                    torch.outer(a_device, ones_device) ** range_device)
        kernel_ON = torch.flip(kernel_ON, dims=(2,)).to(self.device)

        # OFF kernel begins with an exponential whose elements sum to +1, then finishes with +w
        kernel_OFF = - kernel_ON / self.w.unsqueeze(1).unsqueeze(1).to(self.device)
        kernel_OFF[:, 0, -1] = - self.w

        return kernel_ON, kernel_OFF

    def forward(self, spectro_in):
        """
        Convolves each frequency band of input 1-channel spectrogram with filters and standardize the output.

        :param spectro_in: shape is (B, C, F, T) with B=Batch, C=Channels=1 (raw spectrogram), F=#Frequency_bands, T=#Timesteps
        :return: a tensor of shape (B, 2, F, T). First channel is for the ON response, second channel for the OFF one.
        """
        #device = spectro_in.device

        # reshape input spectrogram from single-channel 2D representation to multi-channel 1D
        spectro_in = spectro_in.squeeze(1)                                      # (B, 1, F, T)  --> (B, F, T)

        # build ON and OFF high-pass exponential kernels
        kernel_ON, kernel_OFF = self.build_kernels()
        kernel_ON, kernel_OFF = kernel_ON.to(self.device), kernel_OFF.to(self.device)

        # convolve input spectrogram with the kernels
        spectro_in = nn.functional.pad(spectro_in, pad=(self.K-1, 0), mode='replicate').to(self.device)            # (B, F, T)     --> (B, F, T+1)  # TODO: use padding or not ?
        out_ON = nn.functional.conv1d(spectro_in, kernel_ON, stride=1, groups=self.F)                # (B, F, T+1)   --> (B, F, T)
        out_OFF = nn.functional.conv1d(spectro_in, kernel_OFF, stride=1, groups=self.F)              # (B, F, T+1)   --> (B, F, T)

        # reshape output from a 1D back to 2D representation
        spectro_out = torch.stack([out_ON, out_OFF], dim=1)                              # (B, 2, F, T)

        return spectro_out

    @staticmethod
    def freq_to_tau(freqs):
        """
        Finds the associated midbrain neuron time constants (in ms) to a set of frequencies (in Hz)
        For more details, see Willmore et al. (2016),
            "Incorporating Midbrain Adaptation to Mean Sound Level Improves Models of Auditory Cortical Processing"

        """
        return 500. - 105. * torch.log10(freqs)

    @staticmethod
    def tau_to_a(time_constants, dt: float = 1):
        """
        Converts values of physical time constants (in ms) to corresponding 'a' parameters (adimensional), given a fixed
        time step (in ms)

        """
        return torch.exp(- dt / time_constants)

    def plot_kernels(self, frequency_bin=0):
        kernel_ON, kernel_OFF = self.build_kernels()
        ON_filter = kernel_ON[frequency_bin, :].squeeze().detach().to(self.device).numpy()
        OFF_filter = kernel_OFF[frequency_bin, :].squeeze().detach().to(self.device).numpy()

        plt.figure()
        plt.stem(torch.arange(0, self.K, 1).numpy(), ON_filter, 'r', markerfmt='ro', label='ON')
        plt.stem(torch.arange(0, self.K, 1).numpy(), OFF_filter, 'b', markerfmt='bo', label='OFF')
        plt.legend()
        plt.show()


class Willmore_Adaptation(nn.Module):
    """
    High-pass exponential filter with frequency dependent time constants.

    TODO: description

    Independently filters each frequency band of an input spectrogram along temporal dimension with a parametrized
    exponential kernel:

        kernel = [...; -Cwa²; -Cwa; -Cw; +1]    with    C=1/(... + a² + a + 1)
                                                                so that the sum of the negative terms equals w

    This filter effectively computes the difference between the current value of the signal in each frequency band and
    an exponential average of its recent past.

    # TODO: talk about the flipped version of minisobel and ON-OFF responses

    The kernel is flat along frequency dimension, and we apply padding='same' to keep the same time dimension
    As a result, takes a 1-channel tensor as input, and returns a 2-channel tensor as output.
    input_spectrogram.shape = (B, 1, F, T)
    output_spectrogram.shape = (B, 2, F, T)

    """
    def __init__(self, n_bands: int, init_a, init_w: float = 1., kernel_size: int = 499, learnable: bool = False):
        """
        init_a_vals: a 1D vector of 'a' parameters (related to the time constant of the kernel's exponential). The
         higher the 'a', the higher the corresponding time constant of the exponential

        # TODO: add some variance to w values ?

        init_w_vals: a 1D vector of 'w' parameters (representing the weight given to the exponential average of the
         signal in its recent past)

        """
        super(Willmore_Adaptation, self).__init__()

        # general attributes
        self.F = n_bands
        self.K = kernel_size

        # learnable parameters (one per freq.)
        self.a = init_a if not learnable else Parameter(init_a)
        self.w = torch.ones(self.F) * init_w if not learnable else Parameter(torch.ones(self.F) * init_w)

    def build_kernel(self, device='cpu'):
        """
        TODO: description

        Creates a parametrized kernel

        """
        C = 1 / (torch.outer(self.a, torch.ones(self.K - 1)) ** torch.arange(0, self.K - 1)).sum(dim=1)

        # kernel begins with an exponential whose elements sum to -w, then finishes with +1
        kernel = torch.ones(self.F, 1, self.K)
        kernel[:, 0, 1:] = (-C * self.w).unsqueeze(-1) * (torch.outer(self.a, torch.ones(self.K - 1)) ** torch.arange(0, self.K - 1))
        kernel = torch.flip(kernel, dims=(2,))

        return kernel.to(device)

    def forward(self, x):
        """
        Convolves each frequency band of input 1-channel spectrogram with filters and standardize the output.

        :param spectro_in: shape is (B, C, F, T) with B=Batch, C=Channels=1 (raw spectrogram), F=#Frequency_bands, T=#Timesteps
        :return: a tensor of shape (B, 2, F, T). First channel is for the ON response, second channel for the OFF one.
        """
        # build exponential kernel
        kernel = self.build_kernel(x.device)

        # convolve input spectrogram with the kernel
        x = x.squeeze(1)                                                        # (B, 1, F, T)  --> (B, F, T)
        x = nn.functional.pad(x, pad=(self.K-1, 0), mode='replicate')           # (B, F, T) --> (B, F, T+K-1)
        x = nn.functional.conv1d(x, kernel, stride=1, groups=self.F)            # (B, F, T+K-1) --> (B, F, T)
        x = torch.abs(x)                                                        # (B, F, T), half-wave rectification
        x = x.unsqueeze(1)                                                      # (B, F, T) --> (B, 1, F, T)

        return x

    @staticmethod
    def freq_to_tau(freqs):
        """
        Finds the associated midbrain neuron time constants (in ms) to a set of frequencies (in Hz)
        For more details, see Willmore et al. (2016),
            "Incorporating Midbrain Adaptation to Mean Sound Level Improves Models of Auditory Cortical Processing"

        """
        return 500. - 105. * torch.log10(freqs)

    @staticmethod
    def tau_to_a(time_constants, dt: float = 1):
        """
        Converts values of physical time constants (in ms) to corresponding 'a' parameters (adimensional), given a fixed
        time step (in ms)

        """
        return torch.exp(- dt / time_constants)

    def plot_kernels(self, frequency_bin=0):
        kernel = self.build_kernels()
        filter = kernel[frequency_bin, :].squeeze().detach().to(self.device).numpy()

        plt.figure()
        plt.stem(torch.arange(0, self.K, 1).numpy(), filter, 'r', markerfmt='ro', label='ON')
        plt.legend()
        plt.show()


def get_filters(device):
    '''
    TODO: remove this function !!!

    '''

    print(f"\nselected device: {device}\n")

    n_freqs = 34
    freq_range = (500, 20000)
    filter_length = 1501  # 301  # 1501
    energy_window_length = 442  # 80  # 400
    energy_stride = 221  # 40  # 200
    fbank = FilterBank(filter_type='gammatone', scale='mel',
                       freq_range=freq_range, n_filters=n_freqs,
                       sampling_rate=44100, filter_length=filter_length,
                       energy_window_length=energy_window_length, energy_stride=energy_stride)

    for param in fbank.parameters():
        param.requires_grad_(False)
    fbank.eval()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    fbank = fbank.to(device)

    center_freqs = fbank.CFs
    time_constants = FrequencyDependentHighPassExponential.freq_to_tau(center_freqs)
    a_vals = FrequencyDependentHighPassExponential.tau_to_a(time_constants, dt=5)
    w_vals = torch.ones_like(a_vals) * 0.75

    filters = FrequencyDependentHighPassExponential(init_a_vals=a_vals, init_w_vals=w_vals, kernel_size=200, learnable=True)
    #filters.plot_kernels()

    return filters


def get_init_a(n_bands: int, f_min: float, f_max: float, scale: str, dt: int = 5):

    # TODO: this piece of code is repeated in the FilterBank class --> make a function
    if scale == 'mel':
        min_cf = Hz_to_mel(torch.tensor(f_min))
        max_cf = Hz_to_mel(torch.tensor(f_max))
        CFs = torch.linspace(min_cf, max_cf, n_bands)
        CFs = mel_to_Hz(CFs)
    elif scale == 'greenwood':
        min_cf = Greenwood(torch.tensor(f_min))
        max_cf = Greenwood(torch.tensor(f_max))
        CFs = torch.linspace(min_cf, max_cf, n_bands)
        CFs = inverse_Greenwood(CFs)
    else:
        raise NotImplementedError(f"scale argument should be 'mel' or 'greenwood', not '{scale}'")

    taus = FrequencyDependentHighPassExponential.freq_to_tau(CFs)
    a = FrequencyDependentHighPassExponential.tau_to_a(taus, dt=dt)

    return a
