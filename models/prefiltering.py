import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from models.scales import mel_to_Hz, Hz_to_mel, ERB, Greenwood, inverse_Greenwood
from utils.filterbanks import FilterBank


def get_CFs(min_freq, max_freq, n_freqs, scale):
    """
    Returns n_freqs cochlear/center frequencies (CFs) scaled according to a function (typically logarithmic) within a
    range.

    """
    if scale == 'mel':
        min_cf = Hz_to_mel(torch.tensor(min_freq))
        max_cf = Hz_to_mel(torch.tensor(max_freq))
        CFs = torch.linspace(min_cf, max_cf, n_freqs)
        CFs = mel_to_Hz(CFs)
    elif scale == 'greenwood':
        min_cf = Greenwood(torch.tensor(min_freq))
        max_cf = Greenwood(torch.tensor(max_freq))
        CFs = torch.linspace(min_cf, max_cf, n_freqs)
        CFs = inverse_Greenwood(CFs)
    else:
        raise NotImplementedError(f"'scale' argument should be 'mel' or 'greenwood', not '{scale}'")
    return CFs


def freq_to_tau(freqs):
    """
    Finds the associated midbrain neuron time constants (in ms) to a set of frequencies (in Hz)
    For more details, see Willmore et al. (2016),
        "Incorporating Midbrain Adaptation to Mean Sound Level Improves Models of Auditory Cortical Processing"

    """
    return 500. - 105. * torch.log10(freqs)


def tau_to_a(time_constants, dt: float = 1):
    """
    Converts values of physical time constants (in ms) to corresponding 'a' parameters (adimensional), given a fixed
    time step (in ms)

    """
    return torch.exp(- dt / time_constants)


def a_to_tau(a, dt: float = 1):
    return - dt / torch.log(a)


class Willmore_Adaptation(nn.Module):
    """
    High-pass exponential filter with frequency dependent time constants.

    Independently filters each frequency band of an input spectrogram along temporal dimension with a parametrized
    exponential kernel:

        kernel = [...; -Cwa²; -Cwa; -Cw; +1]    with    C=1/(... + a² + a + 1)
                                                                so that the sum of the negative terms equals w

    This filter effectively computes the difference between the current value of the signal in each frequency band and
    an exponential average of its recent past.

    The kernel is flat along frequency dimension, and we apply padding='same' to keep the same time dimension
    As a result, takes a 1-channel tensor as input, and returns a 2-channel tensor as output.
    input_spectrogram.shape = (B, 1, F, T)
    output_spectrogram.shape = (B, 2, F, T)

    """
    def __init__(self, init_a_vals, kernel_size: int = 2):
        """
        init_a_vals: a 1D vector of 'a' parameters (related to the time constant of the kernel's exponential). The
         higher the 'a', the higher the corresponding time constant of the exponential

        init_w_vals: a 1D vector of 'w' parameters (representing the weight given to the exponential average of the
         signal in its recent past)

        """
        super(Willmore_Adaptation, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # make sure passed a and w are one-dimensional
        assert len(init_a_vals.shape) == 1

        # general attributes
        self.F = len(init_a_vals)
        self.K = kernel_size

        # learnable parameters (one per freq.)
        self.a = init_a_vals

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

        # kernel begins with an exponential whose elements sum to -w, then finishes with +1
        kernel = torch.ones(self.F, 1, self.K).to(self.device)

        C_device = C.to(self.device)
        #w_device = torch.ones_like(self.a).to(self.device)

        #kernel[:, 0, 1:] = (-C_device * w_device).unsqueeze(-1) * (torch.outer(a_device, ones_device) ** range_device)
        kernel[:, 0, 1:] = -C_device.unsqueeze(-1) * (torch.outer(a_device, ones_device) ** range_device)
        kernel = torch.flip(kernel, dims=(2,)).to(self.device)

        return kernel

    def forward(self, spectro_in):
        """
        Convolves each frequency band of input 1-channel spectrogram with filters and standardize the output.

        :param spectro_in: shape is (B, C, F, T) with B=Batch, C=Channels=1 (raw spectrogram), F=#Frequency_bands, T=#Timesteps
        :return: a tensor of shape (B, 2, F, T). First channel is for the ON response, second channel for the OFF one.
        """
        #device = spectro_in.device

        # reshape input spectrogram from single-channel 2D representation to multi-channel 1D
        spectro_in = spectro_in.squeeze(1)                                      # (B, 1, F, T)  --> (B, F, T)

        # build high-pass exponential kernel
        kernel = self.build_kernels()
        kernel = kernel.to(self.device)

        # convolve input spectrogram with the kernels
        spectro_in = F.pad(spectro_in, pad=(self.K-1, 0), mode='replicate').to(self.device)            # (B, F, T)     --> (B, F, T+1)
        out = F.conv1d(spectro_in, kernel, stride=1, groups=self.F)                # (B, F, T+1)   --> (B, F, T)

        # full-wave rectification
        out = torch.abs(out)

        # reshape output from a 1D back to 2D representation
        spectro_out = torch.unsqueeze(out, dim=1)                                        # (B, 1, F, T)

        return spectro_out

    def plot_kernels(self, frequency_bin=0):
        kernel = self.build_kernels()
        filter = kernel[frequency_bin, :].squeeze().detach().cpu().numpy()

        plt.figure()
        plt.stem(torch.arange(0, self.K, 1).numpy(), filter, 'r', markerfmt='ro', label='Willmore')
        plt.legend()
        plt.show()


class AdapTrans(nn.Module):
    """
    High-pass exponential filter with frequency dependent time constants.

    TODO: ref of our paper

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

    Version of AdapTrans with different (a, w) pairs for each polarity

    """
    def __init__(self, init_a_vals, init_w_vals, kernel_size: int = 2, learnable: bool = True):
        """
        init_a_vals: a 1D vector of 'a' parameters (related to the time constant of the kernel's exponential). The
         higher the 'a', the higher the corresponding time constant of the exponential

        init_w_vals: a 1D vector of 'w' parameters (representing the weight given to the exponential average of the
         signal in its recent past)

        """
        super(AdapTrans, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # make sure passed a and w are one-dimensional
        assert init_a_vals.shape == init_w_vals.shape
        assert len(init_a_vals.shape) == 1

        # general attributes
        self.F = len(init_a_vals)
        self.K = kernel_size

        # conversion
        init_d_vals = torch.sqrt(1/torch.Tensor(init_a_vals) - 1)
        init_p_vals = torch.sqrt(1/torch.Tensor(init_w_vals) - 1)

        # learnable parameters (one per freq.)
        self.d_on = init_d_vals if not learnable else Parameter(init_d_vals)
        self.d_off = init_d_vals if not learnable else Parameter(init_d_vals)
        #self.p_on = init_p_vals if not learnable else Parameter(init_p_vals)
        #self.p_off = init_p_vals if not learnable else Parameter(init_p_vals)
        self.p = init_p_vals if not learnable else Parameter(init_p_vals)

    def build_kernels(self):
        """
        Creates two parametrized kernels:
         - one for the ON response, highlighting onsets in the signal
         - one for the OFF response (offsets), which is the flipped version of the ON kernel

        """
        kernel_ON = self.ON_kernel(self.d_on, self.p)
        kernel_OFF = self.OFF_kernel(self.d_off, self.p)
        return kernel_ON, kernel_OFF

    def ON_kernel(self, d, p):
        """
        Creates the ON kernel
        """
        # normalization constant
        a_device = 1 / (1 + d.to(self.device).pow(2))
        ones_device = torch.ones(self.K - 1).to(self.device)
        range_device = torch.arange(0, self.K - 1).to(self.device)

        C = 1 / (torch.outer(a_device, ones_device) ** range_device).sum(dim=1)
        C = C.to(self.device)

        # ON kernel begins with an exponential whose elements sum to -w, then finishes with +1
        kernel_ON = torch.ones(self.F, 1, self.K).to(self.device)

        C_device = C.to(self.device)
        w_device = 1 / (1 + p.to(self.device).pow(2))

        kernel_ON[:, 0, 1:] = (-C_device * w_device).unsqueeze(-1) * (torch.outer(a_device, ones_device) ** range_device)
        kernel_ON = torch.flip(kernel_ON, dims=(2,)).to(self.device)

        return kernel_ON

    def OFF_kernel(self, d, p):
        """
        Creates the ON kernel
        """
        # normalization constant
        a_device = 1 / (1 + d.to(self.device).pow(2))
        ones_device = torch.ones(self.K - 1).to(self.device)
        range_device = torch.arange(0, self.K - 1).to(self.device)

        C = 1 / (torch.outer(a_device, ones_device) ** range_device).sum(dim=1)
        C = C.to(self.device)

        # ON kernel begins with an exponential whose elements sum to -w, then finishes with +1
        kernel_ON = torch.ones(self.F, 1, self.K).to(self.device)

        C_device = C.to(self.device)
        w_device = 1 / (1 + p.to(self.device).pow(2))

        kernel_ON[:, 0, 1:] = (-C_device * w_device).unsqueeze(-1) * (torch.outer(a_device, ones_device) ** range_device)
        kernel_ON = torch.flip(kernel_ON, dims=(2,)).to(self.device)

        # OFF kernel begins with an exponential whose elements sum to +1, then finishes with +w
        kernel_OFF = - kernel_ON / w_device.unsqueeze(1).unsqueeze(1).to(self.device)
        kernel_OFF[:, 0, -1] = - w_device

        return kernel_OFF

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
        spectro_in = nn.functional.pad(spectro_in, pad=(self.K-1, 0), mode='replicate').to(self.device)            # (B, F, T)     --> (B, F, T+1)
        out_ON = nn.functional.conv1d(spectro_in, kernel_ON, stride=1, groups=self.F)                # (B, F, T+1)   --> (B, F, T)
        out_OFF = nn.functional.conv1d(spectro_in, kernel_OFF, stride=1, groups=self.F)              # (B, F, T+1)   --> (B, F, T)

        # reshape output from a 1D back to 2D representation
        spectro_out = torch.stack([out_ON, out_OFF], dim=1)                              # (B, 2, F, T)

        return spectro_out

    def get_a(self):
        a_on = 1 / (1 + self.d_on.cpu().detach().pow(2))
        a_off = 1 / (1 + self.d_off.cpu().detach().pow(2))
        return a_on, a_off

    def get_w(self):
        return 1 / (1 + self.p.cpu().detach().pow(2))

    def plot_kernels(self, frequency_bin=0):
        kernel_ON, kernel_OFF = self.build_kernels()
        ON_filter = kernel_ON[frequency_bin, :].squeeze().detach().cpu().numpy()
        OFF_filter = kernel_OFF[frequency_bin, :].squeeze().detach().cpu().numpy()

        plt.figure()
        plt.stem(torch.arange(0, self.K, 1).numpy(), ON_filter, 'r', markerfmt='ro', label='ON')
        plt.stem(torch.arange(0, self.K, 1).numpy(), OFF_filter, 'b', markerfmt='bo', label='OFF')
        plt.legend()
        plt.show()
