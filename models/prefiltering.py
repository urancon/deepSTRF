import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
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
        spectro_in = F.pad(spectro_in, pad=(self.K-1, 0), mode='replicate').to(self.device)            # (B, F, T)     --> (B, F, T+1)  # TODO: use padding or not ?
        out_ON = F.conv1d(spectro_in, kernel_ON, stride=1, groups=self.F)                # (B, F, T+1)   --> (B, F, T)
        out_OFF = F.conv1d(spectro_in, kernel_OFF, stride=1, groups=self.F)              # (B, F, T+1)   --> (B, F, T)

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


if __name__ == "__main__":

    from encoding.filterbanks import FilterBank

    # create gammatone filterbank
    n_freqs = 64
    freq_range = (50, 20000)
    filter_length = 301  # 301  # 1501
    energy_window_length = 80  # 80  # 400
    energy_stride = 40  # 40  # 200
    fbank = FilterBank(filter_type='gammatone', scale='mel',
                       freq_range=freq_range, n_filters=n_freqs,
                       sampling_rate=44100, filter_length=filter_length,
                       energy_window_length=energy_window_length, energy_stride=energy_stride)

    # create minisobel initialized with a specific distribution of time constants
    center_freqs = torch.flip(fbank.CFs, dims=(0,))
    time_constants = FrequencyDependentHighPassExponential.freq_to_tau(center_freqs)
    a_vals = FrequencyDependentHighPassExponential.tau_to_a(time_constants)
    a_vals[0] = 0.6
    w_vals = torch.ones_like(a_vals) * 0.3
    filters = FrequencyDependentHighPassExponential(init_a_vals=a_vals, init_w_vals=w_vals, kernel_size=10, learnable=True)
    filters.plot_kernels()

    # sound stimulus
    from scipy.io import wavfile

    sample_rate, wave = wavfile.read('../datasets/Misc/dog_bark.wav')
    wave = wave[0:55000, 0]  # keep only 1/2 channel of the stereo file. Shape: (n_samples, 2) --> (n_samples, )
    plt.title('input signal waveform')
    plt.plot(wave, color='black')
    plt.xlabel('time (s)')
    plt.ylabel('sound pressure amplitude')
    plt.show()

    wave = torch._cast_Float(torch.from_numpy(wave).unsqueeze(0).unsqueeze(0))
    #spectro = fbank(wave)

    spectro = torch.zeros(1, 64, 5000) # let's imagine dt=1ms, low channel (dim=1) indices are low center freqs, high channel indices are high center frequencies
    spectro[:, :, 1500:3500] = 100.
    #spectro[:, 0, 0] = 100

    #spectro = torch.ones(1, 64, 5000) * 100.
    #spectro[:, :, 1200:2500] = 0.

    # prefilter spectrogram
    spectro_filtered = filters(spectro)
    # spectro_filtered = torch.abs(filters(spectro))

    ON_response = spectro_filtered[:, 0, :]
    OFF_response = spectro_filtered[:, 1, :]

    plt.subplots(3, 1)
    plt.subplot(311)
    plt.title('spectrogram energy')
    plt.imshow(spectro.detach().squeeze().flip(0))
    plt.xlabel('time')
    plt.ylabel('frequency sub-band')
    plt.colorbar()
    plt.subplot(312)
    plt.title('ON energy')
    plt.imshow(ON_response.detach().squeeze().flip(0))
    plt.xlabel('time')
    plt.ylabel('frequency sub-band')
    plt.colorbar()
    plt.subplot(313)
    plt.title('OFF energy')
    plt.imshow(OFF_response.detach().squeeze().flip(0))
    plt.xlabel('time')
    plt.ylabel('frequency sub-band')
    plt.colorbar()
    plt.show()

    print("ok")
