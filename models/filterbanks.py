import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
import matplotlib.pyplot as plt
import scipy.io as scio

from network.scales import mel_to_Hz, Hz_to_mel, ERB, Greenwood, inverse_Greenwood
from network.wavelets import gammatone, sinc2, sinc


############################################

class FilterBank(nn.Module):
    """
    Class to hold time-domain filters modelling the frequency decomposition of auditory signals inside the cochlea.

    TODO? align impulse responses (they currently have too many points on the right, as these points are very close to
     zero.

    """
    def __init__(self, filter_type='gammatone', scale='mel', freq_range=(20, 20000), n_filters=30, sampling_rate=44100,
                 filter_length=500, energy_window_length=750, energy_stride=375, device = None):
        """
        Builds a filter bank with the desired properties.

        :param filter_type: 'gammatone', 'mel', 'CQT'
        :param scale: 'ERB', 'LIN', 'MEL', 'Bark', greenwood'
        :param sampling_rate: [Hz]
        :param freq_range: (min_freq, max_freq)
        :param n_filters: number of channels
        """
        super(FilterBank, self).__init__()
        self.device = device

        # keep important filterbank properties
        self.filter_type = filter_type
        self.scale = scale
        self.freq_range = freq_range
        self.n_filters = n_filters
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length

        # other important parameters
        self.default_conv_stride = 1
        self.default_conv_padding = 'same'
        self.energy_window_length = energy_window_length
        self.energy_stride = energy_stride

        # define center frequencies [Hz] on selected scale
        if self.scale == 'mel':
            min_cf = Hz_to_mel(torch.tensor(freq_range[0]))
            max_cf = Hz_to_mel(torch.tensor(freq_range[1]))
            self.CFs = torch.linspace(min_cf, max_cf, n_filters)
            self.CFs = mel_to_Hz(self.CFs)
        elif self.scale == 'greenwood':
            min_cf = Greenwood(torch.tensor(freq_range[0]))
            max_cf = Greenwood(torch.tensor(freq_range[1]))
            self.CFs = torch.linspace(min_cf, max_cf, n_filters)
            self.CFs = inverse_Greenwood(self.CFs)
        else:
            raise NotImplementedError(f"scale argument should be 'mel' or 'greenwood', not '{scale}'")

        # define filter bandwidths [Hz] according to selected function
        self.ERBs = ERB(self.CFs)

        # create wavelet filters
        kernels_wavelets = torch.zeros(self.n_filters, 1, filter_length)

        if filter_type == 'gammatone':
            xt = torch.linspace(0, self.filter_length, self.filter_length) / self.sampling_rate
            for i in range(n_filters):
                kernels_wavelets[i, 0, ] = gammatone(xt, fc=self.CFs[i], b=self.ERBs[i])
                kernels_wavelets[i, 0, ] = kernels_wavelets[i, 0, ] / torch.max(kernels_wavelets[i, 0, ])  # TODO? normalize filters

        elif filter_type == 'mel':
            xt = torch.linspace(-self.filter_length//2, self.filter_length//2, self.filter_length) / self.sampling_rate
            for i in range(n_filters):
                kernels_wavelets[i, 0, ] = sinc2(xt, fc=self.CFs[i], b=self.ERBs[i])
                kernels_wavelets[i, 0, ] = kernels_wavelets[i, 0, ] / torch.max(kernels_wavelets[i, 0, ])  # TODO? normalize filters

        else:
            AttributeError("'filter_type' should either be 'gammatone' or 'mel', but not {}".format(filter_type))

        self.spectro_conv = nn.Conv1d(in_channels=1, out_channels=self.n_filters, kernel_size=filter_length, stride=1, padding='same')
        self.spectro_conv.weight.data = kernels_wavelets

        # create kernels for energy framing
        kernel_ones = torch.ones((self.n_filters, 1, self.energy_window_length))
        eps = torch.as_tensor([1e-12]*self.n_filters, dtype=torch.float32)

        self.energy_conv = nn.Conv1d(in_channels=self.n_filters, out_channels=self.n_filters, groups=self.n_filters, kernel_size=self.energy_window_length, stride=self.energy_stride, padding=0)
        self.energy_conv.weight.data = kernel_ones
        self.energy_conv.bias.data = eps

    def convolute(self, signal):
        """
        Convolutes filters with signal in time domain, thereby producing a spectrogram

        :param signal: 1D vector containing audio samples. Shape: (B, Cin, Lin) = (B, 1, N_samples)
        :return: spectrogram. Shape: (B, Cout, Lout) = (B, N_filters, N_samples)
        """
        spectrogram = self.spectro_conv(signal)
        return spectrogram

    def compute_energy(self, spectrogram):
        """
        Applies a framing window of certain length and with a certain stride to compute the logarithmic energy of the
        spectrogram over time, emulating the signal processing of hair cells.

        more efficient method to do the energy framing:
          1. square spectrogram, element-wise
          2. For each frequency band, compute the sum of elements on a sliding time window
             (efficiently implemented with a convolution by a kernel filled with ones)
          3. take log (base 10) and multiply by 10

        :param spectrogram: tensor of shape (B, n_bands, n_samples)
        :return:  log-energy, tensor of shape: (B, Cout, Lout) = (B, n_bands, downsampled_T)
        """
        spectrogram = torch.pow(spectrogram, 2)
        energy = self.energy_conv(spectrogram)
        energy = 10 * torch.log10(energy)
        return energy

    def forward(self, signal):
        spectro = self.convolute(signal)
        energies = self.compute_energy(spectro)
        return energies

    def set_energy_parameters(self, window_length, stride):
        """
        Reset the parameters for energy framing to user-specified values

        :param window_length: an integer number of samples
        :param stride: an integer number of samples
        :return:
        """
        self.energy_window_length = window_length
        self.energy_stride = stride
        kernel_ones = torch.ones((self.n_filters, self.n_filters, self.energy_window_length))
        self.energy_conv.weight.data = kernel_ones
        self.energy_conv.stride = stride

    def plot(self, domain='freq'):
        """
        Displays the filters in logarithmic or linear frequency scale

        :param domain: 'freq' or 'time'
        :return:
        """
        assert domain == 'time' or domain == 'freq' or domain == 'frequency', \
            f"'domain' should either be 'time' or 'freq', but not '{domain}'"

        if domain == 'time':
            if self.filter_type == 'gammatone':
                xt = torch.linspace(0, self.filter_length, self.filter_length) / self.sampling_rate
            elif self.filter_type == 'mel':
                xt = torch.linspace(-self.filter_length//2, self.filter_length//2, self.filter_length) / self.sampling_rate
            else:
                raise NotImplementedError("implemented filter types: 'gammatone', 'mel'")

            for i in range(self.n_filters):
                yt = self.spectro_conv.weight.data[i, 0]
                plt.subplot(self.n_filters, 1, i + 1)
                if i < self.n_filters - 1:
                    plt.tick_params(
                        axis='both',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        left=False,
                        labelleft=False,
                        labelbottom=False)  # labels along the bottom edge are off
                else:
                    plt.tick_params(
                        axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        left=False,
                        labelleft=False)
                label_str = "fc={}, b={}".format(int(self.CFs[i]), int(self.ERBs[i]))
                plt.plot(xt.to(self.device), yt.to(self.device), label=label_str)
                plt.legend(loc="upper right")
            plt.show()

        elif domain == 'freq' or domain == 'frequency':

            xf = torch.linspace(0, self.sampling_rate // 2, self.filter_length // 2)  # positive frequencies only

            for i in range(self.n_filters):
                yt = self.spectro_conv.weight.data[i, 0] * torch.hann_window(self.filter_length)
                yf = torch.abs(torch.fft.fft(yt)[:self.filter_length // 2])
                plt.subplot(self.n_filters, 1, i + 1)
                if i < self.n_filters - 1:
                    plt.tick_params(
                        axis='both',
                        which='both',
                        bottom=False,
                        top=False,
                        left=False,
                        labelleft=False,
                        labelbottom=False)
                else:
                    plt.tick_params(
                        axis='y',
                        which='both',
                        left=False,
                        labelleft=False)
                label_str = "fc={}, b={}".format(int(self.CFs[i]), int(self.ERBs[i]))
                plt.plot(xf.to(self.device), 2 / self.filter_length * yf.to(self.device), label=label_str)
                plt.legend(loc="upper right")
            plt.show()

    def __str__(self):
        descr = f'{self.filter_type} filterbank\n' \
                f'  - centre frequencies scale: {self.scale}\n' \
                f'  - centre frequencies: {self.freq_range[0]}-{self.freq_range[1]} Hz\n' \
                f'  - number of filters: {self.n_filters}\n' \
                f'  - sampling rate: {self.sampling_rate}\n'
        return descr


if __name__ == "__main__":

    fbank = FilterBank(filter_type='gammatone', scale='mel',
                       freq_range=(500, 20000), n_filters=34,
                       sampling_rate=44100, filter_length=1500, energy_stride=240, energy_window_length=500)

    s1 = np.array(scio.loadmat('spectro1.mat')['spectro'][None,:].astype(np.double),dtype=np.double)
    s2 = np.array(scio.loadmat('spectro2.mat')['spectro'][None,:].astype(np.double),dtype=np.double)
    s3 = np.array(scio.loadmat('spectro3.mat')['spectro'][None,:].astype(np.double),dtype=np.double)

    data = torch.load('ns1.pt')

    sig1 = torch.tensor(s1, dtype=torch.float)
    sig2 = torch.tensor(s2,dtype=torch.float)
    sig3 = torch.tensor(s3, dtype=torch.float)

    sig_conv1 = fbank(sig1)
    sig_conv2 = fbank(sig2)
    sig_conv3 = fbank(sig3)

    view1 = sig_conv1.detach().numpy()[0][:]
    view2 = sig_conv2.detach().numpy()[0][:]
    view3 = sig_conv3.detach().numpy()[0][:]
    test1 = data['spectrograms'][0].detach().numpy()
    test2 = data['spectrograms'][1].detach().numpy()
    test3 = data['spectrograms'][2].detach().numpy()

    fig, axs = plt.subplots(2,1)
    axs[0] = plt.imshow(test1[0])
    axs[1] = plt.imshow(view1)
    print(fbank)
    fbank.plot(domain='time')
    fbank.plot(domain='frequency')




