import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

deviceinit = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
class FlipminiWfilter(nn.Module):
    """
    Filters a spectrogram along temporal dimension with a mini sobel-like filter [-w; 1], with 0>w>1
        - w=0 --> raw spectrogram
        - w=1 --> time-derivative of the spectrogram

    The flipped version of the kernel is also included [1; -w], as we've seen that it boosts downstream model. That is,
    OFF responses matter.

    As a result, takes a 1-channel tensor as input, and returns a 2-channel tensor as output.

    """
    def __init__(self, init_w: float = 0.5, learnable: bool = True):
        super(FlipminiWfilter, self).__init__()
        self.w = init_w if not learnable else Parameter(Tensor([init_w]))

    def build_kernel(self):
        device = deviceinit
        #kernel = torch.ones(2, 1, 1, 2).to(self.w.device)
        kernel = torch.ones(2, 1, 1, 2).to(device)
        kernel[0, :, :, 0] *= -self.w  # first output filter: ON response
        kernel[1, :, :, 1] *= -self.w  # second output filter: OFF response = flipped version of ON kernel
        return kernel

    def forward(self, x):
        device = deviceinit
        #device = x.device
        kernel = self.build_kernel().to(device)
        out = F.pad(x, pad=(0, 1, 0, 0), mode='reflect')
        out = F.conv2d(out, kernel, stride=1)
        return out

    def get_w(self):
        return float(self.w.detach())


class ExponentialWfilter(nn.Module):
    """
    Filters a spectrogram along temporal dimension with an exponential high-pass filter of the following form:

                kernel = [...; -Cwa²; -Cwa; -Cw; +1]    with    C=1/(... + a² + a + 1)
                                                                so that the sum of the negative terms equals w

    This filter effectively computes the difference between the current value of the signal and a exponential average of
    its recent past.

    In this kernel, w and a are learnable:
        - w is related to the amplitude of the filter
        - a is related to its time constant

    The 'flipped' version of the kernel, i.e. accounting for OFF responses, is

            kernel_flipped = kernel = [+1; -Cw; -Cwa; -Cwa²; ...]

    As for FlipminiWfilter, takes a 1-channel tensor as input, and returns a 2-channel tensor as output.

    """
    def __init__(self, kernel_length: int,
                 init_w: float = 0.5, w_learnable: bool = True, init_a: float = 0.5, a_learnable: bool = True):
        super(ExponentialWfilter, self).__init__()
        self.kernel_length = kernel_length
        self.w = Tensor([init_w]) if not w_learnable else Parameter(Tensor([init_w]))
        self.a = Tensor([init_a]) if not a_learnable else Parameter(Tensor([init_a]))

    def build_kernel(self):
        kernel = torch.ones(2, 1, 1, self.kernel_length)

        # build C coefficient
        sum = 0.
        for n in range(self.kernel_length-1):
            sum += torch.pow(self.a, n)
        C = 1/sum

        # first output filter: ON response
        for n in range(self.kernel_length-1):
            kernel[0, :, :, 1 + n] *= -C * self.w * torch.pow(self.a, n)

        # second output filter: OFF response = 'flipped' version of ON kernel
        for n in range(self.kernel_length-1):
            kernel[1, :, :, -2 - n] *= -C * self.w * torch.pow(self.a, n)

        return kernel

    def forward(self, x):
        device = deviceinit
        #device = x.device
        kernel = self.build_kernel().to(device)
        P = (self.kernel_length - 1) // 2

        out = F.pad(x, pad=(P, P, 0, 0), mode='reflect')
        out = F.conv2d(out, kernel, stride=1)
        return out

    def get_w(self):
        return float(self.w.detach())

    def get_a(self):
        return float(self.a.detach())

    def plot_kernel(self):
        kernel = self.build_kernel()
        ON_filter = kernel[0, :].squeeze().detach().to(device).numpy()
        OFF_filter = kernel[1, :].squeeze().detach().to(device).numpy()

        plt.figure()
        plt.stem(torch.arange(0, self.kernel_length, 1).numpy(), ON_filter, 'r', markerfmt='ro', label='ON')
        plt.stem(torch.arange(0, self.kernel_length, 1).numpy(), OFF_filter, 'b', markerfmt='bo', label='OFF')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    energies = torch.rand(1, 1, 64, 1001).cuda()

    wfilt = FlipminiWfilter(init_w=0.5, learnable=True)
    energies_filtered_1 = wfilt(energies)
    print(wfilt.build_kernel())
    print(wfilt.get_w())
    print(wfilt.parameters())
    print("ok")

    ewfilt = ExponentialWfilter(kernel_length=7, init_w=0.8, w_learnable=True, init_a=0.4, a_learnable=True)
    energies_filtered_2 = ewfilt(energies)
    print(ewfilt.build_kernel())
    print(ewfilt.get_w(), ewfilt.get_a())
    print(ewfilt.parameters())
    ewfilt.plot_kernel()
