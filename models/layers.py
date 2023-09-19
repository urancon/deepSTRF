import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LearnableExponentialDecay(nn.Module):
    """
    TODO: description --> Rahman et al. (DNet) + Fang et al. (PLIF)

    TODO:
     - multiple input channel       | Now: C_in=1 only
     - multiple output channel      | Now: C_out=1 only
     - make 1d and 2d versions      | Now: 1d processing with expected 2d input

    Takes a (B, F, T) tensor as input, and returns a low-pass version of the same shape)

    """
    def __init__(self, input_size: int, kernel_size: int, init_tau: float = 2., decay_input: bool = True):
        super(LearnableExponentialDecay, self).__init__()

        # general attributes
        self.input_size = input_size
        self.K = kernel_size
        self.decay_input = decay_input

        # initialization as in Rahman et al.
        init_d = torch.ones(input_size).exponential_(lambd=(1/math.sqrt(init_tau - 1.)))

        # learnable parameters (one per feature, so C_out in total)
        self.d = Parameter(init_d)

    def build_kernel(self, device='cpu'):
        """
        Creates a parametrized kernel to be convolved with the last (temporal) dimension of the input tensor
        This output kernel has for shape: (input_size, 1, kernel_size)

        """
        kernel = torch.ones(self.input_size, 1, self.K)
        kernel = kernel * (1 - 1 / (1 + self.d ** 2)).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.K)
        kernel = kernel ** torch.arange(0, self.K).flip(0)
        kernel = kernel / (1 + self.d ** 2).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.K) if self.decay_input else kernel
        return kernel.to(device)

    def forward(self, x):
        """
        Convolves each frequency band of input 1-channel spectrogram with filters and standardize the output.

        :param spectro_in: shape is (B, C, F, T) with B=Batch, C=Channels=1 (raw spectrogram), F=#Frequency_bands, T=#Timesteps
        :return: a tensor of the same shape
        """
        # build exponential kernel
        kernel = self.build_kernel(x.device)

        # convolve input spectrogram with the kernel
        x = x.squeeze(1)                                                        # (B, 1, F, T)  --> (B, F, T)
        x = nn.functional.pad(x, pad=(self.K - 1, 0), mode='replicate')         # (B, F, T) --> (B, F, T+K-1)
        x = nn.functional.conv1d(x, kernel, stride=1, groups=self.input_size)   # (B, F, T+K-1) --> (B, F, T)
        x = x.unsqueeze(1)                                                      # (B, F, T) --> (B, 1, F, T)

        return x

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def tau(self):
        with torch.no_grad():
            tau = 1 + self.d ** 2
        return tau


class ParametricSigmoid(nn.Module):
    """
    4-parameter parametric sigmoid activation, as commonly used in the auditory neural response fitting literature.
    As described in Willmore et al. (2016), "Incorporating Midbrain Adaptation to Mean Sound Level
    Improves Models of Auditory Cortical Processing", JNeuroscience:

        "a is the minimum firing rate, b is the output dynamic range, c is the input inflection point, and d is the
        reciprocal of the gain"


    """
    def __init__(self, num_features: int, bias: bool = True):
        super().__init__()

        self.N = num_features
        self.bias = bias

        if self.bias:
            self.a = torch.nn.Parameter(torch.zeros(self.N))
            torch.nn.init.uniform_(self.a, 0., 1.)

        self.b = torch.nn.Parameter(torch.ones(self.N))
        self.c = torch.nn.Parameter(torch.zeros(self.N))
        self.d = torch.nn.Parameter(torch.ones(self.N))

        torch.nn.init.uniform_(self.b, 0.5, 1.5)
        torch.nn.init.uniform_(self.c, -0.5, 0.5)
        torch.nn.init.uniform_(self.d, 0.5, 1.5)

    def forward(self, x):
        # x.shape = (B, N) or (B, T, N) or (*, N)
        if self.bias:
            y = self.b / (1 + torch.exp(-(x - self.c) / self.d)) + self.a
        else:
            y = self.b / (1 + torch.exp(-(x - self.c) / self.d))
        return y

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    '''
    leaker = ExponentialDecay(input_size=5000, kernel_size=10, init_tau=2., decay_input=True)
    print(leaker.tau())
    print(leaker.tau().mean())
    B, C, F, T = 1, 1, 5000, 200
    x = torch.zeros(B, C, F, T)
    x[:, :, :, 50:100] = 1.
    y = leaker(x)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x.detach()[0, 0, 0], label='x')
    plt.plot(y.detach()[0, 0, 0], label='y')
    plt.legend()
    plt.show()
    '''

    activation = ParametricSigmoid(10, bias=False)
    print(activation(torch.rand(2, 4, 10)).shape)
    print("ok")
