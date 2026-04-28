import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ParametricSigmoid(nn.Module):
    """
    4-parameter parametric sigmoid activation, as commonly used in the auditory neural response fitting literature.
    As described in Willmore et al. (2016), "Incorporating Midbrain Adaptation to Mean Sound Level
    Improves Models of Auditory Cortical Processing", JNeuroscience:

        "a is the minimum firing rate, b is the output dynamic range, c is the input inflection point, and d is the
        reciprocal of the gain"

    """
    def __init__(self, num_features: int, bias: bool = True):
        super(ParametricSigmoid, self).__init__()

        self.N = num_features
        self.bias = bias

        # per-neuron parameters
        self.b = torch.nn.Parameter(torch.ones(self.N))
        self.c = torch.nn.Parameter(torch.zeros(self.N))
        self.d = torch.nn.Parameter(torch.ones(self.N))

        # initialization
        torch.nn.init.uniform_(self.b, 0.5, 1.5)
        torch.nn.init.uniform_(self.c, -0.5, 0.5)
        torch.nn.init.uniform_(self.d, 0.5, 1.5)

        if self.bias:
            self.a = torch.nn.Parameter(torch.zeros(self.N))
            torch.nn.init.uniform_(self.a, 0., 1.)
            def sigmoid_fn(x):
                return self.b / (1 + torch.exp(-(x - self.c) / self.d)) + self.a
        else:
            def sigmoid_fn(x):
                return self.b / (1 + torch.exp(-(x - self.c) / self.d))

        self.sigmoid = sigmoid_fn

    def forward(self, x):
        # x.shape = (B, N) or (B, T, N) or (*, N)
        return self.sigmoid(x)

    def __str__(self):
        return f"ParametricSigmoid({self.N}, bias={self.bias})"


class ParametricDoubleExponential(nn.Module):
    """
    4-parameter parametric double exponential activation, as commonly used in the auditory neural response fitting literature.
    As described in Thorson et al. (2015), "The essential complexity of auditory receptive fields", PLoS Comp. Biol.:

        "the baseline spike rate, saturated firing rate, firing threshold, and gain are represented by b, a, s and k
        respectively"

    """
    def __init__(self, num_features: int, bias: bool = True):
        super(ParametricDoubleExponential, self).__init__()

        self.N = num_features
        self.bias = bias

        # per-neuron parameters
        self.a = torch.nn.Parameter(torch.ones(self.N))
        self.k = torch.nn.Parameter(-torch.ones(self.N))
        self.s = torch.nn.Parameter(torch.zeros(self.N))

        # initialization
        torch.nn.init.uniform_(self.a, 0.5, 1.5)
        torch.nn.init.uniform_(self.k, -0.5, 0.5)
        torch.nn.init.uniform_(self.s, 0.5, 1.5)

        if self.bias:
            self.b = torch.nn.Parameter(torch.zeros(self.N))
            torch.nn.init.uniform_(self.b, 0., 1.)
            def double_exp_fn(x):
                return self.a * torch.exp(-torch.exp(self.k * x - self.s)) + self.b
        else:
            def double_exp_fn(x):
                return self.a * torch.exp(-torch.exp(self.k * x - self.s))

        self.double_exp = double_exp_fn

    def forward(self, x):
        # x.shape = (B, N) or (B, T, N) or (*, N)
        return self.double_exp(x)

    def __str__(self):
        return f"ParametricDoubleExponential({self.N}, bias={self.bias})"
