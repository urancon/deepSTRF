import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional


class RRF2d(nn.Module):
    """
    Implementation of Restricted Receptive Fields for 2D image tensors
    A trade-off between Linear and Conv2d

    """
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(RRF2d, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = True

        self.S_in = input_size
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        prospective_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        prospective_input = torch.rand(1, in_channels, input_size[0], input_size[1])
        prospective_output = prospective_conv(prospective_input)
        self.S_out = prospective_output.shape[-2:]

        self.unfold = torch.nn.Unfold(kernel_size, dilation, padding, stride)
        prospective_unfold = self.unfold(prospective_input)
        self.L = prospective_unfold.shape[-1]

        self.weights = Parameter(torch.rand(self.C_out, self.C_in * self.K[0] * self.K[1], self.L))
        self.biases = Parameter(torch.rand(self.C_out, self.L)) if bias else None

        self.fold = torch.nn.Fold(output_size=self.S_out, kernel_size=(1, 1))

    def forward(self, x):
        patches = self.unfold(x)                                        # (B, C_in, H_in, W_in) --> (B, M, L)
        patches = patches.unsqueeze(1).repeat(1, self.C_out, 1, 1)      # (B, M, L)             --> (B, C_out, M, L)
        
        if self.bias:
            y = torch.sum(patches * self.weights, dim=2) + self.biases  # (B, C_out, M, L)      --> (B, C_out, L)
        else:
            y = torch.sum(patches * self.weights, dim=2)

        return self.fold(y)

    def __str__(self):
        s = f'RRF2d(input_size={self.S_in}, in_channels={self.C_in}, out_channels={self.C_out}, ' \
            f'kernel_size={self.K}, stride={self.stride}'
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s


class RRF1d(nn.Module):
    """
    Implementation of Restricted Receptive Fields for 1D tensors
    A trade-off between Linear and Conv1d

    Note:
        nn.Unfold is only compatible with images, so for 1d inputs, it is necessary to first unsqueeze them, perform the
        same operation as in RRF2d, and finally squeeze them back from 2d to 1d

    """
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(RRF1d, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = True

        self.S_in = input_size
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size

        prospective_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        prospective_input = torch.rand(1, in_channels, input_size)
        prospective_output = prospective_conv(prospective_input)
        self.S_out = prospective_output.shape[-1]

        self.unfold = torch.nn.Unfold((kernel_size, 1), dilation, padding, stride)
        prospective_unfold = self.unfold(prospective_input.unsqueeze(-1))
        self.L = prospective_unfold.shape[-1]

        self.weights = Parameter(torch.rand(self.C_out, self.C_in * self.K, self.L))
        self.biases = Parameter(torch.rand(self.C_out, self.L)) if bias else None

        self.fold = torch.nn.Fold(output_size=(self.S_out, 1), kernel_size=(1, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)                                             # (B, C_in, S_in)       --> (B, C_in, S_in, 1)
        patches = self.unfold(x)                                        # (B, C_in, H_in, W_in) --> (B, M, L)
        patches = patches.unsqueeze(1).repeat(1, self.C_out, 1, 1)      # (B, M, L)             --> (B, C_out, M, L)

        if self.bias:
            y = torch.sum(patches * self.weights, dim=2) + self.biases  # (B, C_out, M, L)      --> (B, C_out, L)
        else:
            y = torch.sum(patches * self.weights, dim=2)

        return y  # contrarily to RRF2d we have here torch.equal(y, self.fold(y).squeeze(-1)) == True because L=S_out

    def __str__(self):
        s = f'RRF1d(input_size={(self.S_in,)}, in_channels={self.C_in}, out_channels={self.C_out}, ' \
            f'kernel_size={self.K}, stride={self.stride}'
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s
