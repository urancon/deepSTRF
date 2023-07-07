import torch
import torch.nn as nn
import torch.nn.functional
import spikingjelly
from spikingjelly.clock_driven.layer import STDPLearner, SynapseFilter, SeqToANNContainer
from spikingjelly.clock_driven.neuron import ParametricLIFNode, LIFNode, IFNode
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode, MultiStepIFNode
from spikingjelly.clock_driven import functional

from network.temporal_filters import FlipminiWfilter
from network.prefiltering import FrequencyDependentHighPassExponential
from network.rrfs import RRF1d, RRF2d
from network.minisobel_init import get_filters

from DCLS.construct.modules import Dcls1d



class RRF_GRU_Learnable_Delays(nn.Module):
    """
    Same as the baseline GRU_RRF1d_Net but with learnable delays, in the first (RRF) layer.
    Each synapse in this RRF layer learns its own specific delay up to T timestep in the past.

    TODO:
     - inclure la batchnorm sur les channels de sortie + la fonction d'activation en sortie de dcls-rrfs
     - use Batchnorm1d, not 2d
     - batchnorm !! do it in layer by layer mode, by passing timesteps in the batch dimension (requires more memory though)

    """

    def __init__(self, max_delay: int = 1, rrf_kernel_size=7, rrf_stride=3, hidden_channels: int = 10):
        super(RRF_GRU_Learnable_Delays, self).__init__()

        self.F = 34  # Rahman et al. dataset
        self.N_delays = 1
        self.T_max = max_delay
        self.K = rrf_kernel_size
        self.S = rrf_stride
        self.C = hidden_channels

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:2')
        self.minisobel = get_filters(device)

        # pre-compute the hidden size out of the RRF layer
        # cf. torch.nn.Conv1d's doc with padding=0 and dilation=0
        self.F_down = int((self.F - self.K) / self.S + 1)

        # dimension of GRUs' hidden state vector
        self.H = self.F_down * self.C

        # synapses with locally connected weights in frequency and learnable delays
        self.P = (self.T_max - 1) // 2
        self.dcls = Dcls1d(in_channels=2 * self.K * self.F_down, out_channels=self.H, padding=self.P,
                           kernel_count=self.N_delays, dilated_kernel_size=self.T_max, groups=self.F_down)

        # normalization + non-linear activation
        # TODO: check whether time step by time step <==> layer by layer propagation for BatchNorm
        self.activation = nn.Sequential(
            SeqToANNContainer(
                nn.BatchNorm1d(self.C),
                nn.Sigmoid(),
            ),
        )

        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        B, C, F, T = x.shape
        x = self.minisobel(x)  # (B, 2, F, T)

        x = torch.nn.functional.unfold(x, kernel_size=(self.K, T), stride=(self.S, 1))    # (B, 2*K*T, F_down)

        x = x.permute(0, 2, 1)      # (B, F_down, 2*K*T)
        x = x.reshape(B, self.F_down * 2 * self.K, T)      # (B, F_down*2*K, T)

        y = self.dcls(x)    # (B, F_down*C_out, T)

        y = y.permute(2, 0, 1)      # (T, B, F_down*C_out)
        y = y.reshape(T, B, self.F_down, self.C).transpose(2, 3)    # (T, B, C_out, F_down)  TODO: ou bien y.reshape(T, B, self.C, self.F_down) ? en fction de si la 2eme dim de la shape est C_out*F_down ou F_down*C_out ?
        y = self.activation(y)      # (T, B, C_out, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)      # (B, T, C_out*F_down)

        z, _ = self.gru(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass
