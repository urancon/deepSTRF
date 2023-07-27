import torch.nn as nn
import torch.nn.functional
import torch.nn.functional
from DCLS.construct.modules import Dcls1d
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.layer import SeqToANNContainer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven.neuron import ParametricLIFNode

from models.minisobel_init import get_filters
from models.rrfs import RRF1d, RRF2d
from models.temporal_filters import FlipminiWfilter


##################################################################################


class StatelessConvNet(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """
    def __init__(self, n_bands=34, temporal_window_size: int = 9, n_hidden: int = 64):
        super(StatelessConvNet, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size
        self.H = n_hidden

        self.convs = nn.Sequential(
            nn.Conv2d(1, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T-1)//2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
            nn.Conv2d(self.H, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class PopStatelessConvNet(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """
    def __init__(self, n_bands=34, temporal_window_size: int = 9, n_hidden: int = 64):
        super(StatelessConvNet, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size
        self.H = n_hidden

        self.convs = nn.Sequential(
            nn.Conv2d(1, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T-1)//2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
            nn.Conv2d(self.H, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

        self.input_layer = nn.Sequential(

        )

        self.output_layer = nn.Sequential(


        )

    def forward(self, x):
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class RahmanDynamicNet(nn.Module):
    """

    Close re-implementation of the DNet model proposed in Rahman et al. (2019):
        "a dynamic network model of temporal receptive fields in auditory cortex"


    """

    def __init__(self, temporal_window_size: int = 5, n_hidden: int = 20):
        super(RahmanDynamicNet, self).__init__()

        self.F = 34
        self.T = temporal_window_size
        self.H = n_hidden

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
        )

        self.lays = nn.Sequential(
            MultiStepParametricLIFNode_V(init_tau=2., decay_input=True, v_threshold=1000., v_reset=None,
                                         backend='cupy'),
            SeqToANNContainer(
                nn.Linear(self.H, 1),
                nn.Sigmoid(),
            ),
            MultiStepParametricLIFNode_V(init_tau=2., decay_input=True, v_threshold=1000., v_reset=None,
                                         backend='cupy'),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, n_timesteps)
        y = self.conv(x)  # (B, H, 1, T)
        y = y.squeeze(2).permute(2, 0, 1)  # (T, B, H)

        v = self.lays(y)  # (T, B, 1)
        v = v.permute(1, 0, 2).squeeze(-1)  # (B, T)
        return v

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        functional.reset_net(self)


class GRU_RRF1d_Net(nn.Module):
    """
    TODO: need a better name !
    CAUTION: Not the sale results as the first version of the RRF1dGRUNet (class called 'GRUConvNet_RRF')
    
    ----------------------------------

    Now our BASELINE model
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a single timestep (i.e. a 1D frequency vector) of
    the spectrogram.

    In addition to help gradient descent make the architecture converge towards a better minimum (better performances),
    this RRF connectivity keeps tonotopicity and is more biologically plausible, because it is observed in the auditory
    pathway !

    -----------------------------------

    Best ANN model so far, and by far, with a mean test correlation coefficient of ~70.5% for the following config:
    - minisobel=True, T=1, C=10, K=7, S=3, H=10, #params=62222

    Other remarkable configs:
    - minisobel=True, T=1, C=7,  K=7, S=3, H=10, #params=30956,         avg_test_CC=70.1%
    - minisobel=True, T=1, C=1,  K=7, S=3, H=10, #params=824 (<1000 !), avg_test_CC=65.0%

    -----------------------------------

    remarkable RRF1d hyperparameters and sizes:
    - L_in=34, K=5, S=2 --> L_out=15
    - L_in=34, K=7, S=3 --> L_out=10

    """
    def __init__(self, n_bands=34, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10):
        super(GRU_RRF1d_Net, self).__init__()

        self.F = n_bands  # Rahman et al. dataset
        self.T = temporal_window_size
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device)

        # restrictive receptive fields
        self.rrfs = nn.Sequential(
                SeqToANNContainer(
                    RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S), # (K=5, S=2) --> H=15
                    nn.BatchNorm1d(self.C),
                    nn.Sigmoid()
                    ),
                )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)                    # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)    # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L*self.C
        
        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        
        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)                                       # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)                                   # (T, B, 2, F)
        y = self.rrfs(x)                                            # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)      # (B, T, C_hidd*F_down)

        z, _ = self.gru(y)                  # (B, T, H)
        w = self.fc(z)                      # (B, T, 1)
        return w.squeeze(-1)                # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class GRU_RRF1dplus_Net(nn.Module):
    """
    Same as the baseline GRU_RRF1d_Net but with delays, so T=temporal_window_size>1

    """
    def __init__(self, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10):
        super(GRU_RRF1dplus_Net, self).__init__()

        self.F = 34  # Rahman et al. dataset
        self.T = temporal_window_size
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels

        # compromise between representing transients and permanent features
        self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)

        # prepare spectrogram for sequential processing
        self.P = self.T - 1 
        self.pad = torch.nn.ZeroPad2d((self.P, 0, 0, 0))  # pad left
        self.unfold = torch.nn.Unfold(kernel_size=(self.F, self.T))  # extract patches

        # restrictive receptive fields
        self.rrfs = nn.Sequential(
                SeqToANNContainer(
                    RRF2d(input_size=(self.F, self.T), in_channels=2, out_channels=self.C, kernel_size=(self.K, self.T), stride=self.S), 
                    nn.BatchNorm2d(self.C),
                    nn.Sigmoid(),
                    ),
                )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF2d(input_size=(self.F, self.T), in_channels=2, out_channels=self.C, kernel_size=(self.K, self.T), stride=self.S)
            prospective_input = torch.rand(1, 2, 34, self.T)               # (B, 2, F, window_T) 
            prospective_output = prospective_rrfs(prospective_input)       # (B, C, F_down, 1)
            self.L = prospective_output.shape[-2]
        self.H = self.L*self.C

        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        B, C, F, T = x.shape
        x = self.minisobel(x)                                       # (B, 2, F, T)
       
        x = self.unfold(self.pad(x))                                # (B, 2*F*window_T, T)
        x = x.permute(2, 0, 1)                                      # (T, B, 2*F*window_T)
        x = x.view(T, B, 2, F, self.T)                              # (T, B, 2, F, self.T) 
     
        y = self.rrfs(x)                                            # (T, B, C_hidd, F_down, 1)
        y = y.squeeze(-1)                                           # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)      # (B, T, C_hidd*F_down)

        z, _ = self.gru(y)                  # (B, T, H)
        w = self.fc(z)                      # (B, T, 1)
        return w.squeeze(-1)                # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


###############################################################################

from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate
import logging
try:
    import cupy
    from spikingjelly.clock_driven import neuron_kernel, cu_kernel_opt
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cu_kernel_opt = None

try:
    import lava.lib.dl.slayer as slayer

except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    slayer = None


def check_backend(backend: str):
    if backend == 'torch':
        return
    elif backend == 'cupy':
        assert cupy is not None, 'CuPy is not installed! You can install it from "https://github.com/cupy/cupy".'
    elif backend == 'lava':
        assert slayer is not None, 'Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl".'
    else:
        raise NotImplementedError(backend)


class MultiStepParametricLIFNode_V(ParametricLIFNode):
    def __init__(self, init_tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        """
        Same module as spikingjelly's MultiStepParametricLIFNode, but outputs the membrane potential instead of the
        spikes

        """
        super().__init__(init_tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepParametricLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.w.sigmoid(), self.decay_input, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return self.v_seq  # spike_seq
        else:
            raise NotImplementedError

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'


###############################################################################

class LIF_RRF1dplus_Net(nn.Module):
    """
    Same as the baseline GRU_RRF1d_Net but with delays, so T=temporal_window_size>1

    The GRUs are replaced by LIF neurons.

    """
    def __init__(self, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10):
        super(LIF_RRF1dplus_Net, self).__init__()

        self.F = 34  # Rahman et al. dataset
        self.T = temporal_window_size
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels

        # compromise between representing transients and permanent features
        self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)

        # prepare spectrogram for sequential processing
        self.P = self.T - 1
        self.pad = torch.nn.ZeroPad2d((self.P, 0, 0, 0))  # pad left
        self.unfold = torch.nn.Unfold(kernel_size=(self.F, self.T))  # extract patches

        # restrictive receptive fields
        self.rrfs_lifs = nn.Sequential(
                SeqToANNContainer(
                    RRF2d(input_size=(self.F, self.T), in_channels=2, out_channels=self.C, kernel_size=(self.K, self.T), stride=self.S),
                    nn.BatchNorm2d(self.C),
                    ),
                MultiStepParametricLIFNode(init_tau=2., decay_input=True, v_threshold=0.5, backend='cupy')
                )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF2d(input_size=(self.F, self.T), in_channels=2, out_channels=self.C, kernel_size=(self.K, self.T), stride=self.S)
            prospective_input = torch.rand(1, 2, 34, self.T)               # (B, 2, F, window_T)
            prospective_output = prospective_rrfs(prospective_input)       # (B, C, F_down, 1)
            self.L = prospective_output.shape[-2]
        self.H = self.L*self.C

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        B, C, F, T = x.shape
        x = self.minisobel(x)                                       # (B, 2, F, T)

        x = self.unfold(self.pad(x))                                # (B, 2*F*window_T, T)
        x = x.permute(2, 0, 1)                                      # (T, B, 2*F*window_T)
        x = x.view(T, B, 2, F, self.T)                              # (T, B, 2, F, self.T) 

        y = self.rrfs_lifs(x)                                       # (T, B, C_hidd, F_down, 1)
        y = y.squeeze(-1)                                           # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)      # (B, T, C_hidd*F_down=H)
        w = self.fc(y)                                              # (B, T, 1)
        return w.squeeze(-1)                                        # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        functional.reset_net(self)


"""
I have to do a pytorch model called "pop stateless convnet". It is inspired by convnet that i will show you.
Here is the definition of pop stateless convnet :
- a bank of temporal convolutional units each followed by one dense unit and static nonlinearity (dense : linear weighting of the outputs of the previous layer)
- inspired by Linear Nonlinear model

How it works :
you fit the parameters for all the recordings
you freeze the parameters
you train the parameters of the last layer for each individual recording.

Write the code using 

"""

##################################################################################

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
        device = torch.device('cuda:0')
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
