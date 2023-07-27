import torch.nn as nn
import torch.nn.functional
import torch.nn.functional
from DCLS.construct.modules import Dcls1d
from spikingjelly.clock_driven.layer import SeqToANNContainer

from models.minisobel_init import get_filters
from models.rrfs import RRF1d



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

##################################################################################

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

##################################################################################

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
