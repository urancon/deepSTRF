import time
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional
from DCLS.construct.modules import Dcls1d
from spikingjelly.clock_driven.layer import SeqToANNContainer
from spikingjelly.activation_based.neuron import ParametricLIFNode, IzhikevichNode
from spikingjelly.activation_based.functional import reset_net

from subLSTM.nn import SubLSTM
from subLSTM.nn import SubLSTM as SubLSTMv2

from models.rrfs import RRF1d
import models.layers as layers

from utils.filterbanks import FilterBank
from models.prefiltering import FrequencyDependentHighPassExponential
import torch


def get_filters(device, minisobel_learnable=True, w=0.75, kernel_size=200):
    print(f"\nselected device: {device}\n")

    n_freqs = 49
    freq_range = (500, 20000)
    filter_length = 1501  # 301  # 1501
    energy_window_length = 442  # 80  # 400
    energy_stride = 221  # 40  # 200
    fbank = FilterBank(filter_type='gammatone', scale='mel',
                       freq_range=freq_range, n_filters=n_freqs,
                       sampling_rate=44100, filter_length=filter_length,
                       energy_window_length=energy_window_length, energy_stride=energy_stride)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    fbank = fbank.to(device)

    center_freqs = fbank.CFs
    time_constants = FrequencyDependentHighPassExponential.freq_to_tau(center_freqs)
    a_vals = FrequencyDependentHighPassExponential.tau_to_a(time_constants, dt=5)
    w_vals = torch.ones_like(a_vals) * w

    filters = FrequencyDependentHighPassExponential(init_a_vals=a_vals, init_w_vals=w_vals, kernel_size=kernel_size,
                                                    learnable=minisobel_learnable)
    # filters.plot_kernels()

    return filters


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
            nn.Conv2d(1, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
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

##################################################################################

class NRFmodel_WillmorePrefiltering(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9, n_hidden: int = 64):
        super(NRFmodel_WillmorePrefiltering, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size
        self.H = n_hidden

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device, minisobel_learnable=False, w=1.)

        self.convs = nn.Sequential(
            nn.Conv2d(1, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
            nn.Conv2d(self.H, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)[:, 0, :, :].abs().unsqueeze(1)  # (B, 1, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class NRFmodel_AdapTrans(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9, n_hidden: int = 64):
        super(NRFmodel_AdapTrans, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size
        self.H = n_hidden

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device)

        self.convs = nn.Sequential(
            nn.Conv2d(2, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
            nn.Conv2d(self.H, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class NRFmodel_Control(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9, n_hidden: int = 64):
        super(NRFmodel_Control, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size
        self.H = n_hidden

        self.convs = nn.Sequential(
            nn.Conv2d(1, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
            nn.Conv2d(self.H, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class Linearmodel_Control(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9):
        super(Linearmodel_Control, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size

        self.convs = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class Linearmodel_AdapTrans(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9):
        super(Linearmodel_AdapTrans, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size

        # compromise between representing transients and permanent features
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device)

        self.convs = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class LinearNonlinearmodel_Control(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9):
        super(LinearNonlinearmodel_Control, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size

        self.convs = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            layers.ParametricSigmoid(1, False)
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class LinearNonlinearmodel_AdapTrans(nn.Module):
    """
    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9):
        super(LinearNonlinearmodel_AdapTrans, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device)

        self.convs = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            layers.ParametricSigmoid(1, False)
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class DNet_Control(nn.Module):
    """
    TODO: description

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9, n_hidden: int = 64, init_tau=2., decay_input=True):
        super(DNet_Control, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size
        self.H = n_hidden

        self.convs = nn.Sequential(
            nn.Conv2d(1, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
            layers.LearnableExponentialDecay(self.H, kernel_size=round(init_tau * 7), init_tau=init_tau,
                                             decay_input=decay_input),
            nn.Conv2d(self.H, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
            layers.LearnableLearnableExponentialDecay(1, kernel_size=round(init_tau * 7), init_tau=init_tau,
                                                      decay_input=decay_input),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class DNet_AdapTrans(nn.Module):
    """
    TODO: description

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 9, n_hidden: int = 64, init_tau=2., decay_input=True):
        super(DNet_AdapTrans, self).__init__()

        self.F = n_bands
        self.T = temporal_window_size
        self.H = n_hidden

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device)

        self.convs = nn.Sequential(
            nn.Conv2d(2, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
            nn.BatchNorm2d(self.H),
            nn.Sigmoid(),
            layers.LearnableExponentialDecay(self.H, kernel_size=round(init_tau * 7), init_tau=init_tau,
                                             decay_input=decay_input),
            nn.Conv2d(self.H, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
            layers.LearnableExponentialDecay(1, kernel_size=round(init_tau * 7), init_tau=init_tau,
                                             decay_input=decay_input),
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        return self.convs(x).squeeze(1).squeeze(1)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class Pennington_2D_CNN_Control(nn.Module):
    """
    TODO: description

    Adapted from, but not entirely equivalent to the so-called '2D-CNN' of Pennington et al. (2023),
        "A convolutional neural network provides a generalizable model of natural sound coding by neural populations
        in auditory cortex", PLOS CB

    Major differences:
     - BatchNorm inside the convolutional backbone
     - Zero padding along frequency dimension --> downsampling along this dimension after self.convs

    Note:
     -

    """

    def __init__(self, n_bands=34, kernel_size: tuple = (3, 9), c_hidden: int = 10, n_hidden: int = 90,
                 output_neurons: int = 1):
        super(Pennington_2D_CNN_Control, self).__init__()

        self.F = n_bands
        self.K = kernel_size
        self.C = c_hidden
        self.H = n_hidden
        self.N_out = output_neurons

        self.convs = nn.Sequential(
            nn.Conv2d(1, self.C, kernel_size=self.K, stride=1, padding=(0, (self.K[1] - 1) // 2)),
            nn.BatchNorm2d(self.C),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.C, self.C, kernel_size=self.K, stride=1, padding=(0, (self.K[1] - 1) // 2)),
            nn.BatchNorm2d(self.C),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.C, self.C, kernel_size=self.K, stride=1, padding=(0, (self.K[1] - 1) // 2)),
            nn.BatchNorm2d(self.C),
            nn.LeakyReLU(0.1),
        )
        F_down = self.F - 3 * (self.K[0] - 1)  # frequency dimension after the 3 convs

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.C * F_down, out_features=self.H),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.H, out_features=self.N_out),
            layers.ParametricSigmoid(self.N_out, False)
        )

    def forward(self, x):
        # x.shape: (B, 1, F, T)
        x = self.convs(x)  # (B, C, F_down, T)
        x = x.flatten(start_dim=1, end_dim=2)  # (B, C*F_down, T)
        x = x.permute(0, 2, 1)  # (B, T, C*F_down)
        x = self.fc(x)  # (B, T, N)
        x = x.squeeze(2)  # TODO: remove this line once the prediction format has been normalized for all models
        return x  # TODO: should be (B, 1, T, N)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class Pennington_2D_CNN_AdapTrans(nn.Module):
    """
    TODO: description

    Adapted from, but not entirely equivalent to the so-called '2D-CNN' of Pennington et al. (2023),
        "A convolutional neural network provides a generalizable model of natural sound coding by neural populations
        in auditory cortex", PLOS CB

    Major differences:
     - BatchNorm inside the convolutional backbone
     - Zero padding along frequency dimension --> downsampling along this dimension after self.convs

    Note:
     -

    """

    def __init__(self, n_bands=34, kernel_size: tuple = (3, 9), c_hidden: int = 10, n_hidden: int = 90,
                 output_neurons: int = 1):
        super(Pennington_2D_CNN_AdapTrans, self).__init__()

        self.F = n_bands
        self.K = kernel_size
        self.C = c_hidden
        self.H = n_hidden
        self.N_out = output_neurons

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device)

        self.convs = nn.Sequential(
            nn.Conv2d(2, self.C, kernel_size=self.K, stride=1, padding=(0, (self.K[1] - 1) // 2)),
            nn.BatchNorm2d(self.C),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.C, self.C, kernel_size=self.K, stride=1, padding=(0, (self.K[1] - 1) // 2)),
            nn.BatchNorm2d(self.C),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.C, self.C, kernel_size=self.K, stride=1, padding=(0, (self.K[1] - 1) // 2)),
            nn.BatchNorm2d(self.C),
            nn.LeakyReLU(0.1),
        )
        F_down = self.F - 3 * (self.K[0] - 1)  # frequency dimension after the 3 convs

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.C * F_down, out_features=self.H),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=self.H, out_features=self.N_out),
            layers.ParametricSigmoid(self.N_out, False)
        )

    def forward(self, x):
        # x.shape: (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = self.convs(x)  # (B, C, F_down, T)
        x = x.flatten(start_dim=1, end_dim=2)  # (B, C*F_down, T)
        x = x.permute(0, 2, 1)  # (B, T, C*F_down)
        x = self.fc(x)  # (B, T, N)
        x = x.squeeze(2)  # TODO: remove this line once the prediction format has been normalized for all models
        return x  # TODO: should be (B, 1, T, N)

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
                RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 2, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z, _ = self.gru(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class SubLSTM_RRF1d_Net(nn.Module):
    """

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10):
        super(SubLSTM_RRF1d_Net, self).__init__()

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
                RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # GRUs for temporal processing
        self.sublstm = SubLSTM(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        # self.sublstm = nn.LSTM(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 2, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        t0 = time.time()
        z, _ = self.sublstm(y)  # (B, T, H)
        t1 = time.time()
        print("BABA", t1 - t0)
        print("KIKI", y.shape, z.shape)
        print("GOGO", y.device, self.sublstm.weight_hh_l0.device)

        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class SubLSTMv2_RRF1d_Net(nn.Module):
    """

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10):
        super(SubLSTMv2_RRF1d_Net, self).__init__()

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
                RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # GRUs for temporal processing
        self.sublstm = SubLSTMv2(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 2, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z, _ = self.sublstm(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


##################################################################################

##################################################################################

class Leaky_RRF1d_Net(nn.Module):
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
        super(Leaky_RRF1d_Net, self).__init__()

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
                RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # PLIs for temporal processing
        self.linear = nn.Linear(self.H, self.H)
        self.plif = ParametricLIFNode(init_tau=2., v_threshold=float('inf'), decay_input=True, step_mode='m',
                                      backend='cupy', store_v_seq=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 2, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z = self.linear(y)  # (B, T, H)
        self.plif(z.permute(1, 0, 2))  # (T, B, H)
        v = self.plif.v_seq
        w = self.fc(v.permute(1, 0, 2))  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        reset_net(self)


class Izhi_RRF1d_Net(nn.Module):
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
        super(Izhi_RRF1d_Net, self).__init__()

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
                RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # PLIs for temporal processing
        self.linear = nn.Linear(self.H, self.H)
        self.izhi = IzhikevichNode(tau=2.0, v_threshold=float('inf'), step_mode='m', backend='torch', store_v_seq=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 2, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z = self.linear(y)  # (B, T, H)
        self.izhi(z.permute(1, 0, 2))  # (T, B, H)
        v = self.izhi.v_seq
        print(v)
        w = self.fc(v.permute(1, 0, 2))  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        reset_net(self)


class GRU_RRF1d_Net_WillmorePrefiltering(nn.Module):
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
        super(GRU_RRF1d_Net_WillmorePrefiltering, self).__init__()

        self.F = n_bands  # Rahman et al. dataset
        self.T = temporal_window_size
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device, minisobel_learnable=False, w=1.)

        # restrictive receptive fields
        self.rrfs = nn.Sequential(
            SeqToANNContainer(
                RRF1d(input_size=self.F, in_channels=1, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)[:, 0, :, :].abs().unsqueeze(1)  # (B, 1, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 1, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z, _ = self.gru(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class GRU_RRF1d_Net_nonLearnableMinisobel(nn.Module):
    """

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10):
        super(GRU_RRF1d_Net_nonLearnableMinisobel, self).__init__()

        self.F = n_bands  # Rahman et al. dataset
        self.T = temporal_window_size
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device, minisobel_learnable=False)

        # restrictive receptive fields
        self.rrfs = nn.Sequential(
            SeqToANNContainer(
                RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 2, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z, _ = self.gru(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class GRU_RRF1d_Net_2ElemMinisobel(nn.Module):
    """

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10):
        super(GRU_RRF1d_Net_2ElemMinisobel, self).__init__()

        self.F = n_bands  # Rahman et al. dataset
        self.T = temporal_window_size
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels

        # compromise between representing transients and permanent features
        # self.minisobel = FlipminiWfilter(init_w=0.75, learnable=True)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.minisobel = get_filters(device, minisobel_learnable=False, kernel_size=2)

        # restrictive receptive fields
        self.rrfs = nn.Sequential(
            SeqToANNContainer(
                RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, 1)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = self.minisobel(x)  # (B, 2, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 2, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z, _ = self.gru(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass


class GRU_RRF1d_Net_noMinisobel(nn.Module):
    """

    """

    def __init__(self, n_bands=34, temporal_window_size: int = 1, kernel_size=7, stride=3, hidden_channels: int = 10,
                 n_neurons: int = 1):
        super(GRU_RRF1d_Net_noMinisobel, self).__init__()

        self.F = n_bands  # Rahman et al. dataset
        self.T = temporal_window_size
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels
        self.N = n_neurons

        # restrictive receptive fields
        self.rrfs = nn.Sequential(
            SeqToANNContainer(
                RRF1d(input_size=self.F, in_channels=1, out_channels=self.C, kernel_size=self.K, stride=self.S),
                # (K=5, S=2) --> H=15
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            ),
        )

        # get output shape and embedding space dim for GRUs
        with torch.no_grad():
            prospective_rrfs = RRF1d(input_size=self.F, in_channels=2, out_channels=self.C, kernel_size=self.K,
                                     stride=self.S)
            prospective_input = torch.rand(1, 2, n_bands)  # (B, 2, F)
            prospective_output = prospective_rrfs(prospective_input)  # (B, C, L)
            self.L = prospective_output.shape[-1]
        self.H = self.L * self.C

        # GRUs for temporal processing
        self.gru = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)

        # readout from GRUs' hidden state
        self.fc = nn.Linear(self.H, self.N)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        x = x.permute(3, 0, 1, 2)  # (T, B, 1, F)
        y = self.rrfs(x)  # (T, B, C_hidd, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_hidd*F_down)

        z, _ = self.gru(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def freeze(self):
        for params in self.rrfs.parameters():
            params.requires_grad = False
        for params in self.gru.parameters():
            params.requires_grad = False

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

        x = torch.nn.functional.unfold(x, kernel_size=(self.K, T), stride=(self.S, 1))  # (B, 2*K*T, F_down)

        x = x.permute(0, 2, 1)  # (B, F_down, 2*K*T)
        x = x.reshape(B, self.F_down * 2 * self.K, T)  # (B, F_down*2*K, T)

        y = self.dcls(x)  # (B, F_down*C_out, T)

        y = y.permute(2, 0, 1)  # (T, B, F_down*C_out)
        y = y.reshape(T, B, self.F_down, self.C).transpose(2,
                                                           3)  # (T, B, C_out, F_down)  TODO: ou bien y.reshape(T, B, self.C, self.F_down) ? en fction de si la 2eme dim de la shape est C_out*F_down ou F_down*C_out ?
        y = self.activation(y)  # (T, B, C_out, F_down)
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)  # (B, T, C_out*F_down)

        z, _ = self.gru(y)  # (B, T, H)
        w = self.fc(z)  # (B, T, 1)
        return w.squeeze(-1)  # (B, T)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def detach(self):
        pass
