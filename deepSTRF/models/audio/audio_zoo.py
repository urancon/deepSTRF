import torch
import torch.nn as nn
import torch.nn.functional

from .audio_model import AudioNeuralModel

import deepSTRF.models.layers as layers
from deepSTRF.models.dependencies.s4 import S4Block
from deepSTRF.models.dependencies.lmu import LMU
from deepSTRF.models.dependencies.mamba import MambaBlock, MambaConfig
from deepSTRF.models.prefiltering import AdapTrans


# TODO: for all models but L, allow to choose the output nonlinearity, e.g. a 4-parameter sigmoid ?

# TODO: j'ai l'impression qu'avec la parametrization DCLS les gaussiennes sont initialisees dans la partie superieure
#  de la fenetre spectro-temporelle ! Verifier que tout va bien.


class Linear(AudioNeuralModel):
    """
    The canonical, unregularized, unparametrized Linear (L) model

    TODO: implement other types of parametrization, e.g. separable kernels

    """
    def __init__(self, n_frequency_bands=34, temporal_window_size: int = 9, out_neurons: int = 1, prefiltering=None, parameterization=None):
        super(Linear, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

        if parameterization is None:
            self.parameterization = False
            self.conv = nn.Conv2d(self.C_in, self.O, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2))
        else:
            assert isinstance(parameterization, dict) and 'type' in parameterization.keys(), "Unvalid format for 'parametrization'argument. Expected dict with 'type' key."
            self.parameterization = True
            parameterization_type = parameterization['type']
            if parameterization_type == 'DCLS':
                self.num_gaussians = parameterization['num_gauss']
                self.conv = layers.ParametricSTRF(self.F, self.T, self.C_in, self.O, self.num_gaussians)
            else:
                raise NotImplementedError(f"Unknown parameterization {parameterization_type}. Currently supported STRF parameterizations are 'DCLS'.")

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x
        y = self.conv(y)
        return y

    def STRFs(self, polarity='ON'):

        # check argument validity
        if polarity in ['ON', 'On', 'on', 0]:
            polarity_idx = 0
        elif polarity in ['OFF', 'Off', 'off', 1]:
            polarity_idx = 1
        else:
            raise ValueError("argument 'polarity' must be either 'ON', 'On', 'on', 'OFF', 'Off' or 'off'")

        # construct STRF depending on parametrization
        if not self.parameterization:
            strf = self.conv.weight.data.cpu()
        else:
            strf = self.conv.build_kernel()

        # choose polarity between None/ON/OFF
        if isinstance(self.prefiltering_block, AdapTrans):
            strf = strf[0, polarity_idx, :, :]
        else:
            strf = strf[0, 0, :, :]

        return strf.detach()


class LinearNonlinear(AudioNeuralModel):
    """
    A Linear model, but with a nonlinear activation functionat its output.

    """
    def __init__(self, n_frequency_bands=34, temporal_window_size: int = 9, out_neurons: int = 1, prefiltering=None, parameterization=None):
        super(LinearNonlinear, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

        if parameterization is None:
            self.parameterization = False
            self.conv = nn.Conv2d(self.C_in, self.O, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2))
        else:
            assert isinstance(parameterization, dict) and 'type' in parameterization.keys(), "Unvalid format for 'parametrization'argument. Expected dict with 'type' key."
            self.parameterization = True
            parameterization_type = parameterization['type']
            if parameterization_type == 'DCLS':
                self.num_gaussians = parameterization['num_gauss']
                self.conv = layers.ParametricSTRF(self.F, self.T, self.C_in, self.O, self.num_gaussians)
            else:
                raise NotImplementedError(f"Unknown parameterization {parameterization_type}. Currently supported STRF parameterizations are 'DCLS'.")

        self.activation = nn.Sigmoid()

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x
        y = self.activation(self.conv(y))
        return y

    def STRFs(self, polarity='ON'):

        # check argument validity
        if polarity in ['ON', 'On', 'on', 0]:
            polarity_idx = 0
        elif polarity in ['OFF', 'Off', 'off', 1]:
            polarity_idx = 1
        else:
            raise ValueError("argument 'polarity' must be either 'ON', 'On', 'on', 'OFF', 'Off' or 'off'")

        # construct STRF depending on parametrization
        if not self.parameterization:
            strf = self.conv.weight.data.cpu()
        else:
            strf = self.conv.build_kernel()

        # choose polarity between None/ON/OFF
        if isinstance(self.prefiltering_block, AdapTrans):
            strf = strf[0, polarity_idx, :, :]
        else:
            strf = strf[0, 0, :, :]

        return strf.detach()


class NetworkReceptiveField(AudioNeuralModel):
    """
    The Network Receptive Field (NRF) model, a LN model with a hidden layer comprising multiple units.

        cf. Harper et al. (2016), "Network Receptive Field Modeling Reveals Extensive Integration and Multi-feature
            Selectivity in Auditory Cortical Neurons", Plos Comp. Biol., https://doi.org/10.1371/journal.pcbi.1005113

    Contrarily to the original paper, we can also parameterize the filters even in this model !

    """
    def __init__(self, n_frequency_bands=34, temporal_window_size: int = 9, n_hidden: int = 20, out_neurons: int = 1, prefiltering=None, parameterization=None):
        super(NetworkReceptiveField, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

        self.H = n_hidden

        if parameterization is None:
            self.parameterization = False
            self.convs = nn.Sequential(
                nn.Conv2d(self.C_in, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
                nn.BatchNorm2d(self.H),
                nn.Sigmoid(),
                nn.Conv2d(self.H, self.O, kernel_size=1, stride=1),
                nn.Sigmoid(),
            )

        else:
            assert isinstance(parameterization, dict) and 'type' in parameterization.keys(), "Unvalid format for 'parametrization'argument. Expected dict with 'type' key."
            self.parameterization = True
            parameterization_type = parameterization['type']
            if parameterization_type == 'DCLS':
                self.num_gaussians = parameterization['num_gauss']
                self.convs = self.convs = nn.Sequential(
                    layers.ParametricSTRF(self.F, self.T, self.C_in, self.H, self.num_gaussians),
                    nn.BatchNorm2d(self.H),
                    nn.Sigmoid(),
                    nn.Conv2d(self.H, self.O, kernel_size=1, stride=1),
                )
            else:
                raise NotImplementedError(f"Unknown parameterization {parameterization_type}. Currently supported STRF parameterizations are 'DCLS'.")

        self.activation = nn.Sigmoid()

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x
        y = self.activation(self.convs(y))
        return y

    def STRFs(self, hidden_idx=0, polarity='ON'):

        # check argument validity
        if polarity in ['ON', 'On', 'on', 0]:
            polarity_idx = 0
        elif polarity in ['OFF', 'Off', 'off', 1]:
            polarity_idx = 1
        else:
            raise ValueError("argument 'polarity' must be either 'ON', 'On', 'on', 'OFF', 'Off' or 'off'")

        # construct STRF depending on parametrization
        if not self.parameterization:
            strf = self.convs[0].weight.data.cpu()
        else:
            strf = self.convs[0].build_kernel()

        # choose polarity between None/ON/OFF
        if isinstance(self.prefiltering_block, AdapTrans):
            strf = strf[hidden_idx, polarity_idx, :, :]
        else:
            strf = strf[hidden_idx, 0, :, :]

        return strf.detach()


class DNet(AudioNeuralModel):
    """
    The Dynamic Network (DNet) model, basically a NRF model in which hidden and output units are stateful and leaky.

        cf. Rahman et al. (2019), "A dynamic network model of temporal receptive fields in primary auditory cortex",
            Plos Comp. Biol., https://doi.org/10.1371/journal.pcbi.1006618

    """
    def __init__(self, n_frequency_bands=34, temporal_window_size: int = 9, n_hidden: int = 20, init_tau=2., decay_input=True, out_neurons: int = 1, prefiltering=None, parameterization=None):
        super(DNet, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

        self.H = n_hidden

        if parameterization is None:
            self.parameterization = False
            self.convs = nn.Sequential(
                nn.Conv2d(self.C_in, self.H, kernel_size=(self.F, self.T), stride=1, padding=(0, (self.T - 1) // 2)),
                nn.BatchNorm2d(self.H),
                nn.Sigmoid(),
                layers.LearnableExponentialDecay(self.H, kernel_size=round(init_tau * 7), init_tau=init_tau, decay_input=decay_input),
                nn.Conv2d(self.H, self.O, kernel_size=1, stride=1),
                nn.Sigmoid(),
                layers.LearnableExponentialDecay(1, kernel_size=round(init_tau * 7), init_tau=init_tau, decay_input=decay_input)
            )

        else:
            assert isinstance(parameterization, dict) and 'type' in parameterization.keys(), "Unvalid format for 'parametrization'argument. Expected dict with 'type' key."
            self.parameterization = True
            parameterization_type = parameterization['type']
            if parameterization_type == 'DCLS':
                self.num_gaussians = parameterization['num_gauss']
                self.convs = self.convs = nn.Sequential(
                    layers.ParametricSTRF(self.F, self.T, self.C_in, self.H, self.num_gaussians),
                    nn.BatchNorm2d(self.H),
                    nn.Sigmoid(),
                    layers.LearnableExponentialDecay(self.H, kernel_size=round(init_tau * 7), init_tau=init_tau, decay_input=decay_input),  # TODO: why is kernel_size = 7 * init_tau ???
                    nn.Conv2d(self.H, self.O, kernel_size=1, stride=1),
                    nn.Sigmoid(),
                    layers.LearnableExponentialDecay(1, kernel_size=round(init_tau * 7), init_tau=init_tau, decay_input=decay_input)
                )
            else:
                raise NotImplementedError(f"Unknown parameterization {parameterization_type}. Currently supported STRF parameterizations are 'DCLS'.")

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x
        y = self.convs(y)
        return y

    def STRFs(self, hidden_idx=0, polarity='ON'):

        # check argument validity
        if polarity in ['ON', 'On', 'on', 0]:
            polarity_idx = 0
        elif polarity in ['OFF', 'Off', 'off', 1]:
            polarity_idx = 1
        else:
            raise ValueError("argument 'polarity' must be either 'ON', 'On', 'on', 'OFF', 'Off' or 'off'")

        # construct STRF depending on parametrization
        if not self.parameterization:
            strf = self.convs[0].weight.data.cpu()
        else:
            strf = self.convs[0].build_kernel()

        # choose polarity between None/ON/OFF
        if isinstance(self.prefiltering_block, AdapTrans):
            strf = strf[hidden_idx, polarity_idx, :, :]
        else:
            strf = strf[hidden_idx, 0, :, :]

        return strf.detach()


class ConvNet2D(AudioNeuralModel):
    """
    Adapted from, but not entirely equivalent to the so-called '2D-CNN' of Pennington et al. (2023),
        "A convolutional neural network provides a generalizable model of natural sound coding by neural populations
        in auditory cortex", PLOS CB

    Major differences:
     - BatchNorm inside the convolutional backbone
     - Zero padding along frequency dimension --> downsampling along this dimension after self.convs

    """
    def __init__(self, n_frequency_bands=34, kernel_size: tuple = (3, 9), c_hidden: int = 10, n_hidden: int = 20, out_neurons: int = 1, prefiltering=None):
        temporal_window_size = kernel_size[1]  # TODO: find the formula for RF size
        super(ConvNet2D, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

        # general
        self.K = kernel_size
        self.C = c_hidden
        self.H = n_hidden

        self.convs = nn.Sequential(
            nn.Conv2d(self.C_in, self.C, kernel_size=self.K, stride=1, padding=(0, (self.K[1] - 1) // 2)),
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
            nn.Linear(in_features=self.H, out_features=self.O),
        )

        self.activation = nn.Sigmoid()  # layers.ParametricSigmoid(self.O, False)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x
        y = self.convs(y)                       # (B, C, F_down, T)
        y = y.flatten(start_dim=1, end_dim=2)   # (B, C*F_down, T)
        y = y.permute(0, 2, 1)                  # (B, T, C*F_down)
        y = self.activation(self.fc(y))         # (B, T, N)
        y = y.permute(0, 2, 1).unsqueeze(2)     # (B, N, R=1, T)
        return y


class Transformer(AudioNeuralModel):
    """
    Attention-based, Transformer model

        cf. Rançon et al. (2025), "Temporal recurrence as a general mechanism to explain neural responses in
                the auditory system", BioRxiv

    """

    def __init__(self, n_frequency_bands=34, temporal_window_size=1, token_size=(34, 1), embedding_dim=48, n_heads=1,
                 n_layers=1, out_neurons: int = 1, prefiltering=None):
        super(Transformer, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

        # patch dimensions and strides
        self.K_f, self.K_t = token_size

        # padding left only (causal inference)
        self.pad = nn.ZeroPad2d(((self.T - 1), 0, 0, 0))

        # im2col --> create context windows for attention/transformer layers
        self.unfold_context = nn.Unfold(kernel_size=(self.F, self.T), stride=(1, 1), padding=(0, 0))

        # Patchify with learnable embdeddings
        self.embedding_dim = embedding_dim
        self.conv_patches = nn.Conv2d(self.C_in, self.embedding_dim, kernel_size=(self.K_f, self.K_t), stride=(self.K_f, self.K_t))

        # determine sizes to define the readout layer
        prospective_input = torch.rand(1, self.C_in, self.F, self.T)
        # prospective_patches = self.unfold_patches(prospective_input)
        prospective_patches = self.conv_patches(prospective_input).flatten(-2, -1)
        _, patch_dim, n_patches = prospective_patches.shape
        self.H = patch_dim

        self.n_heads = n_heads
        self.n_layers = n_layers
        self.positional_encoding = torch.nn.Parameter(torch.rand(1, n_patches, patch_dim))
        self.tsfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=patch_dim, nhead=self.n_heads, dim_feedforward=32, dropout=0.1,
                                       batch_first=True),
            num_layers=self.n_layers,
        )

        self.readout = nn.Sequential(
            nn.Linear(patch_dim, self.O),
            layers.DoubleExponential(self.O)
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        B, C, F, L = x.shape
        y = self.prefiltering_block(x) if self.prefiltering else x  # (B, C=1|2, F, L)
        y = self.unfold_context(self.pad(y))  # (B, C*F*T, L)
        y = y.permute(0, 2, 1).flatten(0, 1)  # (B*L, C*F*T)

        y = y.reshape(-1, self.C_in, self.F, self.T)    # (B*L, C, F, T)
        y = self.conv_patches(y)                        # (B*L, C+, F-, T-)
        y = y.flatten(start_dim=-2, end_dim=-1)         # (B*L, patch_dim = C+, n_patches = F- * T-)
        y = y.permute(0, 2, 1)                          # (B*L, n_patches, patch_dim)

        y = self.tsfm(y + self.positional_encoding)  # (B*L, n_patches, patch_dim)

        # global average pooling
        y = y.mean(1, keepdim=True)  # (B*L, 1, patch_dim)

        y = y.flatten(start_dim=1, end_dim=2)   # (B*L, n_patches * patch_dim)
        y = y.unflatten(dim=0, sizes=(B, L))    # B, L, n_patches * patch_dim)
        y = self.readout(y)                     # (B, L, N)
        y = y.permute(0, 2, 1)                  # (B, N, L)
        return y


class StateNet(AudioNeuralModel):
    """
    Fully stateful model. Without delays and only relies on temporal recurrence to implicitly extract information from
    stimulus sequences.

        cf. Rançon et al. (2025), "Temporal recurrence as a general mechanism to explain neural responses in
                the auditory system", BioRxiv

    """
    def __init__(self, n_frequency_bands=34, temporal_window_size=1, kernel_size: int = 7, stride: int = 3,
                 hidden_channels: int = 7, connectivity: str = 'LC', rnn_type: str = 'GRU', out_neurons: int = 1, prefiltering=None):
        super(StateNet, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

        # general
        self.K = kernel_size
        self.S = stride
        self.C = hidden_channels
        self.rnn_type = rnn_type

        # stateless spectral encoder
        if connectivity == 'LC':
            self.encoder_layers = nn.Sequential(
                layers.LocallyConnected1d(input_size=self.F, in_channels=self.C_in, out_channels=self.C, kernel_size=self.K, stride=self.S),
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            )
        elif connectivity == 'FC':
            F_down = int((self.F - kernel_size) / self.S + 1)
            self.encoder_layers = nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),                               # (B, C_in, F) --> (B, C_in * F)
                nn.Linear(self.C_in * self.F, self.C * F_down),                     # (B, C_in * F) --> (B, C * F_down)
                nn.Unflatten(dim=-1, unflattened_size=(self.C, F_down)),            # (B, C, F_down)
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            )
        elif connectivity == 'CONV':
            self.encoder_layers = nn.Sequential(
                nn.Conv1d(self.C_in, self.C, kernel_size=self.K, stride=self.S),    # (B, C, F)
                nn.BatchNorm1d(self.C),
                nn.Sigmoid()
            )

        # get output shape and embedding space dim for GRUs
        self.L = (n_frequency_bands - self.K) // self.S + 1
        self.H = self.L * self.C

        # RNNs for temporal processing
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif (self.rnn_type == 'vanilla') or (self.rnn_type == 'RNN'):
            self.rnn = nn.RNN(input_size=self.H, hidden_size=self.H, num_layers=1, batch_first=True)
        elif self.rnn_type == 'LMU':
            self.rnn = LMU(input_size=self.H, hidden_size=self.H, memory_size=128, theta=99, learn_a=False, learn_b=False)  # ok perfs but slow ! try memory_size=1024 and theta=50
        elif self.rnn_type == "Mamba":
            self.rnn = MambaBlock(MambaConfig(d_model=self.H, n_layers=1))
        elif self.rnn_type == "S4":
            self.rnn = S4Block(d_model=self.H, transposed=False)
        else:
            raise NotImplementedError(f"received unknown rnn_type '{rnn_type}': please choose between: "
                                      f"'GRU' (default), 'LSTM', 'RNN', 'vanilla', 'LMU', 'S4', 'Mamba'")

        # readout from RNNs' hidden state
        self.fc = nn.Linear(self.H, self.O)

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x  # (B, 1|2, F, T)
        y = y.permute(3, 0, 1, 2)                                   # (T, B, 2, F)

        # pass through spectral encoder efficiently
        y_shape = [y.shape[0], y.shape[1]]
        y = y.flatten(0, 1)
        y = self.encoder_layers(y)
        y_shape.extend(y.shape[1:])
        y = y.view(y_shape)                                         # (T, B, C_hidd, F_down)

        # prepare for RNN
        y = y.flatten(start_dim=2, end_dim=3).permute(1, 0, 2)      # (B, T, C_hidd*F_down)

        # RNN
        if self.rnn_type != "Mamba":
            y, _ = self.rnn(y)      # (B, T, H)
        else:
            y = self.rnn(y)         # for mamba

        y = self.fc(y)              # (B, T, N)
        y = y.permute(0, 2, 1)      # (B, N, T)
        return y
