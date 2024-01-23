import torch
import torch.nn as nn
import torch.nn.functional

import models.layers as layers
from models.prefiltering import get_CFs, freq_to_tau, tau_to_a, AdapTrans, Willmore_Adaptation


#    #################
#       MOTHER CLASS
#    #################
#
# All models inherit from the following class, which has some predefined methods; its constructor prepares the
# instanciation of the cochleagram prefiltering block, which comes before the computational backbone.
#
#

class AudioResponseModel(nn.Module):
    """
    Mother class from which all other auditory response models inherit

    Takes as input a single-channel spectrogram of shape (B, C, F, T)
    The output neuron's response at each timestep is predicted from a past temporal window of the spectrogram.

    prefiltering_dict = {'prefiltering': 'AdapTrans', 'dt': 1.0', 'min_freq': 500, 'max_freq': 20000, 'scale': 'mel'}

    prefiltering: None (default), 'adaptrans' (recommended), 'willmore'

    """
    def __init__(self, n_frequency_bands: int = 64, temporal_window_size: int = 1, out_neurons: int = 1, prefiltering=None):
        super(AudioResponseModel, self).__init__()

        # general
        self.F = n_frequency_bands
        self.T = temporal_window_size
        self.O = out_neurons

        # prefiltering: None / AdapTrans / Willmore
        if prefiltering is None:
            self.prefiltering = False
            self.C_in = 1
        else:
            assert isinstance(prefiltering, dict) and 'type' in prefiltering.keys(), "Unvalid format for 'prefiltering'argument. Expected dict with 'type' key."
            prefiltering_type = prefiltering['type']

            if prefiltering_type.lower() == 'adaptrans':
                self.prefiltering = True
                self.dt = prefiltering['dt']                    # 5.0 [ms]
                self.fmin = prefiltering['min_freq']            # 500 [Hz]
                self.fmax = prefiltering['max_freq']            # 20,000 [Hz]
                self.CF_scale = prefiltering['scale']           # 'mel'
                cf = get_CFs(self.fmin, self.fmax, self.F, self.CF_scale)
                tau = freq_to_tau(cf)
                a = tau_to_a(tau)
                w = torch.ones_like(a) * 0.75
                K = round(3 * max(tau).item()) + 1
                self.prefiltering_block = AdapTrans(init_a_vals=a, init_w_vals=w, kernel_size=K, learnable=True)
                self.C_in = 2

            elif prefiltering_type.lower() == 'willmore':
                self.prefiltering = True
                self.C_in = 1
                self.dt = prefiltering['dt']
                self.fmin = prefiltering['min_freq']
                self.fmax = prefiltering['max_freq']
                self.CF_scale = prefiltering['scale']
                cf = get_CFs(self.fmin, self.fmax, self.F, self.CF_scale)
                tau = freq_to_tau(cf)
                a = tau_to_a(tau)
                K = round(3 * max(tau).item()) + 1
                self.prefiltering_block = Willmore_Adaptation(init_a_vals=a, kernel_size=K)
                self.C_in = 2

            else:
                raise NotImplementedError(f"Unkown prefiltering {prefiltering_type}. Currently supported spectrogram prefiltering are 'adaptrans' and 'willmore'.")

    def forward(self, spectrogram):
        # takes in a spectrogram of.shape (B, 1, F, T)
        # outputs a population activity over time of.shape (B, N, T)
        raise NotImplementedError

    def detach(self):
        # for stateful models that keep their states even after forward()
        pass

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def STRFs(self):
        raise NotImplementedError


#    ###############
#       MODEL ZOO
#    ###############
#
# All models should receive single-channel image-like 2D sound spectrograms. Shape: (B, C=1, F, T)
#  ..    ..    ..   output time-series of shape (B, R=1, T, N)
#
# See README_models.md in the docs/ folder
#
#
#

# TODO: harmonize output to be (B, 1, N, T) for all models
# TODO: for all models but L, allow to choose the output nonlinearity, e.g. a 4-parameter sigmoid ?


class Linear(AudioResponseModel):
    """
    The canonical, unregularized, unparametrized Linear (L) model

    TODO: allow parameterization --> param_dict = {'type': 'DCLS', 'gaussians': 10}
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
        return self.conv(y).squeeze(1).squeeze(1)

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


class LinearNonlinear(AudioResponseModel):
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
        return self.activation(self.conv(y)).squeeze(1).squeeze(1)

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


class NetworkReceptiveField(AudioResponseModel):
    """
    The Network Receptive Field (NRF) model, a LN model with a hidden layer comprising multiple units.

    TODO: ref NRF paper

    Contrarily to the original paper, we can also parameterize the filters even in this model !

    TODO: j'ai l'impression qu'avec la parametrization DCLS les gaussiennes sont initialisees dans la partie superieure
     de la fenetre spectro-temporelle ! Verifier que tout va bien.

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
                    nn.Sigmoid(),
                )
            else:
                raise NotImplementedError(f"Unknown parameterization {parameterization_type}. Currently supported STRF parameterizations are 'DCLS'.")

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x
        return self.convs(y).squeeze(1).squeeze(1)

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


class DNet(AudioResponseModel):
    """
    The Dynamic Network (DNet) model, basically a NRF model in which hidden and output units are stateful and leaky.

    TODO: ref DNet paper

    TODO: j'ai l'impression qu'avec la parametrization DCLS les gaussiennes sont initialisees dans la partie superieure
     de la fenetre spectro-temporelle ! Verifier que tout va bien.

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
        return self.convs(y).squeeze(1).squeeze(1)

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


class ConvNet2D(AudioResponseModel):
    """
    TODO: description

    Adapted from, but not entirely equivalent to the so-called '2D-CNN' of Pennington et al. (2023),
        "A convolutional neural network provides a generalizable model of natural sound coding by neural populations
        in auditory cortex", PLOS CB

    Major differences:
     - BatchNorm inside the convolutional backbone
     - Zero padding along frequency dimension --> downsampling along this dimension after self.convs

    Note:
     - TODO
    """
    def __init__(self, n_frequency_bands=34, kernel_size: tuple = (3, 9), c_hidden: int = 10, n_hidden: int = 20, out_neurons: int = 1, prefiltering=None):
        temporal_window_size = kernel_size[1]  # TODO: find the formula for RF size
        super(ConvNet2D, self).__init__(n_frequency_bands, temporal_window_size, out_neurons, prefiltering)

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
            nn.Linear(in_features=self.H, out_features=self.N_out),
            layers.ParametricSigmoid(self.N_out, False)
        )

    def forward(self, x):
        # x.shape must be (B, 1, F, T)
        y = self.prefiltering_block(x) if self.prefiltering else x
        y = self.convs(y)                       # (B, C, F_down, T)
        y = y.flatten(start_dim=1, end_dim=2)   # (B, C*F_down, T)
        y = y.permute(0, 2, 1)                  # (B, T, C*F_down)
        y = self.fc(y)                          # (B, T, N)
        return y.squeeze(2)

    def STRFs(self, hidden_idx=0, polarity='ON'):
        # TODO: define and implement a STRF extraction method for models with small convs
        raise NotImplementedError
