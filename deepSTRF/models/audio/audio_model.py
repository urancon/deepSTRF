import torch
from torch.nn.parameter import Parameter

from deepSTRF.models.neural_model import NeuralModel
from deepSTRF.models.prefiltering import get_CFs, freq_to_tau, tau_to_a, AdapTrans, Willmore_Adaptation


class AudioNeuralModel(NeuralModel):
    """
    General mother class for datasets of AUDIO sensory neural responses

    TODO: description

    TODO: is it the best way (i.e., syntax) to integrate prefiltering schemes ?
    The constructor prepares the instanciation of the cochleagram prefiltering block, which comes before the
    computational backbone.
    prefiltering_dict = {'prefiltering': 'AdapTrans', 'dt': 1.0', 'min_freq': 500, 'max_freq': 20000, 'scale': 'mel'}
    prefiltering: None (default), 'adaptrans' (recommended), 'willmore'



    The forward() method:
     - takes as input a single-channel spectrogram of shape (B, C=1, F, T)
     - outputs a population activity over time of shape (B, N, R=1, T)

    """

    def __init__(self, n_frequency_bands, temporal_window_size, out_neurons: int = 1, prefiltering: dict = None, *args, **kwargs):
        super().__init__(out_neurons, *args, **kwargs)

        # general attributes for AUDIO response models
        self.F = n_frequency_bands
        self.T = temporal_window_size

        # prefiltering: None / AdapTrans / Willmore
        if prefiltering is None:
            self.prefiltering = False
            self.C_in = 1
        else:
            assert isinstance(prefiltering, dict) and 'type' in prefiltering.keys(), "Unvalid format for 'prefiltering'argument. Expected dict with 'type' key."
            prefiltering_type = prefiltering['type']

            if prefiltering_type.lower() == 'adaptrans':
                self.prefiltering = True
                self.dt = prefiltering['dt']  # 5.0 [ms]
                self.fmin = prefiltering['min_freq']  # 500 [Hz]
                self.fmax = prefiltering['max_freq']  # 20,000 [Hz]
                self.CF_scale = prefiltering['scale']  # 'mel'
                cf = get_CFs(self.fmin, self.fmax, self.F, self.CF_scale)
                tau = freq_to_tau(cf)
                a = tau_to_a(tau, dt=self.dt)
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
                a = tau_to_a(tau, dt=self.dt)
                K = round(3 * max(tau).item()) + 1
                self.prefiltering_block = Willmore_Adaptation(init_a_vals=a, kernel_size=K)
                self.C_in = 2

            else:
                raise NotImplementedError(
                    f"Unknown prefiltering {prefiltering_type}. Currently supported spectrogram prefiltering are 'adaptrans' and 'willmore'.")

    def STRF_gradmap(self, T=None):
        """
            Get the Spectro-Temporal Receptive Field (STRF) of the OUTPUT neurons, with a history of T timesteps, as
             the changes in the stimulus that elicit an increase in output activity.

            cf. Rançon et al. (2025), "Temporal recurrence as a general mechanism to explain neural responses in
                the auditory system", BioRxiv

            Returns a (N, 1, F, T) tensor

            # TODO: handle multiple input channels (on & off) because of adaptrans ?
            # TODO: allow custom losses ? (e.g. population ? sustained activity rather than last spike ?)
        """
        B = self.O      # use the batch dimension to parallelize

        # initial stim = null stimulus = absence of bias / absence of information / no entropy
        if T is not None:
            stim_opt = Parameter(torch.zeros(B, 1, self.F, T), requires_grad=True)
        else:
            stim_opt = Parameter(torch.zeros(B, 1, self.F, self.T), requires_grad=True)

        # forward pass
        response = self.forward(stim_opt)  # (B=N, N, T)

        # Spike-Triggered Average (STA) loss = activation at the last timestep
        loss = - response[:, :, -1].mean()     # (B=N,) --> scalar

        # backward pass
        loss.backward()

        # STRF / gradmap = gradient of this loss w.r.t. this null input
        strf_gradmap = stim_opt.grad

        return strf_gradmap
