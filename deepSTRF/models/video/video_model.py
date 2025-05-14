import torch
from torch.nn.parameter import Parameter

from deepSTRF.models.neural_model import NeuralModel


class VideoNeuralModel(NeuralModel):
    """
    General mother class for datasets of VIDEO sensory neural responses

    TODO: description

    """

    def __init__(self, spatial_resol, temporal_window_size: int, out_neurons: int = 1, *args, **kwargs):
        super().__init__(out_neurons, *args, **kwargs)

        # general attributes for VIDEO neural response models
        self.HW = spatial_resol
        self.T = temporal_window_size

    def STRF_gradmap(self, T=None):
        """
            Get the Spatio-Temporal Receptive Field (STRF) of the OUTPUT neurons, with a history of T timesteps, as
             the changes in the stimulus that elicit an increase in output activity.

            cf. Rançon et al. (2025), "Temporal recurrence as a general mechanism to explain neural responses in
                the auditory system", BioRxiv

            # TODO: handle multiple input channels (on & off) because of adaptrans ?
            # TODO: allow custom losses ?
        """
        B = self.O      # use the batch dimension to parallelize

        # initial stim = null stimulus = absence of bias / absence of information / no entropy
        if T is not None:
            stim_opt = Parameter(torch.zeros(B, 1, self.H, self.W, T), requires_grad=True)
        else:
            stim_opt = Parameter(torch.zeros(B, 1, self.H, self.W, self.T), requires_grad=True)

        # forward pass
        response = self.forward(stim_opt)  # (B=N, N, T)

        # Spike-Triggered Average (STA) loss = activation at the last timestep
        loss = - torch.trace(response[:, :, -1])     # scalar

        # backward pass
        loss.backward()

        # STRF / gradmap = gradient of this loss w.r.t. this null input
        strf_gradmap = stim_opt.grad

        return strf_gradmap
