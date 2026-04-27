import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from deepSTRF.models.neural_model import NeuralModel


class VideoEncodingModel(NeuralModel):
    """
    General mother class for ENCODING models of VIDEO sensory neural responses.

    The forward() method takes as input a single-channel spectrogram of shape (B, C, H, W, T)

    TODO: prefiltering_dict example

    """
    def __init__(self, spatial_resol, temporal_window_size: int, out_neurons: int = 1, output_activation: nn.Module = nn.Identity(), *args, **kwargs):
        super().__init__(out_neurons, output_activation, *args, **kwargs)

        # general attributes for VIDEO neural response models
        self.H, self.W = spatial_resol
        self.T = temporal_window_size

    def validate(self):
        super().validate()
        assert isinstance(self.H, int) and self.H > 0, \
            f"self.H must be a positive int (got {self.H!r})"
        assert isinstance(self.W, int) and self.W > 0, \
            f"self.W must be a positive int (got {self.W!r})"
        assert isinstance(self.T, int) and self.T > 0, \
            f"self.T must be a positive int (got {self.T!r})"

    def STRF_gradmap(self, T=None):
        """
            Get the SPATIO-Temporal Receptive Field (STRF) of the OUTPUT neurons, with a history of T timesteps, as
             the changes in the stimulus that elicit an increase in output activity.

            cf. Rançon et al. (2025), "Temporal recurrence as a general mechanism to explain neural responses in
                the auditory system", BioRxiv

            Returns a (N, C, H, W, T) tensor

            TODO:
             - handle multiple input channels (on & off) because of adaptrans ?
             - allow custom losses ? (e.g. sustained activity rather than last spike ?)
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
