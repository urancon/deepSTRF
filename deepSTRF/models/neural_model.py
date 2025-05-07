import torch
import torch.nn as nn
import torch.functional as F


class NeuralModel(nn.Module):
    """
    General mother class for datasets of sensory neural responses

    All daughter model class should output time-series of shape (B, R=1, T, N)
    See README_models.md in the docs/ folder


    TODO: description

    """

    def __init__(self, out_neurons: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # general attributes for neural response model
        self.O = out_neurons

    def forward(self, stimulus):
        """Takes a sensory stimulus as input and output a tensor of population neural response"""
        raise NotImplementedError

    def count_trainable_params(self):
        """Returns the total number of trainable parameters within the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
