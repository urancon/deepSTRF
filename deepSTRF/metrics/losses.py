import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class NegativePoissonLogLikelihood(nn.Module):
    """
    Negative Poisson Log-Likelihood loss.

    See for instance:
      -  Singer et al. (2023) Hierarchical temporal prediction captures motion processing along the visual pathway.
        eLife, https://doi.org/10.7554/eLife.52599.
      -  Wang et al. (2025) Foundation model of neural activity predicts response to new stimulus types.
        Nature, https://www.nature.com/articles/s41586-025-08829-y.

    """
    def __init__(self, reduction='mean'):
        super(NegativePoissonLogLikelihood, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, psth):
        # both input tensors should be positive and of shape (B, N, T)
        assert not (prediction < 0.).any() and not (psth < 0.).any(), "detected non-positive elements in predicted tensor"
        assert prediction.shape == psth.shape, f"shapes of prediction {prediction.shape} and psth {psth.shape} did not match"

        L = - torch.mean((psth * torch.log(-prediction + 1e-9) - prediction), dim=(-2, -1))
        if self.reduction == "mean":
            L = L.mean(0)
        elif self.reduction == "sum":
            L = L.sum()
        elif self.reduction == 'None':
            pass
        else:
            raise NotImplementedError

        return L
