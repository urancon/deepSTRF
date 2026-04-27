from abc import ABC, abstractmethod

import torch.nn as nn


class NeuralModel(nn.Module, ABC):
    """General mother class for ENCODING models of sensory (audio, video) neural responses.

    The ``forward()`` method:
     - takes as input a batched stimulus tensor of various shape ``(B, *)`` depending on the modality
     - outputs a population activity over time of shape ``(B, N, R=1, T)``

    Features an output activation function (linear or nonlinear, with or without
    learnable parameters per neuron).

    Subclasses must implement ``forward``. They should also override ``validate``
    (calling ``super().validate()``) to add modality-specific invariants.
    """

    def __init__(self, out_neurons: int = 1, output_activation: nn.Module = nn.Identity(), *args, **kwargs):
        super().__init__(*args, **kwargs)

        # general attributes for neural response model
        self.O = out_neurons
        self.output_activation = output_activation

    @abstractmethod
    def forward(self, stimulus):
        """Take a sensory stimulus as input and output a tensor of population neural response."""
        raise NotImplementedError

    def detach(self):
        """Detach stateful variables and parameters from the computational graph (cf. spikingjelly)."""
        pass

    def count_trainable_params(self):
        """Return the total number of trainable parameters within the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def validate(self):
        """Check that the instance is deepSTRF-compatible.

        Subclasses should call ``super().validate()`` and then add their own checks
        (e.g. ``AudioEncodingModel`` checks ``F, T > 0``).
        """
        assert isinstance(self.O, int) and self.O > 0, \
            f"self.O must be a positive int (got {self.O!r})"
        assert isinstance(self.output_activation, nn.Module), \
            f"self.output_activation must be an nn.Module (got {type(self.output_activation).__name__})"
