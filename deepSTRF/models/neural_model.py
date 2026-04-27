from abc import ABC

import torch.nn as nn


class NeuralModel(nn.Module, ABC):
    """
    Base class for encoding models of sensory neural responses (audio,
    video, ...).

    A four-slot template defines the canonical forward pipeline:

        forward(x):
            x = self.wav2spec(x)        # raw-waveform front-end (future)
            x = self.prefiltering(x)    # AdapTrans / ICAdaptation / Identity
            f = self.core(x)            # shared feature backbone
            return self.readout(f)      # per-neuron projection (B, N, 1, T)

    Concrete subclasses populate the slots in their ``__init__``. Default
    values for ``wav2spec``, ``prefiltering``, ``core`` are ``nn.Identity``,
    so a minimal model only needs to provide a ``readout``. Subclasses
    may override ``forward`` for architectures that don't fit the
    four-slot pipeline (e.g. StateNet's recurrent reshape, Transformer's
    per-frame attention).

    See ``docs/_source/md/model_paradigm.md`` for the full contract.
    """

    def __init__(self, out_neurons: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # number of output neurons N (used by validate() and STRF_gradmap)
        self.O = out_neurons

        # canonical pipeline slots — defaults are no-ops; subclasses override
        self.wav2spec = nn.Identity()
        self.prefiltering = nn.Identity()
        self.core = nn.Identity()
        # readout has no sensible default; subclasses must set it before
        # forward() is called. validate() enforces this.

    def forward(self, stimulus):
        """Default template forward: wav2spec → prefiltering → core → readout."""
        x = self.wav2spec(stimulus)
        x = self.prefiltering(x)
        f = self.core(x)
        return self.readout(f)

    def detach(self):
        """Detach stateful variables and parameters from the computational graph (cf. spikingjelly)."""
        pass

    def count_trainable_params(self):
        """Return the total number of trainable parameters within the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def validate(self):
        """
        Check that the instance is deepSTRF-compatible.

        Subclasses should call ``super().validate()`` and then add their own
        checks (e.g. :class:`AudioEncodingModel` checks ``F, T > 0``).
        """
        assert isinstance(self.O, int) and self.O > 0, \
            f"self.O must be a positive int (got {self.O!r})"
        assert hasattr(self, 'readout') and isinstance(self.readout, nn.Module), \
            f"{type(self).__name__}.readout must be set to an nn.Module before validate()"
        for slot in ('wav2spec', 'prefiltering', 'core'):
            assert isinstance(getattr(self, slot), nn.Module), \
                f"self.{slot} must be an nn.Module (got {type(getattr(self, slot)).__name__})"
