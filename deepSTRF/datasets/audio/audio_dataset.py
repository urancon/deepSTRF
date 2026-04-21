from deepSTRF.datasets.neural_dataset import NeuralDataset


class AudioNeuralDataset(NeuralDataset):
    """Neural dataset class for auditory stimuli.

    Stimuli are in the form of (1, F, T) tensors.

    Subclasses must additionally set ``self.F`` (number of frequency bins in
    the spectrogram) in their ``__init__``, before calling ``self.validate()``.
    """

    def __init__(self, path: str, dt_ms: float):
        super().__init__(path, dt_ms)
        self.F = -1

    def get_F(self):
        """Return the number of frequency bins in the spectrograms."""
        return self.F

    def validate(self):
        super().validate()
        assert isinstance(self.F, int) and self.F > 0, \
            f"self.F must be a positive int (got {self.F!r})"
