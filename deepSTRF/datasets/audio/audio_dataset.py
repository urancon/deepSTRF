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
        """Return the number of frequency bins in the spectrograms.

        Returns
        -------
        int
            ``self.F``, the spectrogram frequency-band count.
        """
        return self.F

    def _concat_check_compat(self, other):
        super()._concat_check_compat(other)
        assert isinstance(other, AudioNeuralDataset), \
            f"Cannot concatenate AudioNeuralDataset with {type(other).__name__}"
        assert self.F == other.F, \
            f"F mismatch: {self.F} vs {other.F}. Re-instantiate with matching n_mels."

    def _concat_copy_attrs(self, source):
        super()._concat_copy_attrs(source)
        self.F = source.F

    def validate(self):
        super().validate()
        assert isinstance(self.F, int) and self.F > 0, \
            f"self.F must be a positive int (got {self.F!r})"
