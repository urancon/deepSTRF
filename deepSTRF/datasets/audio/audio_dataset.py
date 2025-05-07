from deepSTRF.datasets.neural_dataset import NeuralDataset


class AudioNeuralDataset(NeuralDataset):
    """
    Neural dataset class for auditory stimuli

    Stimuli are in the form of (F, T) tensors
    TODO: description

    """

    def __init__(self, path: str):
        super().__init__(path)
        self.F = -1

    def get_F(self):
        """ Returns the number of frequency bins in the spectrograms. """
        return self.F
