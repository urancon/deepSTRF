from deepSTRF.datasets.neural_dataset import NeuralDataset


class VideoNeuralDataset(NeuralDataset):
    """
    Neural dataset class for video stimuli

    Stimuli are in the form of (C, H, W, T) tensors
    TODO: description

    """

    def __init__(self, path: str):
        super().__init__(path)
        self.HW = (-1, -1)

    def get_HW(self):
        """ Returns the number of frequency bins in the spectrograms. """
        return self.HW