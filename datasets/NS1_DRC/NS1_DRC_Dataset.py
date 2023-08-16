import torch
from torch.utils.data.dataset import Dataset

class NS1Dataset(Dataset):
    """
    A PyTorch dataset for handling neural data from the NS1 Dataset.
    See original paper for details:
    - "Network receptive field modeling reveals extensive integration and multi-feature selectivity in auditory cortical neurons" by Nicol S. Harper et al. (2016)

    Dataset Details:
    Stimuli: 20 clips of natural sounds, averaged over 20 repeats.
    Neurons: 73 single_unit auditory neurons
    """

    def __init__(self, path: str, composition='ns1', neuron_indexes=range(73)):
        """
        Initializes the NS1Dataset.

        Parameters:
            path (str): Path to the data file.
            composition (str): Composition of the dataset, can be 'ns1' for the main dataset, 'drc' for some other composition, or 'all' for all compositions.
            neuron_indexes (iterable): Indexes of neurons to include in the dataset.

        Note:
        For neuron_indexes, the dataset includes 73 valid neurons by default. Neuron indexes are used to select specific neurons from the dataset.
        """
        data = torch.load(path)
        if composition == 'ns1':
            self.n = len(data['responses'][0:20])
            self.spectrograms = data['spectrograms'][0:20]               # Shape: (n_samples, 1, F, T) = (20, 1, 34, 999)
            self.responses = data['responses'][neuron_indexes][0:20]     # Shape: (n_neurons,n_samples, T) = (73, 20, 999)
            self.ccmaxes = data['ccmax'][neuron_indexes][0:20]           # Shape: (n_neurons,n_samples) = (73, 20)
            self.response = None
            self.ccmax = None
        if composition == 'drc':
            raise NotImplementedError
        if composition == 'all':
            raise NotImplementedError

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.spectrograms)

    def __getitem__(self, sound_index):
        """Retrieves a single sample from the dataset."""
        spectro = self.spectrograms[sound_index]
        response = self.response[sound_index]
        ccmax = self.ccmax[sound_index]
        return spectro, response, ccmax

    def get_F(self):
        """
        Returns the number of frequency bins in the spectrograms.
        """
        return self.spectrograms.shape[2]

    def get_T(self):
        """
        Returns the number of time bins in the spectrograms.
        """
        return self.spectrograms.shape[3]

    def set_actual_neuron(self, neuron_index):
        """
        Sets the currently selected neuron's response and CCmax.
        """
        self.response = self.responses[neuron_index]
        self.ccmax = self.ccmaxes[neuron_index]
