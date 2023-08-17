import torch
from torch.utils.data.dataset import Dataset

class NAT4Dataset(Dataset):
    """
        A PyTorch dataset for handling neural data from the A1 & PEG Dataset.
        See original papers for details:
        - "Can deep learning provide a generalizable model for dynamic sound encoding in auditory cortex?" by Jacob R. Pennington et al. (2022)
        - "A convolutional neural network provides a generalizable model of natural sound coding by neural populations in auditory cortex" by Jacob R. Pennington et al. (2023)

        Dataset Details:
        - Stimuli: 20 repetitions of 18 sounds + 1 repetition of 577 sounds (1.5s each)
        - Neurons: Total 849 (A1), 398 (PEG), Valid 777 (A1), 339 (PEG) auditory neurons
        """
    def __init__(self, path: str, composition='val', neuron_indexes=range(100)):
        """
        Initializes the NAT4Dataset.

        Parameters:
            path (str): Path to the data file.
            composition (str): Composition of the dataset, can be 'val' for validation dataset or 'est' for estimation dataset.
            neuron_indexes (iterable): Indexes of neurons to include in the dataset, or 'auditory' for auditory neurons.

        Note:
        If neuron_indexes is set to 'auditory', only neurons marked as auditory in the study will be included.
        """

        data = torch.load(path)
        if composition == 'val':
            self.n = len(data['val_responses'])
            self.spectrograms = data['val_spectrograms']  # Shape: (n_samples, 1, F, T) = (18, 1, 34, X)
            if neuron_indexes == "auditory":
                self.responses = data['val_responses'][data['auditory']]
                self.ccmaxes = data['ccmax'][data['auditory']]
            else:
                self.responses = data['val_responses'][neuron_indexes]  # Shape: (n_neurons,n_samples, T) = (849, 18, X)
                self.ccmaxes = data['ccmax'][neuron_indexes]
            self.response = None
            self.ccmax = None
        if composition == 'est':
            self.n = len(data['est_responses'])
            self.spectrograms = data['est_spectrograms']  # Shape: (n_samples, 1, F, T) = (575, 1, 34, X)
            if neuron_indexes == "auditory":
                self.responses = data['est_responses'][data['auditory']]
                self.ccmaxes = data['ccmax'][data['auditory']]
            else:
                self.responses = data['est_responses'][neuron_indexes]  # Shape: (n_neurons,n_samples, T) = (849, 18, X)
                self.ccmaxes = data['ccmax'][neuron_indexes]
            self.response = None
            self.ccmax = None

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

    def set_actual_neuron(self,neuron_index):
        """Sets the currently selected neuron's response and CCmax.
        """
        self.response = self.responses[neuron_index]
        self.ccmax = self.ccmaxes[neuron_index]