import torch
from torch.utils.data.dataset import Dataset

class NAT4Dataset(Dataset):
    """***WIP***"""

    def __init__(self, path: str, composition='val', neuron_indexes=range(73)):
        data = torch.load(path)
        if composition == 'val':
            self.n = len(data['val_responses'])
            self.spectrograms = data['val_spectrograms']            # Shape: (n_samples, 1, F, T) = (18, 1, 34, X)
            self.responses = data['val_responses'][neuron_indexes]  # Shape: (n_neurons,n_samples, T) = (849, 18, X)
            self.ccmaxes = data['ccmax'][neuron_indexes]            # Shape: (n_neurons,n_samples) = (849, 18 )
            self.response = None
            self.ccmax = None
        if composition == 'est':
            self.n = len(data['est_responses'])
            self.spectrograms = data['est_spectrograms']            # Shape: (n_samples, 1, F, T) = (575, 1, 34, X)
            self.responses = data['est_responses'][neuron_indexes]  # Shape: (n_neurons,n_samples, T) = (849, 575, X)
            self.ccmaxes = data['ccmax'][neuron_indexes]            # Shape: (n_neurons,n_samples) = (849, 575)
            self.response = None
            self.ccmax = None
        if composition == 'all':
                a = 1

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, sound_index):
        spectro = self.spectrograms[sound_index]
        response = self.response[sound_index]
        ccmax = self.ccmax[sound_index]
        return spectro, response, ccmax

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]

    def set_actual_neuron(self,neuron_index):
        self.response = self.responses[neuron_index]
        self.ccmax = self.ccmaxes[neuron_index]