import torch
from torch.utils.data.dataset import Dataset

class NS1Dataset(Dataset):
    """***WIP***"""

    def __init__(self, path: str, neuron_indexes=range(73)):
        data = torch.load(path)
        self.n = len(data['responses'])
        self.spectrograms = data['spectrograms']                # Shape: (n_samples, 1, F, T) = (20, 1, 34, 999)
        self.responses = data['responses'][neuron_indexes]      # Shape: (n_neurons,n_samples, T) = (73, 20, 999)
        self.ccmaxes = data['ccmax'][neuron_indexes]            # Shape: (n_neurons,n_samples) = (73, 20)
        self.response = None
        self.ccmax = None

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