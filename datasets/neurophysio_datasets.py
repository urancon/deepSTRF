import copy
import os.path
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

class NS1Dataset(Dataset):
    """
    Dataset object for loading the dataset used in the studies by Rahman et al.
    - Harper et al. (2016), "Network Receptive Field Modeling Reveals Non-linear Characteristics of Auditory Cortical Neurons"
    - Rahman et al. (2019), "A dynamic network model of temporal receptive fields in primary auditory cortex"
    - Rahman et al. (2022), "Simple transformations capture auditory input to cortex"

    Anesthetized ferrets were presented with natural sounds while recording auditory cortex neurons.
    Sounds were encoded into mel spectrogram representations using a mel filterbank and a neural network model of the auditory periphery.
    The network was trained to match the recorded neuron activations.

    This dataset consists of:
    - Input data: Mel spectrograms of the presented sounds (shape: (n_samples, F, T))
    - Targets: Probability of firing of the recorded/output neuron (shape: (n_samples, T))
    """
    def __init__(self, data,neuron_index):
        self.spectrograms = data['spectrograms']            # Shape: (n_samples, 1, F, T)
        self.responses = data['responses'][neuron_index]    # Shape: (n_samples, T)
        self.ccmax = data['ccmax'][neuron_index]

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectro = self.spectrograms[index]
        response = self.responses[index]
        ccmax = self.ccmax[index]
        return spectro, response,ccmax

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]

class NS2Dataset(Dataset):
    """
    Dataset object for loading the dataset used in the studies by M. L. Espejo et al.
    - Espejo et al. (2019), "Spectral tuning of adaptation supports coding of sensory context in auditory cortex"
    - Rahman et al. (2022), "Simple transformations capture auditory input to cortex"

    Anesthetized ferrets were presented with natural sounds while recording auditory cortex neurons.
    Sounds were encoded into mel spectrogram representations using a mel filterbank and a neural network model of the auditory periphery.
    The network was trained to match the recorded neuron activations.

    This dataset consists of:
    - Input data: Mel spectrograms of the presented sounds (shape: (n_samples, F, T))
    - Targets: Probability of firing of the recorded/output neuron (shape: (n_samples, T))
    """
    def __init__(self, spectrograms: torch.Tensor, responses: torch.Tensor):
        self.spectrograms = spectrograms
        self.responses = responses

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectrogram = self.spectrograms[index]
        response = self.responses[index]
        return spectrogram, response

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]

class NAT4Dataset(Dataset):
    """
        Dataset object for loading the dataset used in the studies by J. R. Pennington et al.
        Pennington, J. R., & David, S. V. (2022). "Can deep learning provide a generalizable model for dynamic sound encoding in auditory cortex?"
        Pennington, J. R., & David, S. V. (2023). "A convolutional neural network provides a generalizable model of natural sound coding by neural populations in auditory cortex".

        Anesthetized ferrets were presented with natural sounds while recording auditory cortex neurons.
        Sounds were encoded into mel spectrogram representations using a mel filterbank and a neural network model of the auditory periphery.
        The network was trained to match the recorded neuron activations.

        This dataset consists of:
        - Input data: Mel spectrograms of the presented sounds (shape: (n_samples, F, T))
        - Targets: Probability of firing of the recorded/output neuron (shape: (n_samples, T))
        """
    def __init__(self, filepath):
        data = torch.load(filepath)
        self.spectrograms = data['spectrograms']
        self.responses = data['responses']

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectrogram = self.spectrograms[index]
        response = self.responses[index]
        return spectrogram, response

    def get_F(self):
        return self.spectrograms.shape[2]

    def get_T(self):
        return self.spectrograms.shape[3]
