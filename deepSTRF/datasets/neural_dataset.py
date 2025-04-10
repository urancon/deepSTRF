import os
import numpy as np
import pandas as pd
import csv
from scipy.io import wavfile
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data.dataset import Dataset



class NeuralDataset(Dataset):
    """
    General mother class for datasets of sensory neural responses

    TODO: description

    """

    def __init__(self, path: str):
        self.path = path
        self.N_neurons = -1
        self.I = [-1]

    def __len__(self):
        """
        Returns the number of stimulus samples in the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, stim_index):
        """Retrieves a single sample from the dataset."""
        raise NotImplementedError

    def select_neuron(self, neuron_index):
        assert (isinstance(neuron_index, int)) and (neuron_index >= 0) and (neuron_index < self.N_neurons), \
            "neuron_index must be positive and < to the # neurons"
        self.I = [neuron_index]

    def select_population(self, neuron_indices):
        for neuron_index in neuron_indices:
            assert (isinstance(neuron_index, int)) and (neuron_index >= 0) and (neuron_index < self.N_neurons), \
                "neuron_index must be positive and < to the # neurons"
        self.I = list(neuron_indices)

    def get_N(self):
        """ Returns the number of neurons """
        return self.N_neurons

    def get_pop_metadata(self):
        """Retrieve metadata for each currently selected neuron"""
        return [self.nrn_meta[i] for i in self.I]
