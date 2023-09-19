import os
import torch
from torch.utils.data.dataset import Dataset


class NS1_DRC_Dataset(Dataset):
    """
    A PyTorch dataset for handling neural data from the NS1 Dataset.
    See original paper for details:
    - "Network receptive field modeling reveals extensive integration and multi-feature selectivity in auditory cortical neurons" by Nicol S. Harper et al. (2016)


    ============= ABOUT ==============

    data acquisition:
    - 73 neurons recorded in A1 in deeply anesthetized ferrets
    - external recordings, spike-sorted individual units  --> spiking responses and PSTHs
    - presentation of natural and synthetic stimuli

    spectrograms:
    - simple amplitude short-term fourier transform
    - spectrograms with a temporal resolution of 5 msec (default)
    - 20 clips of natural sounds, averaged over 20 repeats, + 12 clips of Dynamic Random Chords (DRC) (5 repeats)


    ============= STRUCTURE ==============

    data is contained in the class attribute 'self.data', which contains a list of N_neurons dictionaries  of the
    following structure:
    e.g.
        {"spectrograms": list,      # N_sounds * torch.tensor(1, F, T)
         "responses": list,         # N_sounds * torch.tensor(N_repeats, T)
         "stim_type": list          # str, e.g. 'nat'/'drc'
        }


    ============= SOURCE ==============

    Original data freely available at:
        https://osf.io/ayw2p/

    Original paper:
        B. Willmore, J. W. H. Schnupp, A. J. King (2016), "Network Receptive Field Modeling Reveals Extensive
        Integration and Multi-feature Selectivity in Auditory Cortical Neurons", PLoS CB, 10.1371/journal.pcbi.1005113

    """

    def __init__(self, path: str, stimuli=('nat', 'drc'), neuron_indexes=tuple(range(73))):
        """
        Initializes and loads the (preprocessed) NS1_DRC_Dataset.

        Parameters:
            path (str): Path to the 'data/' folder containing the preprocessed 'ns1_drc_responses.pt and 'ns1_drc_spectrograms.pt'
            stimuli: stimuli of the dataset, can be 'ns1' for the netural sounds, 'drc' for dynamic random chords, or 'all' for both.
            neuron_indexes (iterable): List or array of indexes of neurons to include in the dataset.

        Note:
        For neuron_indexes, the dataset includes 73 valid neurons by default. Neuron indexes are used to select specific
        neurons from the dataset.
        """
        # stimuli are the same for all neurons
        spectrograms = torch.load(os.path.join(path, 'ns1_drc_spectrograms.pt'))
        spectrograms = torch.cat([spectrograms, torch.ones(12, 1, 34, 999) * torch.nan])    # TODO: replace NaN by actual DRC stim
        stim_types = 20 * ['nat'] + 12 * ['drc']
        N_sounds = 32

        # load neural responses (different for each neuron)
        response_data = torch.load(os.path.join(path, 'ns1_drc_responses.pt'))
        self.N_neurons = len(neuron_indexes)
        self.response_data = [response_data[neuron_idx] for neuron_idx in neuron_indexes]   # filter selected neurons
        self.I = 0      # select neuron #0 by default

        # filter by stimulus type
        stim_mask = [True if ((stim_type in stimuli) or (stim_type in stimuli)) else False for stim_type in stim_types]
        self.spectrograms = spectrograms[stim_mask]
        for neuron_idx in range(len(self.response_data)):
            neuron_responses = self.response_data[neuron_idx]['responses']
            neuron_responses = [neuron_responses[sound_idx] for sound_idx in range(N_sounds) if stim_mask[sound_idx]]
            self.response_data[neuron_idx]['responses'] = neuron_responses

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.spectrograms)

    def __getitem__(self, sound_index):
        """Retrieves a single sample from the dataset."""
        spectro = self.spectrograms[sound_index]            # (1, F, T)
        neuron_data = self.response_data[self.I]            # select neuron according to current index
        responses = neuron_data['responses'][sound_index]   # (N_repeats, T)
        return spectro, responses

    def select_neuron(self, neuron_index):
        assert (neuron_index >= 0) and (neuron_index < self.N_neurons), "neuron_index must be positive and < to the # neurons"
        self.I = neuron_index

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

    def get_N(self):
        """
        Returns the number of neurons
        """
        return self.N_neurons
