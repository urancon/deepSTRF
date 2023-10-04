import os
import torch
from torch.utils.data.dataset import Dataset


NAT4_A1_ALL_NEURONS = 849
NAT4_A1_AUDITORY_NEURONS = 777
NAT4_PEG_ALL_NEURONS = 398
NAT4_PEG_AUDITORY_NEURONS = 339


class NAT4Dataset(Dataset):
    """
        A PyTorch dataset for handling neural data from the A1 & PEG Dataset.
        See original papers for details:
        - "Can deep learning provide a generalizable model for dynamic sound encoding in auditory cortex?" by Jacob R. Pennington et al. (2022)
        - "A convolutional neural network provides a generalizable model of natural sound coding by neural populations in auditory cortex" by Jacob R. Pennington et al. (2023)


          ============= STRUCTURE ==============

        data is contained in the class attributes 'self.spectrograms' and 'self.responses', which have the following
        shapes:
            spectrograms:   (N_sounds, 1, N_bands, N_timebins)
            responses:      (N_neurons, N_sounds, N_repeats, N_timebins)


        Dataset Details:
        - Stimuli:
            1.5s each
            20 repetitions of 18 sounds (so-called 'validation set')
            1 repetition of 577 sounds (so-called 'estimation set')
        - Neurons: Total 849 (A1), 398 (PEG) of which 777 (A1), 339 (PEG) are valid auditory neurons
        """
    def __init__(self, path: str, area='A1', set='val', neuron_indexes="auditory"):
        """
        Initializes the NAT4Dataset.

        Parameters:
            path (str): Path to the folder containing the datafiles.
            area (str): the cortical area of the recordings, 'A1' or 'PEG'
            set (str): set of the dataset, can be 'val' for validation dataset or 'est' for estimation dataset.
            neuron_indexes (iterable): Indexes of neurons to include in the dataset, or 'auditory' for auditory neurons, or simply 'all' for all neurons

        """
        assert area == "A1" or area == "PEG", f"Unexpected value '{area}' for argument 'area', choose between 'A1' or 'PEG'"
        assert set == "est" or set == "val", f"Unexpected value '{set}' for argument 'set', choose between 'est' or 'val'."

        self.area = area
        self.set = set
        self.neuron_indexes = neuron_indexes

        # select brain area and load the data
        datafile_name = f'nat4_{area.lower()}.pt'
        datafile_path = os.path.join(path, datafile_name)
        data = torch.load(datafile_path)

        # select neurons
        if neuron_indexes == "all":
            neuron_indexes = list(range(len(data[f'{set}_responses'])))
        elif neuron_indexes == "auditory":
            neuron_indexes = data['auditory']
        elif neuron_indexes == "non_auditory":
            raise NotImplementedError  # TODO: allow to choose non-auditory neurons !
        else:
            pass
        self.N_neurons = len(neuron_indexes)

        # select the proper set of the data
        self.spectrograms = data[f'{set}_spectrograms']             # (N_sounds, 1, N_bands, N_timebins)
        self.responses = data[f'{set}_responses'][neuron_indexes]   # (N_neurons, N_sounds, N_repeats, N_timebins)
        self.ccmaxes = data[f'{set}_ccmaxes'][neuron_indexes]       # (N_neurons, N_sounds)
        self.ttrcs = data[f'{set}_ttrcs'][neuron_indexes]           # (N_neurons, N_sounds)

        # let us get rid of stimuli eliciting zero response/spike.
        # for each neuron, we will get the indexes of the stimuli that actually elicit nonzero responses
        # the output will be a list of N_neurons tensors of N_valid_sounds sound indexes (different for each neuron)
        self.valid_data = []
        if set == 'est':
            # estimation sounds only have 1 repeat, so filtering out null neural responses is straightforward
            for nrn_idx in range(self.N_neurons):
                responses = self.responses[nrn_idx].sum(dim=(1, 2))     # (N_sounds,) because for each sound sum spikes over trials and timebins
                valid_indexes = responses.nonzero().squeeze(1)          # (N_valid_sounds < N_sounds)
                self.valid_data.append(valid_indexes)
        else:
            # validation sounds have multiple repeats: if not enough spikes in the response, the ccmax is NaN
            for nrn_idx in range(self.N_neurons):
                valid_indexes = (~self.ccmaxes[nrn_idx].isnan()).nonzero().squeeze(1)
                self.valid_data.append(valid_indexes)

        self.I = 0  # select neuron #0 by default

        print("dataset loaded !")

    def __len__(self):
        """Returns the number of samples (sound stimuli) in the dataset."""
        return len(self.valid_data[self.I])

    def __getitem__(self, sound_index):
        """Retrieves a single sample from the dataset."""
        valid_stim_index = self.valid_data[self.I][sound_index]
        spectro = self.spectrograms[valid_stim_index, :, :, :]       # (1, F, T)
        responses = self.responses[self.I, valid_stim_index, :, :]   # (R, T)
        ccmax = self.ccmaxes[self.I, valid_stim_index]
        ttrc = self.ttrcs[self.I, valid_stim_index]
        return spectro, responses, ccmax, ttrc

    def select_neuron(self, neuron_index):
        assert (neuron_index >= 0) and (neuron_index < self.N_neurons), "neuron_index must be positive and < to the # neurons"
        self.I = neuron_index

    def get_F(self):
        """Returns the number of frequency bins in the spectrograms."""
        return self.spectrograms.shape[-2]

    def get_T(self):
        """Returns the number of time bins in the spectrograms."""
        return self.spectrograms.shape[-1]

    def get_N(self):
        """Returns the number of neurons"""
        return self.N_neurons
