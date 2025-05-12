import os
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset


RAHMAN_TRAINVAL_SET_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
RAHMAN_TEST_SET_INDICES = [3, 6, 9, 19]


class NS1_DRC_Dataset(AudioNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the NS1 Dataset.
    See original paper for details:
     "Network receptive field modeling reveals extensive integration and multi-feature selectivity in auditory cortical
        neurons" by Nicol S. Harper et al. (2016)


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
         "ccmaxes": list,           # N_sounds * float
         "ttrcs" : list,            # N_sounds * float
         "stim_type": list          # N_sounds * str, e.g. 'nat'/'drc'
        }


    ============= SOURCE ==============

    Original data freely available at:
        https://osf.io/ayw2p/

    Original paper:
        B. Willmore, J. W. H. Schnupp, A. J. King (2016), "Network Receptive Field Modeling Reveals Extensive
        Integration and Multi-feature Selectivity in Auditory Cortical Neurons", PLoS CB, 10.1371/journal.pcbi.1005113
        """

    def __init__(self, path: str, stimuli=('nat', 'drc'), neuron_indexes=tuple(range(73)), normalize_resps=False):
        super().__init__(path)

        # stimuli are the same for all neurons
        spectrograms = torch.load(os.path.join(path, 'ns1_drc_spectrograms.pt'))
        spectrograms = torch.cat([spectrograms, torch.ones(12, 1, 34, 999) * torch.nan])    # TODO: replace NaN by actual DRC stim
        stim_types = 20 * ['nat'] + 12 * ['drc']
        N_sounds = 32

        # load neural responses (different for each neuron)
        response_data = torch.load(os.path.join(path, 'ns1_drc_responses.pt'))
        self.N_neurons = len(neuron_indexes)
        self.response_data = [response_data[neuron_idx] for neuron_idx in neuron_indexes]   # filter selected neurons
        self.I = [0]      # select neuron #0 by default

        # filter by stimulus type
        stim_mask = [True if ((stim_type in stimuli) or (stim_type in stimuli)) else False for stim_type in stim_types]
        self.spectrograms = spectrograms[stim_mask]
        for neuron_idx in range(len(self.response_data)):

            # responses
            neuron_responses = self.response_data[neuron_idx]['responses']
            neuron_responses = [neuron_responses[sound_idx] for sound_idx in range(N_sounds) if stim_mask[sound_idx]]
            neuron_responses = torch.stack(neuron_responses, dim=0)     # (S, R, T)
            if normalize_resps:
                psth_max = neuron_responses.mean(-2).amax(-1)
                neuron_responses /= psth_max.unsqueeze(-1).unsqueeze(-1)
            self.response_data[neuron_idx]['responses'] = neuron_responses

            # ccmaxes
            neuron_ccmaxes = self.response_data[neuron_idx]['ccmaxes']
            neuron_ccmaxes = [neuron_ccmaxes[sound_idx] for sound_idx in range(N_sounds) if stim_mask[sound_idx]]
            self.response_data[neuron_idx]['ccmaxes'] = neuron_ccmaxes

            # TTRCs
            neuron_ttrcs = self.response_data[neuron_idx]['ttrcs']
            neuron_ttrcs = [neuron_ttrcs[sound_idx] for sound_idx in range(N_sounds) if stim_mask[sound_idx]]
            self.response_data[neuron_idx]['ttrcs'] = neuron_ttrcs

        self.species = 'ferret'
        self.F = self.spectrograms.shape[-2]

    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return len(self.spectrograms)

    def __getitem__(self, sound_index):
        """Retrieves a single sample from the dataset."""
        # Because all neurons are responsive to all stimuli, the neuron mask is True everywhere
        spectro = self.spectrograms[sound_index]            # (1, F, T)
        pop_data = [self.response_data[i] for i in self.I]      # select neuron(s) according to current indices
        responses = [neuron_data['responses'][sound_index] for neuron_data in pop_data]     # N * (R, T)
        responses = torch.stack(responses, 0)                                           # --> (N, R, T)
        nrn_mask = torch.ones(len(self.I)).bool()                                           # (N,)
        return spectro, responses, nrn_mask
