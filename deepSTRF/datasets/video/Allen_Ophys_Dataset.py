import os
import numpy as np
import torch

from deepSTRF.datasets.video import VideoNeuralDataset


SET_IDX_DICT = {"train": 3, "val": 2, "valid": 2, "test": 1}


class Allen_Ophys_Dataset(VideoNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the Allen "Visual Coding" Neuropixel dataset and its many
    recording sites ('VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl')

    Features:
     - Mouse visual cortex neurons
     - 6 different areas ('VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl')
     - 2-photons imaging (neuropixel)
     - Responses to all three natural videos available (intro of the movie "Touch of evil", 1958)
     - official split for benchmarking
          - movie_three (2 mins, 10 trials) --> train
          - movie_two (30 secs, 10 trials) --> valid
          - movie_one (30 secs, 30 trials) --> test


     ============= STRUCTURE ==============

     self.stims: tensor of shape (T, C=1, H, W) with T depending on the chosen optimization split
     self.resps: tensor of shape (N, R, T) with R and T depending on the chosen optimization split
     self.nrn_meta: list of N str each containing the brain area of the corresponding neuron


        TODO:
         - description
         - select high SNR (e.g., >2) neurons ?
         - guarantee no optotagging ?
         - warping and gaze metadata (cf. https://community.brain-map.org/t/how-can-i-access-the-natural-images-used-in-the-brain-observatory/32/3)
         - include "natural_movie_shuffled" stimuli ?
         - include the 118 "natural_scenes" stimuli as a 30 s video ?

    """

    def __init__(self, path: str, areas: tuple=('VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'), spat_res:tuple = (304, 608),
                 seq_len:int = None, optim_set: str='train'):
        """
        Initializes the Allen_OPhys Dataset.

        Specific units can be selected according to their recording site.

        Parameters:
            path (str): Path to the 'Allen_OPhys/data/' folder containing files as indicated in our readme
            areas (tuple of str): recording sites of interest, can be 'VISal', 'VISam', 'VISl', 'VISp', 'VISpm' or 'VISrl'
            spat_res (tuple of int): spatial resolution to downsample the stimuli. Default: 304 x 608
            optim_set (str): either of 'train', 'valid', 'test' to select the corresponding official set
        """

        super().__init__(path)

        # load the data
        optim_set_idx = SET_IDX_DICT[optim_set]
        self.stims = np.load(os.path.join(path, f"natural_movies/natural_movie_{optim_set_idx}.npy"))               # (T, H, W)
        self.stims = torch.from_numpy(self.stims)
        self.stims = torch.unsqueeze(self.stims, dim=1)                                                             # (T, C=1, H, W)
        self.resps = []
        self.nrn_meta = []
        for area in areas:
            resps = torch.load(os.path.join(path, f"responses/{area}/allen_ophys_resps_movie{optim_set_idx}.pt"))   # (N, R, T)
            N = len(resps)
            self.resps.append(resps)
            self.nrn_meta = self.nrn_meta + ([area] * N)
        self.resps = torch.cat(self.resps, dim=0) if len(self.resps) > 1 else self.resps[0]     # concatenate neuron population along nrn dim

        # change spatial resol / downsample
        if spat_res is not None:
            self.stims = torch.nn.functional.interpolate(self.stims, size=spat_res, mode='bilinear')
        else:
            pass
        self.HW = self.stims.shape[-2:]

        # cut sequence into sub-sequences of length seq_len
        T = self.stims.shape[0]
        if seq_len is not None:
            self.seq_len = T-seq_len
            self.indices = list(range(0, T-seq_len, 1))
        else:
            self.seq_len = T
            self.indices = [0]

        # general attributes
        self.dt = 0.333 # ms
        self.species = 'mouse'
        self.area = areas
        self.N_neurons = self.resps.shape[0]
        self.I = [0]

    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return len(self.indices)

    def __getitem__(self, stim_index):
        """Retrieves a single sample from the dataset."""
        index = self.indices[stim_index]
        stim = self.stims[index:index+self.seq_len]             # (T, C=1, H, W)
        resp = self.resps[:, :, index:index+self.seq_len]       # (N, R, T)
        nrn_mask = torch.ones(len(self.I)).bool()               # (N,)
        return stim, resp, nrn_mask
