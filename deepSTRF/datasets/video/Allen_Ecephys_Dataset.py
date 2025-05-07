import os
import glob
import numpy as np
import torch

from deepSTRF.datasets.video import VideoNeuralDataset


SET_IDX_DICT = {"train": 3, "valtest": 1}

class Allen_Ecephys_Dataset(VideoNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the Allen "Visual Coding" Neuropixel imaging dataset and its many
    recording sites ('VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl')

     - Mouse visual cortex neurons
     - electrophysiology (Neuropixel)
     - Responses to two natural videos (intro of the movie "Touch of evil", 1958)
     - high SNR (> 1.5)

     OFFICIAL SPLIT (Example)
      movie_three (2 mins, 10 trials) --> train
      movie_one (30 secs, 20 trials) --> valid/test
      scenes (30 secs) --> ?

        TODO:
         - description
         - select high SNR neurons ? --> add SNR to self.nrn_meta
         - warping and gaze metadata (cf. https://community.brain-map.org/t/how-can-i-access-the-natural-images-used-in-the-brain-observatory/32/3)
         - include the 118 "natural_scenes" stimuli as a 30 s video ?

    """

    def __init__(self, path: str, areas: tuple=('VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'), spat_res:tuple = (304, 608),
                 seq_len:int = None, optim_set: str='train'):
        """
        Initializes the Allen_Ecephys Dataset.

        Specific units can be selected according to their recording site.

        Parameters:
            path (str): Path to the 'Allen_OPhys/data/' folder containing files as indicated in our readme
            areas (tuple of str): recording sites of interest, can be 'VISal', 'VISam', 'VISl', 'VISp', 'VISpm' or 'VISrl'
            spat_res (tuple of int): spatial resolution to downsample the stimuli
            optim_set (str): either of 'train', 'valid', 'test' to select the corresponding official set
        """

        super().__init__(path)

        # load the data
        optim_set_idx = SET_IDX_DICT[optim_set]
        self.stims = np.load(os.path.join(path, f"natural_movies/natural_movie_{optim_set_idx}.npy"))   # (T, C, H, W)
        self.stims = torch.from_numpy(self.stims)

        self.resps = []
        self.nrn_meta = []
        for area in areas:
            resp_files = glob.glob(path + '/**/*.pt', recursive=True)
            resp_files = [f for f in resp_files if (f'movie{optim_set_idx}' in f) and area in f.split('_')]
            for file in resp_files:
                resps = torch.load(file)   # (N, R, T)
                N = len(resps)
                area = file.split('_')[-2]
                self.resps.append(resps)
                self.nrn_meta = self.nrn_meta + ([area] * N)
        self.resps = torch.cat(self.resps, dim=0) if len(self.resps) > 1 else self.resps[0]     # concatenate responses along nrn dim

        # TODO: make sure that the indexing of neurons stays the same between train and valtest ! --> add neuron_id to self.nrn_meta ?

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


if __name__ == "__main__":
    train_set = Allen_Ecephys_Dataset("./Allen_Ecephys/data/", areas=('VISp', 'VISpm'),
                                      spat_res=(152, 304),
                                      seq_len=75,
                                      optim_set='train')
