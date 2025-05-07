import os
import numpy as np
import torch

from deepSTRF.datasets.video.video_dataset import VideoNeuralDataset


SET_IDX_DICT = {"train": 3, "val": 2, "valid": 2, "test": 1}


class Allen_Ophys_Dataset(VideoNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the Allen "Visual Coding" Neuropixel dataset.
    This dataset contains neural responses and corresponding video stimuli recorded from the mouse visual cortex,
    encompassing multiple visual areas.
    
    =============== SOURCE ================
    
    The full dataset and additional details are available from the Allen Institute for Brain Science:
        https://portal.brain-map.org/explore/circuits/visual-coding
    
    Further documentation and publications regarding the dataset can be found on the Allen Institute's website.

    =============== ABOUT ================
    
    The Allen "Visual Coding" dataset comprises high-quality neural recordings obtained via Neuropixel probes
    from multiple areas of the mouse visual cortex. The dataset includes responses to natural movies and is 
    organized according to official splits for training, validation, and testing.
    
    Features:
      - Neural responses from mouse visual cortex.
      - Recordings are acquired from 6 areas: 'VISal', 'VISam', 'VISl', 'VISp', 'VISpm', and 'VISrl'.
      - Data collected via 2-photon imaging (Neuropixel).
      - Stimuli include three natural movies (from the intro of "Touch of Evil", 1958).
      - Official set splits:
          - train: movie_three (2 mins, 10 trials)
          - valid: movie_two (30 secs, 10 trials)
          - test:  movie_one (30 secs, 30 trials)

    =============== STRUCTURE ================
    
    The dataset is organized into the following main tensors and attributes:
      - self.stims: A tensor of shape (T, C=1, H, W) containing the video stimulus frames.
                 After segmenting into sub-sequences, these are arranged as (N_seqs, H, W, C, seq_len)
                 after permutation.
      - self.resps: A tensor of shape (N_total_neurons, R, T) containing neural responses.
                 After segmentation, they are arranged as (N_total_neurons, N_seqs, R, seq_len).
      - self.signal_powers: Precomputed signal power for each neuron over the sequences, useful for further analysis.
    
    Additional attributes include:
      - self.dt: The temporal resolution (default: 0.333 ms).
      - self.seq_len: The length (in time bins) of each sub-sequence (set by the user or equal to the full sequence length).
      - Other attributes (e.g. self.species and self.area) provide metadata about the recordings.
    
    =============== WARNING NOTE ================
    
    Due to the high temporal resolution and the potential length of video sequences, the dataset can be very large 
    and memory-intensive. Users should ensure they have sufficient system resources before processing. Additionally,
    while some preprocessing options are provided (e.g., sequence segmentation, spatial resolution adjustments), 
    further custom processing may be required for some analyses.
    """

    def __init__(self, 
                 path: str, 
                 set: str='train',
                 areas: tuple=('VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'), 
                 seq_len: int = None,
                 spatial_resol:tuple = (304, 608),
                 normalize_videos: bool=False, 
                 normalize_responses: bool=True,
                 response_smoothing: bool=False,
                 add_noise: bool=False,):
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
        optim_set_idx = SET_IDX_DICT[set]
        self.stims = np.load(os.path.join(path, f"data/natural_movies/natural_movie_{optim_set_idx}.npy"))               # (T, H, W)
        self.stims = torch.from_numpy(self.stims)
        self.stims = torch.unsqueeze(self.stims, dim=1)                                                             # (T, C=1, H, W)
        self.resps = []
        self.nrn_meta = []
        for area in areas:
            resps = torch.load(os.path.join(path, f"data/responses/{area}/allen_ophys_resps_movie{optim_set_idx}.pt"))   # (N, R, T)
            N = len(resps)
            self.resps.append(resps)
            self.nrn_meta = self.nrn_meta + ([area] * N)
        self.resps = torch.cat(self.resps, dim=0) if len(self.resps) > 1 else self.resps[0]     # concatenate neuron population along nrn dim 

        # TODO : Stopped here !
        
        # Pre-cut the full sequence into sub-sequences
        T = self.stims.shape[0]
        self.seq_len = seq_len if seq_len is not None else T # In timesteps
        if seq_len is not None:
            self.all_stims = [self.stims[i:i+seq_len] for i in range(0, T - seq_len + 1)] # Each element has shape (seq_len, 1, H, W)
            self.all_resps = [self.resps[:, :, i:i+seq_len] for i in range(0, T - seq_len + 1)] # Each corresponding element in self.all_resps has shape (N_total_neurons, R, seq_len)
        else:
            self.all_stims = [self.stims]
            self.all_resps = [self.resps]
            
        self.all_stims = torch.stack(self.all_stims).permute(0, 2, 3, 4, 1)  # (N_seqs, C=1, H, W, seq_len)
        self.all_resps = torch.stack(self.all_resps).permute(1, 0, 2, 3)  # (N_total_neurons, N_seqs, R, seq_len)
        
        self.N_neurons, _, self.R, _ = self.all_resps.shape
        self.S, self.C, self.H, self.W, self.T = self.all_stims.shape

        # additional attributes
        self.dt = 33.3  # ms
        self.species = 'mouse'
        self.area = areas
        
        self.signal_powers = self.get_responses_signal_power(self.all_resps) # (N_neurons, N_seqs)
        
        # Optionally normalize responses
        if normalize_responses:
            self.all_resps = self.normalize_responses(self.all_resps)
            
        # Optionally smooth responses with a gaussian kernel
        if response_smoothing:
            self.all_resps = self.smooth_responses(self.all_resps)

        # change spatial resolution of video stims
        if spatial_resol != (self.H, self.W):
            self.all_stims = self.change_spatial_resolution(self.all_stims, spatial_resol)

        # Optionally normalize videos
        if normalize_videos:
            self.all_stims = self.normalize_videos(self.all_stims)
            
        # Optionally add random noise to videos
        if add_noise:
            self.all_stims = self.add_noise_to_videos(self.all_stims)
        
        self.I = [0]
        print(f"{set} set contains {len(self.all_stims)} videos.")

    def __len__(self):
        """Returns the number of sub-sequences in the dataset."""
        return len(self.all_stims)

    def __getitem__(self, stim_index):
        """Retrieves a single sample from the dataset."""
        stim = self.all_stims[stim_index]           # (T, C=1, H, W)
        resp = self.all_resps[self.I, stim_index]       # (N, R, T)
        signal_power = self.signal_powers[self.I, stim_index]  # (N, R)
        nrn_mask = torch.ones(len(self.I)).bool()               # (N,)
        return stim, resp, signal_power, nrn_mask
