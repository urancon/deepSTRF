import os
import h5py
import mat73
import tables
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from deepSTRF.datasets.video.video_dataset import VideoNeuralDataset

class CRCNS_PVC11_Dataset(VideoNeuralDataset):
    """
    A PyTorch dataset for handling a subset of the CRCNS-PVC11 dataset, which contains neural responses 
    and video stimuli recorded from the primary visual cortex (V1) of macaque monkeys.
    
    =============== SOURCE ================
    
    The original dataset is freely available from:
        https://crcns.org/data-sets/vc/pvc/about

    Key Publications:
      - Smith, M.A. & Kohn, A. (2008). "Spatial and temporal scales of neuronal correlation in primary visual cortex", 
        Journal of Computational Neuroscience, 28:12591-12603.
      - Kelly, M.C. et al. (2010). "Local field potentials indicate network state and account for neuronal response variability", 
        Journal of Computational Neuroscience, 29:567-579, http://dx.doi.org/10.1007/s10827-009-0208-9.

    =============== DETAILS ===============
    
    This dataset comprises extracellular recordings from visually evoked neural responses to movie stimuli in the primary visual 
    cortex (V1) of anesthetized macaques.
    
    Data Acquisition:
      - Two macaque monkeys passively viewed dynamic visual stimuli.
      - Recordings were made using Utah array extracellular recording coupled with spike sorting.
      - Monkey #1 provides 69 clean units; Monkey #2 provides 104 clean units.
      
    Video Stimuli:
      - Stimulus types include 'natural' (e.g. a movie of a monkey swimming), 'noise', and 'gratings'.
      - Each stimulus is 30 seconds long and presented 120 times (total ~1 hour per type).
      - Videos are provided in grayscale at a resolution of 320 × 320 pixels.
      - The video recordings have a frame rate of 25 FPS (i.e., one frame every 40 ms).
      - Each 30-second stimulus was presented 120 times (~1 hour of data per stimulus type).

    =============== STRUCTURE ================
    
    The dataset is organized into four main tensors:
      - videos: Tensor of shape (N_stim_types, Channels, Height, Width, Time)
      - responses: Tensor of shape (N_neurons, N_stim_types, N_trials, Time)
      - ccmaxes & ttrcs: Precomputed metrics for each neuron and stimulus 
         (shape: (N_stim_types, N_neurons))
      
    Additional preprocessing options include:
      - Changing temporal resolution.
      - Normalizing and smoothing neural responses.
      - Adjusting the spatial resolution of video stimuli.
      - Adding noise and splitting video clips into shorter segments.

    =============== WARNING NOTE ================
    
    - Video stimuli are typically very large files, especially when operating at high temporal 
    resolutions (e.g., 1 ms). Processing these large datasets may lead to high memory usage. Use caution 
    and ensure that your system has adequate resources.
    - Neurons are pooled across monkeys; neuron identity is not matched across stimulus types. 
    """

    def __init__(self, 
                 path: str, 
                 spatial_resol=(320, 320),
                 animal_no=(1, 2), 
                 stimuli=('noise', 'gratings', 'natural'), 
                 normalize_videos: bool=False, 
                 normalize_responses: bool=True,
                 response_smoothing: bool=False,
                 add_noise: bool=False,
                 temporal_resolution: int=1,
                 split_into_clips : bool=False,
                 new_clip_len: int=3000):
        
        self.videos = []
        self.responses = []

        # Load the data for each selected stimulus.
        for stim in stimuli:
            print(f'stim = {stim}')
            assert stim in ['noise', 'gratings', 'natural'], \
                f"Received invalid stimulus type {stim}, expected 'noise', 'gratings' or 'natural'."

            # Load the stimulus (video) data.
            stim_file_path = os.path.join(path, f'data/{stim}_movie.mat')
            stim_data = sio.loadmat(stim_file_path)['M']  # Shape: (H, W, T)
            stim_data = torch.from_numpy(stim_data).float()
            stim_data = stim_data.unsqueeze(0)  # Reshape to: (C=1, H, W, T)
            self.videos.append(stim_data)

            resps_curr_stim = []
            for animal_idx in animal_no:
                assert animal_idx in (1, 2), \
                    f"Received invalid animal number {animal_idx}, expected 1 or 2."

                # Load neural responses.
                resp_file_path = os.path.join(path, f'data/S_monkey{animal_idx}_{stim}_movie.mat')
                try:
                    resp_data = mat73.loadmat(resp_file_path)['S']['spikes']
                except:
                    resp_data = sio.loadmat(resp_file_path)['S']['spikes']
                resp_data = torch.from_numpy(np.stack(resp_data))  # Shape: (R, N, T)
                resp_data = resp_data.permute(1, 0, 2).float()       # Reshape to: (N, R, T)
                resps_curr_stim.append(resp_data)

            # Concatenate responses from different animals.
            resps_curr_stim = torch.cat(resps_curr_stim, dim=0)  # Shape: (N_total, R, T)
            self.responses.append(resps_curr_stim)

        # Stack the data along the stimulus dimension.
        self.responses = torch.stack(self.responses).permute(1, 0, 2, 3)  # Final shape: (N, S, R, T)
        self.videos = torch.stack(self.videos)  # Final shape: (S, C, H, W, T)
        
        self.N_neurons, _, self.R, _ = self.responses.shape
        self.S, self.C, self.H, self.W, self.T = self.videos.shape
        
        self.native_temporal_resolution_video = 40  # ms per frame in videos
        self.native_temporal_resolution_responses = 1  # ms per sample in responses
        
        # Adapt temporal resolution to synchronize videos and responses.
        assert (temporal_resolution >= 1) and (temporal_resolution <= self.native_temporal_resolution_video), \
            f"Temporal resolution must be in [1, {self.native_temporal_resolution_video}] ms"
        self.responses, self.videos, self.T = self.change_temporal_resolution(
            self.native_temporal_resolution_video,
            self.videos,
            self.native_temporal_resolution_responses,
            self.responses,
            temporal_resolution,
        )
        self.dt = temporal_resolution  # Updated temporal resolution (ms)
        
        # Optionally split each clip into shorter clips.
        if split_into_clips:
            self.responses, self.videos = self.split_into_clips(
                clip_length_frames=new_clip_len,
            )     

        # Optionally normalize responses.
        if normalize_responses:
            self.responses = self.normalize_responses(self.responses)
            
        # Optionally smooth responses with a Gaussian kernel.
        if response_smoothing:
            self.responses = self.smooth_responses(self.responses)

        # Adjust the spatial resolution of video stimuli if necessary.
        if spatial_resol != (self.H, self.W):
            self.videos = self.change_spatial_resolution(self.videos, spatial_resol)

        # Optionally normalize videos.
        if normalize_videos:
            self.videos = self.normalize_videos(self.videos)
            
        # Optionally add random noise to videos.
        if add_noise:
            self.videos = self.add_noise_to_videos(self.videos, noise_level=0.1)                
          
        self.I = [0]  # Default: select neuron #0
        print(f"set contains {len(self.videos)} videos.")
        self.signal_powers = self.get_responses_signal_power(self.responses)  # Precomputed signal power, (N_neurons_ N_seqs)

    def __getitem__(self, video_index):
        """
        Retrieves a single sample from the dataset.

        Returns:
            video (torch.Tensor): Video stimulus of shape (C, H, W, T).
            response (torch.Tensor): Neural responses of shape (N_neurons, N_repeats, T).
            signal_power (torch.Tensor): Precomputed signal power (one value per neuron).
            nrn_mask (torch.Tensor): Boolean mask indicating the selected neurons.
        """
        video = self.videos[video_index]
        response = self.responses[self.I, video_index]
        signal_power = self.signal_powers[self.I, video_index]
        nrn_mask = torch.ones(len(self.I)).bool()
        return video, response, signal_power, nrn_mask
    
    def __len__(self):
        """
        Returns the number of video samples available in the dataset.
        """
        return len(self.videos)
    

