import os
from scipy.io import wavfile
import torch
import tables
import numpy as np

from deepSTRF.datasets.video.video_dataset import VideoNeuralDataset

def read_mat_file(path):
        try:
            with tables.open_file(path, 'r') as f:
                data = {
                    'cellid': torch.tensor(f.get_node('/cellid')[:], dtype=torch.int32),
                    'psths': torch.tensor(f.get_node('/psths')[:], dtype=torch.float32),
                    'rawStims': torch.tensor(f.get_node('/rawStims')[:], dtype=torch.float32)
                }
            f.close()
            return data
        except (OSError, tables.exceptions.HDF5ExtError) as e:
            print(f"Error reading file {path}")
            return None


class CRCNS_MT2_Dataset(VideoNeuralDataset):
    """
    A PyTorch dataset for handling the CRCNS-MT2 dataset, which contains extracellular recordings 
    from area MT of awake macaque monkeys during exposure to naturalistic movie stimuli.

    =============== SOURCE ================
    
    The original dataset is freely available from:
        https://crcns.org/data-sets/vc/mt-2/about

    Key Publications:
      - Nishimoto, S. & Gallant, J.L. (2011). "A three-dimensional spatiotemporal receptive field model explains responses of area MT neurons to naturalistic movies", 
        Journal of Neuroscience, 31(41):14551–14564.
      - Nishimoto, S. & Gallant, J.L. (2018). "Extracellular recordings from area MT of awake macaques in response to naturalistic movies", 
        CRCNS.org, http://dx.doi.org/10.6080/K0DN4374.

    =============== DETAILS ===============
    
    This dataset comprises extracellular single-neuron recordings from the middle temporal (MT) area 
    of awake macaque monkeys performing a fixation task while viewing motion-enhanced natural movies.
    
    Data Acquisition:
      - Extracellular signals were recorded using epoxy-coated tungsten electrodes (FHC).
      - Signals were spike-sorted to isolate single neurons.
      - 45 neurons were recorded across two monkeys (Macaca mulatta).
      - Spike counts (PSTHs) were trial-averaged and binned at 83 Hz (i.e., one bin per video frame, ~12 ms).

    Movie Stimuli:
      - Naturalistic movies were used, enhanced to emphasize motion components.
      - Movies were spatially cropped to approximately twice each neuron's receptive field.
      - Frames were downsampled to grayscale 128 × 128 pixels.
      - Frame rate: 83 frames per second (~12 ms per frame).

    Data Organization:
      - Each .mat file contains data for one neuron, including stimulus clips and neural responses.
      - Some neurons share the same movie stimuli (e.g., 'ct0053_arg0466d_128.mat' and 'ct0055_arg0466d_128.mat').

    =============== STRUCTURE ===============
    
    After loading and preprocessing, the dataset tensors are organized as:
      - videos: Tensor of shape (N_neurons, N_sequences, Channels=1, Height, Width, Time)
      - responses: Tensor of shape (N_neurons=1, N_sequences, Repeat=1, Time)
    
    Key processing steps:
      - Temporal resolution can be changed (default dt = 12 ms, but options for dt = 24 ms or 36 ms are allowed).
      - Spatial resolution can be adjusted as needed.
      - Data can be normalized or smoothed.
      - Long sequences can be split into smaller clips of fixed length (`seq_len`).
      - Neuron indices are selectable for flexible subset creation.

    =============== PARAMETERS ===============
    
    Args:
        path (str): Path to the directory containing the .mat files.
        set (str): Either 'train' or 'test'; determines which frame ranges are used. (Currently not enforced.)
        seq_len (int, optional): Length of sequences (in number of frames) to split into. 
                                 If None, full sequence is used.
        spatial_resol (tuple, optional): Target spatial resolution (Width, Height) of videos. Default is (128, 128).
        normalize_videos (bool): If True, normalize the video stimuli.
        normalize_responses (bool): If True, normalize the neural responses.
        response_smoothing (bool): If True, apply Gaussian smoothing to responses.
        add_noise (bool): If True, add random noise to video stimuli.
        temporal_resolution (int): Desired temporal resolution in ms. Default is 12 ms (native resolution).

    =============== WARNING NOTE ================
    
    - As it is processed here, the dataset is suitable only for fitting a model to a single neuron, and not a population of neurons.  
    - Use `sorted()` when loading neuron indices to ensure repeatability across different platforms.
    - Some neurons may share the same stimulus movie clips; be cautious during population-level analyses.
    - Handling long video sequences or very fine temporal resolutions may cause high memory usage.
    - While the original dataset defines training and test ranges, this implementation currently loads all data without enforcing a train/test split.
    
    """

    def __init__(self, 
                 path: str, 
                 neuron_indices: list=list(range(8, 12)), # TODO
                 set: str=None,
                 clip_length: int=2000, 
                 spatial_resol: tuple=(128, 128),
                 normalize_videos: bool=False, 
                 normalize_responses: bool=False,
                 response_smoothing: bool=False,
                 add_noise: bool=False,
                 temporal_resolution: int=12,
                 ):

        #assert set == "train" or set == "test", f"Unexpected value '{set}' for argument 'set', choose between 'train' or 'test'."

        # load response data
        all_videos = []
        all_responses = []
        
        self.native_temporal_resolution_video = 12  # ms per frame in videos
        self.native_temporal_resolution_responses = 12  # ms per sample in responses
        
        successful_reads = 0
        failed_reads = 0
        
        for neuron_indice in neuron_indices:
            file_path = os.path.join(path, f'mt-2/data/ct{neuron_indice:04d}_arg0466d_128.mat')
            data = read_mat_file(file_path)
            
            if data:
                successful_reads += 1
                current_movie = data['rawStims']  # (T, H, W)
                current_movie = current_movie.permute(1, 2, 0).unsqueeze(0)  # (C=1, H, W, T)
                    
                split_clips = torch.stack(torch.split(current_movie, clip_length, dim=-1)) # (N_seqs, C=1, H, W, new_T)
                split_responses = torch.stack(torch.split(data['psths'], clip_length, dim=-1)) # (N_seqs, R=1, new_T)
                
                all_videos.append(split_clips)
                all_responses.append(split_responses)
                    
            else:
                failed_reads += 1
                    
            print(f"Files reading status: {successful_reads} successful / {failed_reads} failed")

        # We consider that each clip corresponds to a same neuron (acceptable when doing single neuron fitting). 
        self.videos = torch.cat(all_videos, dim=0).unsqueeze(0)        # Shape: (N_neurons = 1, N_seq = sum(N_clips), C, H, W, T)
        self.responses = torch.cat(all_responses, dim=0).unsqueeze(0)    # Shape: (N_neurons = 1, N_seq = sum(N_clips), R=1, T)
        self.responses = torch.nan_to_num(self.responses, nan=0.0)
        
        self.N_neurons, _, self.R, _ = self.responses.shape
        _, self.S, self.C, self.H, self.W, self.T = self.videos.shape
        
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
        
        # additional attributes
        self.species = 'macaque'
        self.areas = 'MT'
        
        self.I = [0]
        print(f"{set} set contains {self.S} videos for {self.N_neurons} neuron.")

    def __getitem__(self, video_index):
        """Retrieves a single sample from the dataset."""
        video = self.videos[self.I, video_index]     # (1, C=1, H, W, T)
        resp = self.responses[self.I, video_index]    # (R=1, T) 
        signal_power = torch.ones(len(self.I)).bool()
        nrn_mask = torch.ones(len(self.I)).bool()
        return video, resp, signal_power, nrn_mask
    
    def __len__(self):
        n_seqs = self.S
        return n_seqs


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Ensure the directory and file paths exist
    dataset_path = 'datasets/CRCNS_MT2/'  # Replace with your directory path
    neuron_indices = [1, 5]  # Replace with your neuron indices

    # Initialize the dataset
    dataset = CRCNS_MT2_Dataset(dataset_path, neuron_indices, set='train')

    print("Dataset initialized successfully!")

       

