import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset


def get_animals_ids(data_path):
    """
    Takes in the path of the 'CRCNS_AA1/data/' folder, goes through 'Field_L/' and 'MLd', and outputs a list of unique
     animal ids, which are the string preceding the first underscore of each subfolder.
     e.g., 'gg0304_4_B' --> 'gg0304'
    """
    animal_ids = []
    for area_folder in ['Field_L_cells', 'MLd_cells']:
        cell_names = os.listdir(os.path.join(data_path, area_folder))
        for cell_name in cell_names:
            animal_id = cell_name.split('_')[0]
            if animal_id not in animal_ids:
                animal_ids.append(animal_id)
    return animal_ids


def get_area_cells(data_path):
    """
    From the cell_regions.csv file, returns a dictionary with area labels as keys and lists of cell names as values.
    """
    cell_dict = {
        'Field_L': [],    # e.g., '[pupu2122_2_A, pupu2122_2_B, ...]
        'MLd': []
    }
    for area in cell_dict.keys():
        area_folder = f'{area}_cells/'
        cell_names = os.listdir(os.path.join(data_path, area_folder))
        for cell_name in cell_names:
            cell_dict[area].append(cell_name)
    return cell_dict


def get_stim_ids(data_path):
    """
    From the cell_regions.csv file, returns a dictionary with area labels as keys and lists of cell names as values.
    """
    stim_dict = {
        'conspecific': [],  # e.g., ['723792DF8CA8D0B99B8059503E5006BA.wav', '4922458336F516A1D0E31DA099896C0A.wav', ...]
        'flatrip': []
    }
    for stim_type in stim_dict.keys():
        stim_folder = f'all_stims/{stim_type}/'
        stim_names = os.listdir(os.path.join(data_path, stim_folder))
        for stim_name in stim_names:
            stim_dict[stim_type].append(stim_name)
    return stim_dict


class CRCNS_AA1_Dataset(AudioNeuralDataset):
    """
    TODO: description

    TODO:
     - find a way to use pre-onset activity ?
     - get_F() method



    50 cells in field L
    50 cells in field MLd

    only cell 'pipu1018_2_A' in 'MLd' does not have responses to 'conspecific' stims
    only cell 'pipu1018_2_B' in 'Field_L' does not have responses to 'conspecific' stims

    """

    def __init__(self, path: str, areas=('Field_L', 'MLd'), stimuli=('conspecific', 'flatrip'), animals='all',
                 neuron_indexes='all', dt=1, smooth=True):
        """
        TODO: update description

        smooth --> smooth with 21 ms Hanning window

        TODO: use sorted() for repeatability across platforms

        """

        super().__init__(path)

        self.species = 'zebra finch'
        # hop_length (samples) | dt (ms)
        # 320 | 10
        # 160 | 5
        # 32  | 1
        self.dt = dt
        hl = dt * 32
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=10*hl, hop_length=hl, n_mels=32)   # n_fft=800
        self.F = 32

        #######################
        # 1. get metadata
        #######################

        # get all animal ids
        ANIMAL_IDs = get_animals_ids(path)

        # get all cells for each area
        AREA_CELLs = get_area_cells(path)

        # get all stims
        STIM_IDs = get_stim_ids(path)


        ################################
        # 2. cells & stims selection
        ################################

        # pre-selection of cell names, based on stim_type, animal, and area
        cells = []

        # filter cells by area
        cell_areas = []
        if areas == 'all':
            areas = ('Field_L', 'MLd')
        for area in areas:
            cells += AREA_CELLs[area]
            cell_areas += ([area] * len(AREA_CELLs[area]))

        # filter cell by animal
        if animals == 'all':
            animals = ANIMAL_IDs
        cells = [cell for cell in cells if cell.split('_')[0] in animals]
        cell_animals = [cell.split('_')[0] for cell in cells]

        # filter stimuli by stimulus type
        stims = []
        stim_types = []
        for stim_type in stimuli:
            stims += STIM_IDs[stim_type]
            stim_types += [stim_type] * len(STIM_IDs[stim_type])

        # filter cells by stimulus type:
        # remove cells if they don't have any stimulus of the required type
        for area in ['Field_L', 'MLd']:
            area_path = os.path.join(path, f'{area}_cells')
            for cell in os.listdir(area_path):
                # if the cell is in the current selection of cells, check if it has responses to at least one stim type
                if cell in cells:
                    cell_path = os.path.join(area_path, cell)
                    cell_stim_types = os.listdir(cell_path)
                    i = 0
                    for stim_type in cell_stim_types:
                        if stim_type not in stimuli:
                            i += 1
                    # if none of the available stim types for this cell were among the required, remove the cell
                    if i == len(cell_stim_types):
                        cell_idx = cells.index(cell)
                        cells.remove(cell)
                        cell_animals.remove(cell.split('_')[0])
                        cell_areas.remove(cell_areas[cell_idx])
                else:
                    continue

        self.N_neurons = len(cells)

        ################################
        # 3. cells & stims selection
        ################################

        self.spectrograms = []      # --> S * (1, F, T)
        self.responses = []         # --> S * (N, R, T)
        self.nrn_masks = []         # --> S * (N,)
        self.stim_meta = []         # --> S * ['stim_name', 'stim_type']
        stim_meta = list(zip(stims, stim_types))                    # S * ['stim_name', 'stim_type']
        self.nrn_meta = list(zip(cells, cell_animals, cell_areas))  # --> N * ['cell_id', 'animal_id', 'area']

        for s, (stim_name, stim_type) in enumerate(stim_meta):

            # =========== load the stim ============

            stims_dir = os.path.join(path, f"all_stims/{stim_type}/")

            wav, sr = torchaudio.load(os.path.join(stims_dir, stim_name), normalize=True)  # sample rate: 32 kHz (mono)
            spec = torch.log10(transform(wav))  # (T,) --> (1, F, T-)

            # =========== load the resps ============

            pop_resps = []
            no_data_nrn_idces = []

            for n, (cell_name, animal_id, area) in enumerate(self.nrn_meta):  # TODO: for loop on self.nrn_meta

                # some cells may not have any response for the current stim type;
                # if that is the case --> null response directly
                if stim_type not in os.listdir(os.path.join(path, f'{area}_cells/', cell_name)):
                    resp = torch.zeros(20, 1000)  # TODO: use torch.full(..., fill_value=torch.nan) instead ?
                    no_data_nrn_idces.append(n)

                # if they do have responses to this stim_type:
                else:

                    #  1. find the stim file corresponding to the stim_name
                    #  2. find the spike file corresponding to that stim file, if any
                    spike_dir = os.path.join(path, f'{area}_cells/', cell_name, stim_type)
                    no_stim = True
                    stim_files = [file for file in os.listdir(spike_dir) if 'stim' in file]
                    for stim_file in stim_files:
                        i = int(stim_file[4:])  # e.g., 'stim20'  --> '20'
                        with open(os.path.join(spike_dir, stim_file)) as f:
                            wavname = f.readlines()[0][:-1]
                        if wavname == stim_name:
                            no_stim = False
                            break

                    # some neurons may have responses to stims of this type, but not this one in particular;
                    # in this case --> null response
                    if no_stim:
                        resp = torch.zeros(20, 1000)  # TODO: use torch.full(..., fill_value=torch.nan) instead ?
                        no_data_nrn_idces.append(n)

                    #  3. if they do indeed have a response to this specific stim, get the response
                    else:
                        spike_file = os.path.join(spike_dir, f'spike{i}')
                        try:
                            resp = load_spike_file(spike_file, dt=self.dt)  # (R, T)
                        # some neurons have a 'stimXX' file, but not the corresponding 'spikeXX' file
                        except FileNotFoundError:
                            resp = torch.zeros(20, 1000)  # TODO: use torch.full(..., fill_value=torch.nan) instead ?
                            no_data_nrn_idces.append(n)

                # add the cell's response, whether it is null or not, to the population activity for this stim
                pop_resps.append(resp)

            # make a neuron mask for this stim: 1 --> response data for this neuron, 0 --> no response data
            assert len(pop_resps) == self.N_neurons
            mask = torch.ones(self.N_neurons)
            mask[no_data_nrn_idces] = 0
            mask = mask.bool()

            # if none of the neurons have emitted a spike, do not keep this stim and its response
            if mask.sum() == 0:
                continue

            # if activity (spikes) have been recorded, keep it and continue the processing
            else:

                # pad responses so that their tensors all have the same time dimension as the stimulus duration;
                # also remove extra trials to the responses of very few units   TODO: add null trials instead ??
                T = spec.shape[-1]  # nbr of timesteps of curr stim
                R = min([resps.shape[-2] for resps in
                         pop_resps])  # min nbr of repeats in the pop for this stim   # TODO: change this !! no discarding of data !!!
                for n in range(self.N_neurons):
                    # if the response of the neuron is shorter than the sound, pad to the right (future)
                    if pop_resps[n].shape[-1] <= T:
                        Pt = T - pop_resps[n].shape[-1]
                        pop_resps[n] = torch.nn.functional.pad(pop_resps[n], pad=(0, Pt), mode='constant', value=0.)
                    # if the response of the neuron is longer than the sound (because spikes were detected after sound
                    # termination), just keep the part of the response associated with the stim
                    else:
                        pop_resps[n] = pop_resps[n][:, :T]
                    # only keep the minimum number of trials    TODO: add null trials  up to the max number of trials instead ??
                    pop_resps[n] = pop_resps[n][:R, :]

                assert len(mask) == len(pop_resps), f"{len(mask)}, {len(pop_resps)}"

                pop_resps = torch.stack(pop_resps, dim=0)  # (N, R, T)

                # finally, apply a 21 ms hanning window to smooth the PSTHs
                if smooth:
                    Kt_hanning = (21 // dt) if ((21 // dt % 2) == 1) else (21 // dt) + 1  # odd kernel
                    pop_resps = apply_hanning_window(pop_resps, hanning_size=Kt_hanning)

                self.responses.append(pop_resps)
                self.nrn_masks.append(mask)
                self.spectrograms.append(spec)
                self.stim_meta.append((stim_name, stim_type))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.spectrograms)

    def __getitem__(self, sound_index):
        """Retrieves a single sample from the dataset."""
        spectro = self.spectrograms[sound_index]            # (1, F, T)
        responses = self.responses[sound_index][self.I]     # (N, R, T)
        nrn_mask = self.nrn_masks[sound_index][self.I]      # (N,)
        return spectro, responses, nrn_mask

    def get_pop_metadata(self):
        # TODO: return an array of metadata containing for each neuron: area and animal
        raise NotImplementedError


def time_binning(spike_times, dt=1.):
    """
    spike_times is a list of POSITIVE floats (in ms), relative to stimulus onset

    returns a spike count tensor of shape (T,) with T the number of time bins of size dt ms
    """
    # Step 0: Check for the absence of spikes
    if len(spike_times) == 0:
        return torch.zeros(1)

    else:
        # Step 1: Get the index of each spike in each bin
        bin_indices = [int(t // dt) for t in spike_times]

        # Step 2: Determine the bin index for each timing
        max_bin_index = max(bin_indices)
        spike_counts = [0] * (max_bin_index + 1)
        for bin_index in bin_indices:
            spike_counts[bin_index] += 1

        # Step 3: Convert the counts to a 1D torch tensor
        spike_counts_tensor = torch.tensor(spike_counts, dtype=torch.float32)

        return spike_counts_tensor


def load_spike_file(path, dt=1.):
    """
    Reads a spikeX .txt file of a unit's response to a stimulus.

     Converts the post-stimulus onset spike arrival times into a (R, T) torch.Tensor
      where R is the number of repeats/trials and T the number of time bins (in dt ms)

    """
    with open(path, 'r') as spike_f:

        responses_post = []
        for line in spike_f:
            spiketimes_ms = line.split(' ')[:-1]  # unwanted '\n'
            spiketimes_ms = [float(t) for t in spiketimes_ms]  # str --> float
            spiketimes_ms_post = [t for t in spiketimes_ms if t >= 0]  # post-stimulus onset spikes
            spiketimes_post = time_binning(spiketimes_ms_post, dt=dt)   # tensor of shape (T,)
            responses_post.append(spiketimes_post)

        # pad responses_post to the right with zeros (no activity) such that they all have the same nbr of time steps
        max_post_duration = max([r.shape[-1] for r in responses_post])  # in nbr of time steps
        for i in range(len(responses_post)):
            P_post = max_post_duration - len(responses_post[i])
            responses_post[i] = torch.nn.functional.pad(responses_post[i], pad=(0, P_post), mode='constant', value=0.)

        # stack response trials in a new dimension
        responses_post = torch.stack(responses_post, dim=0) # (R, T)

        return responses_post


def apply_hanning_window(tensor, hanning_size=21):
    L = hanning_size  # Length of the Hanning window
    N, R, T = tensor.shape

    # Create a Hanning window of length L
    hanning_window = np.hanning(L)
    hanning_window = torch.tensor(hanning_window, dtype=tensor.dtype, device=tensor.device).unsqueeze(0).unsqueeze(0)

    # Pad the tensor to apply the window correctly
    pad_size = (L - 1) // 2
    padded_tensor = F.pad(tensor, (pad_size, pad_size), mode='constant')

    # Apply the Hanning window using convolution
    padded_tensor = padded_tensor.flatten(0, 1).unsqueeze(1)  # (N, R, T) --> (N*R, 1, T)
    result = F.conv1d(padded_tensor, hanning_window)
    result = result.unflatten(0, (N, R))[:, :, 0, :]

    return result
