import os
from collections import defaultdict
import h5py
import numpy as np
from scipy.signal import convolve
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Sequence

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset


def get_subgroups(group):
    """Return list of subgroup names under `group` in an h5 File."""
    return [name for name, obj in group.items() if isinstance(obj, h5py.Group)]


AA4_ANIMAL_IDS = ('BlaBro09xxF', 'GreBlu9508M', 'LblBlu2028M', 'WhiBlu5396M', 'WhiWhi4522M', 'YelBlu6903F')


class CRCNS_AA4_Dataset(AudioNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the AA4 dataset.


    =============== SOURCE ================

    See original papers for details:
     - "Meaning in the avian auditory cortex: Neural representation of communication calls" by Elie JE and Theunissen FE. (2015)
            European Journal of Neuroscience.
     - "Invariant neural responses for sensory categories revealed by the time-varying information for communication calls" by Elie JE and Theunissen FE. (2019)
            Plos Computational Biology.

    Data available at: https://crcns.org/data-sets/aa/aa-4/about-aa-4


    =============== DETAILS ================

    More details can be found in the dataset source, our dedicated readme file, or in the original papers.
    But in a nutshell:
    - 1401 extracellular, spike-sorted single and multi units of adult zebra finches (4 males, 2 females)
    - Field L, caudolateral and caudomedial mesopallium (CLM and CMM) and caudomedial nidopallium (NCM)
    - units were not precisely assigned one of the above areas
    - 3 stimulus classes: conspecific songs, calls, and ripple noise.
    - stimuli lasted for a few seconds and were each presented ~10 times
    - population fitting-compatible
    - batch-compatible


    =============== STRUCTURE ================

    Several main attributes:
     - self.spectrograms                list, [S * (1, F, T)]
     - self.responses                   list, [S * [N * (R, T)]]
     (- self.nrn_masks                   list, [S * (N,)])  TODO: not useful anymore in the new paradigm !!! simplify by removing it ?
     - self.stim_metadata               list, [S * ('stim_md5', 'stim_type', 'stim_class')]
     - self.pop_metadata                list, [N * ('animal_id', 'sex', (ldepth, rdepth))]

    """
    def __init__(self, path: str, stimuli=('song', 'call', 'mlnoise'), animals='all', dt_ms=1, smooth=True):
        super().__init__(path)

        # parameters
        self.dt = dt_ms  # ms
        self.F = 32
        self.animals = AA4_ANIMAL_IDS if animals == 'all' else animals
        self.stim_types = set(stimuli)

        # psth smoothing window
        self.smooth = smooth
        window_ms = 21
        win_len = max(1, int(window_ms / self.dt))
        self.smooth_win = np.hanning(win_len)
        self.smooth_win /= self.smooth_win.sum()

        # to collect per-stimulus info
        stim_uids = []  # ordered list of unique md5 stimuli
        stim_meta_map = {}  # md5 -> (md5, stim_type, stim_class)
        stim_spec_map = {}  # md5 -> spectrogram tensor (1, F, T)

        # to collect per-neuron info
        units_data = []  # list of dicts: {'meta': (animal, sex, (ld, rd)), 'responses': {md5: np.array}}

        # preload all wav spectrograms by stim_id per animal
        wav_specs_by_animal = {}
        for animal in self.animals:
            wav_dir = os.path.join(path, animal, 'wavfiles')
            specs = {}
            for fname in os.listdir(wav_dir):
                if not fname.endswith('.wav'): continue
                sid = os.path.splitext(fname)[0]    # 'stim85'
                waveform, sr = torchaudio.load(os.path.join(wav_dir, fname))
                hop = int(sr * self.dt / 1000)
                n_fft = hop * 10
                mel_tf = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr, n_mels=self.F,
                    n_fft=n_fft, hop_length=hop
                )
                spec = mel_tf(waveform)  # (1, F, T)
                if spec.ndim == 2:
                    spec = spec.unsqueeze(0)
                specs[sid] = spec
            wav_specs_by_animal[animal] = specs

        # iterate through neurons
        for animal in self.animals:
            sex = animal[-1]
            animal_path = os.path.join(path, animal)
            for fname in sorted(os.listdir(animal_path)):
                if not fname.endswith('.h5'): continue
                h5_path = os.path.join(animal_path, fname)
                with h5py.File(h5_path, 'r') as celldata:
                    try:
                        sortType = celldata.attrs.get('sortType', b'').decode()
                    except AttributeError:
                        sortType = celldata.attrs.get('sortType', b'')[0].decode()
                    if sortType in ('tdt', 'noise'): continue

                    # neuron metadata
                    ldepth = float(celldata.attrs.get('ldepth', np.nan))
                    rdepth = float(celldata.attrs.get('rdepth', np.nan))
                    nrn_meta = (animal, sex, (ldepth, rdepth))
                    responses = {}

                    # loop stimuli in this neuron
                    for cls in get_subgroups(celldata):
                        if cls in ('class_info', 'extra_info'): continue
                        grp = celldata[cls]
                        for stim in get_subgroups(grp):
                            stim_grp = grp[stim]
                            dur = float(stim_grp.attrs.get('stim_duration', 0.0))
                            if dur <= 0: continue

                            # stimulus metadata
                            stim_type = stim_grp.attrs.get('stim_type', b'').decode()
                            stim_class = stim_grp.attrs.get('stim_class', b'').decode()
                            stim_md5 = stim_grp.attrs.get('stim_md5', b'').decode()
                            # filter by requested types
                            if stim_type not in self.stim_types:
                                continue

                            # register unique stimulus
                            if stim_md5 not in stim_uids:
                                stim_uids.append(stim_md5)
                                stim_meta_map[stim_md5] = (stim_md5, stim_type, stim_class)
                                # fetch spectrogram by stim id
                                spec = wav_specs_by_animal[animal].get(f'stim{stim}')
                                stim_spec_map[stim_md5] = spec

                            # bin spikes
                            # TODO: if slight mismatch between temporal dimension of resps (below) and spec, use instead T = spec.size(-1)
                            T = int(np.ceil(dur / (self.dt * 1e-3)))
                            trial_spikes = []
                            for trial in get_subgroups(stim_grp):
                                times = stim_grp[trial]['spike_times'][()]
                                times = times[times >= 0]
                                if times.size > 0:
                                    trial_spikes.append(times)
                            if not trial_spikes: continue

                            R = len(trial_spikes)
                            counts = np.zeros((R, T), dtype=np.float32)
                            for i, times in enumerate(trial_spikes):
                                idx = (times / (self.dt * 1e-3)).astype(int)
                                idx = idx[idx < T]
                                for b in idx:
                                    counts[i, b] += 1
                            # smooth (optional)
                            if self.smooth:
                                for i in range(R):
                                    counts[i] = convolve(counts[i], self.smooth_win, mode='same')
                            if np.all(counts == 0): continue

                            responses[stim_md5] = counts

                    # keep neuron if any responses
                    if responses:
                        units_data.append({'meta': nrn_meta, 'responses': responses})

        # build final attributes
        self.S = len(stim_uids)
        self.N_neurons = len(units_data)
        # stimuli spectrogram list
        self.stims = [stim_spec_map[uid] for uid in stim_uids]
        # stim metadata list
        self.stim_meta = [stim_meta_map[uid] for uid in stim_uids]
        # neuron metadata
        self.nrn_meta = [u['meta'] for u in units_data]

        # responses: list of S lists, each of length N
        self.responses = []
        self.nrn_masks = []
        for uid in stim_uids:
            resp_list = []
            mask = []
            for u in units_data:
                if uid in u['responses']:
                    arr = u['responses'][uid]
                    tensor = torch.from_numpy(arr)
                    resp_list.append(tensor)
                    mask.append(True)
                else:
                    resp_list.append(torch.full((1, 1), float('nan')))
                    mask.append(False)
            self.responses.append(resp_list)
            self.nrn_masks.append(torch.tensor(mask, dtype=torch.bool))

        print("dataset loaded!")


    def __len__(self):
        """Return number of stimuli for which at least one selected neuron has a valid response. """
        if not hasattr(self, 'I') or len(self.I) == 0:
            return 0
        count = 0
        for mask in self.nrn_masks:
            # mask is a tensor of shape (N,)
            # check if any selected neuron index is True
            if any(mask[i].item() for i in self.I):
                count += 1
        return count

    def __getitem__(self, idx):
        """
        Retrieve stimulus-response pairs for given stimulus index or indices, only for selected neurons (self.I).

        Args:
            idx (int, slice, list of int): stimulus index or indices.

        Returns:
            spectrograms: Tensor or list of Tensors [(1, F, T)]
            responses: list or list of lists of Tensors [(R, T)], only for neurons in self.I
            nrn_masks: Tensor or list of Tensors [(len(self.I),)]
            stim_meta: tuple or list of tuples
        """
        # determine list of stimulus indices
        if isinstance(idx, int):
            indices = [idx]
            single = True
        elif isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self.stims))))
            single = False
        elif isinstance(idx, (list, tuple)):
            indices = list(idx)
            single = False
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

        # ensure self.I exists
        if not hasattr(self, 'I'):
            self.I = [0]

        # gather outputs
        specs = [self.stims[i] for i in indices]
        metas = [self.stim_meta[i] for i in indices]
        # subset responses and masks to selected neurons
        resps = []
        masks = []
        for i in indices:
            # original list of responses for all neurons to stimulus i
            all_resps = self.responses[i]
            # original mask tensor of shape (N,)
            all_mask = self.nrn_masks[i]
            # take selected neuron indices
            sel_resps = [all_resps[n] for n in self.I]
            sel_mask = all_mask[self.I]
            resps.append(sel_resps)
            masks.append(sel_mask)

        if single:
            return specs[0], resps[0], masks[0], metas[0]
        return specs, resps, masks, metas

    def select_animal(self, animal_ids):
        """
        Select neurons recorded from the given animal_ids.
        animal_ids: list or tuple of strings
        Updates self.I to indices of matching neurons.
        """
        # find indices of neurons whose meta[0] (animal) is in the list
        selected = []
        for idx, (animal, _, _) in enumerate(self.nrn_meta):
            if animal in animal_ids:
                selected.append(idx)
        self.I = selected



