import gc
import os
from tqdm import tqdm

import torch
import pandas as pd

try:
    from nems0.recording import load_recording
    from nems0 import xforms, preprocessing, epoch
    _NEMS_AVAILABLE = True
except ImportError:
    _NEMS_AVAILABLE = False

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset


_NEMS_INSTALL_HINT = (
    "NAT4_Dataset needs the NEMS0 library to read its .tgz recording format. "
    "Install it with:\n"
    "    pip install 'deepSTRF[nems]'\n"
    "See https://github.com/LBHB/NEMS0 for details."
)


# TODO:
#  1. load both A1 and PEG at the same time, then select neurons by area with the provided API ?
#  2. (stim, nrn) pairs with null (R, T) responses --> (R=1, T=1) nan ?
#  3. smooth resps ?
#  4. dt_ms ? what determines the size of spectrogram time bins in NEMS ??
#  5. integrate to NAT4 constructor --> remove preprocessing_script
#  6. update README



class NAT4_Dataset(AudioNeuralDataset):
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

        for population training.


        """
    def __init__(self, path: str, area='A1'):
        """
        Initializes the NAT4Dataset.

        Parameters:
            path (str): Path to the folder containing the datafiles.
            area (str): the cortical area of the recordings, 'A1' or 'PEG'
        """

        if not _NEMS_AVAILABLE:
            raise ImportError(_NEMS_INSTALL_HINT)

        super().__init__(path)
        assert area == "A1" or area == "PEG", f"Unexpected value '{area}' for argument 'area', choose between 'A1' or 'PEG'"

        # =========  LOAD THE DATA  ===========

        datafile = path + f'/{area}_NAT4_ozgf.fs100.ch18.tgz'
        rec = load_recording(datafile)

        context = {'rec': rec}
        context.update(xforms.normalize_sig(sig='stim', norm_method='minmax', log_compress=1, **context))  # normalize spectrograms (log-compression, important)
        context.update(xforms.normalize_sig(sig='resp', norm_method='minmax', **context))  # normalize responses
        context.update(preprocessing.split_pop_rec_by_mask(**context))

        cells = context['rec']['resp'].chans
        val_sounds = epoch.epoch_names_matching(context['rec']['resp'].epochs, "^STIM_00cat")
        est_sounds = epoch.epoch_names_matching(context['rec']['resp'].epochs, "^STIM_cat")

        # =========  EXTRACT STIMULUS SPECTROGRAMS  ===========

        # [(S_est + S_val) * {'uid': str, 'subset': str}
        self.stim_meta = []

        est_spectros = []
        for est_sound in est_sounds:
            est_spectro = context['rec']['stim'].extract_epoch(est_sound)
            est_spectros.append(torch.from_numpy(est_spectro))
            self.stim_meta.append({'uid': est_sound, 'subset': 'est'})
        est_spectros = torch.stack(est_spectros)  # (575, 1, 18, 150) = (S, 1, F, T)

        val_spectros = []
        for val_sound in val_sounds:
            val_spectro = context['rec']['stim'].extract_epoch(val_sound)  # (1, F=18, T=150)
            val_spectros.append(torch.from_numpy(val_spectro))
            self.stim_meta.append({'uid': val_sound, 'subset': 'val'})
        val_spectros = torch.stack(val_spectros)  # (18, 1, 18, 150) = (S, 1, F, T)

        self.stims = torch.cat([est_spectros, val_spectros], dim=0)  # (S_est + S_val, 1, F, T)

        # =========  MASK FOR 'AUDITORY RESPONSIVE' NEURONS  ===========

        # [N * {'uid': str, 'area': str, 'auditory': bool}]
        self.pop_metadata = []

        # register the "auditory responsiveness" of neurons, pre-determined by the dataset's authors
        list_neurons = pd.read_csv(path + f'/{area}_pred_correlation.csv')
        for cell in cells:
            cell_auditory = list_neurons.loc[list_neurons['cellid'] == cell]['sig_auditory'].item()
            self.pop_metadata.append({'uid': cell, 'area': area, 'auditory': cell_auditory})

        # =========  EXTRACT CORRESPONDING RESPONSE TRIALS (ESTIMATION SET) ===========
        # estimation stimuli were presented only once, sequentially

        est_responses = []
        for est_sound in est_sounds:
            est_resp = context['rec']['resp'].extract_epoch(est_sound)  # (R=1, N, T=150), N=849 for A1 and N=398 for PEG
            est_responses.append(torch.from_numpy(est_resp))
        est_responses = torch.cat(est_responses)  # (575, N, 150) = (S, N, T)
        est_responses = est_responses.unsqueeze(2)  # (S, N, R=1, T)

        # =========  EXTRACT CORRESPONDING RESPONSE TRIALS (VALIDATION SET)  ===========

        val_responses = []
        val_cells = []  # variable to keep track of cells' order as we browse through val files

        del rec
        del context
        gc.collect()

        FILES_LIST = os.listdir(os.path.join(path, f'{area}_single_sites/'))

        for filename in tqdm(FILES_LIST):

            # ignore 'TNCxxx' cells since they do not have est set responses
            if 'TNC' in filename:
                print(f"skipping {filename} (no est set responses)...")
                continue

            datafile = path + f'/{area}_single_sites/' + filename
            single_site_rec = load_recording(datafile)

            val_cells += single_site_rec['resp'].chans  # progressively adds units, but in a different order as in the 'cells' variable
            responses = []

            for val_sound in val_sounds:
                resp = single_site_rec['resp'].rasterize()  # (N_subpop, T)
                subpop_resp = resp.extract_epoch(val_sound)  # (R=20, N_subpop, T_stim_ms=1500)
                R, N_subpop, T_stim_ms = subpop_resp.shape
                subpop_resp = subpop_resp.reshape(R, N_subpop, -1, 10).sum(axis=-1)  # (R, N_subpop, N_timebins/10): dt=1ms --> dt=10ms
                responses.append(torch.from_numpy(subpop_resp))

                del resp
                gc.collect()

            responses = torch.stack(responses)  # (S, R, N_subpop, T)
            val_responses.append(responses)

            del single_site_rec
            gc.collect()

        val_responses = torch.cat(val_responses, dim=2)  # (S, R, N, T)
        val_responses = val_responses.permute(0, 2, 1, 3)  # (S, N, R, T)

        # at this point responses of all neurons to all val stims are registered (cf. val_responses.shape)
        # but the order of cells in the neuron dimension differs from est responses. In other words:
        #  - cells == debug_cells --> False
        #  - set(cells) == set(debug_cells) --> True
        # So we need to reorder cells in this dimension to match that in 'est'
        index_map = {u: i for i, u in enumerate(val_cells)}
        perm_indices = [index_map[u] for u in cells]  # length N
        perm_tensor = torch.tensor(perm_indices, dtype=torch.long)
        val_responses = val_responses.index_select(dim=1, index=perm_tensor)
        assert [val_cells[i] for i in perm_indices] == cells

        # final responses attribute
        self.responses = [r for r in est_responses] + [r for r in val_responses]

        # TODO: check (stim, nrn) pairs with null responses --> nan (using special method) ?
        # val_responses.mean(dim=(2, 3)) --> (S, N)
        # val_responses.mean(dim=(2, 3)).count_nonzero() / val_responses.mean(dim=(2, 3)).numel()

        # neuron mask attribute. TODO: 1) use compute_nrn_masks() method ? 2) make this attribute obsolete soon ?
        # for the moment, consider that all neurons had valid responses (even null ones) to all stims, for this dataset
        self.nrn_masks = torch.ones(len(self.stim_meta), len(self.pop_metadata)).bool()  # (S, N)

        # general attributes
        self.area = area
        self.N_neurons = len(self.pop_metadata)
        self.I = list(range(self.N_neurons))
        self.dt = 10
        self.F = 18
        self.species = 'ferret'


    @staticmethod
    def replace_null_resps_by_nan_placeholder(responses):
        """
        Some stimuli elicit no spikes to some neurons in any trial at all: these are null responses.
        They take the same amount of memory as other responses as an (R, T) tensor full of zeros.
        This function replaces them by a (1, 1) NaN placeholder, in conformity with deepSTRF's guidelines.

        # TODO: also discard null trials ???
        """

        final_resps = []
        S_val, N, _, _ = responses.shape
        mask = responses.mean(dim=(2, 3)) > 0.
        for s in range(S_val):

            temp_stim_resps = []

            for n in range(N):
                # if at least one spike was elicited for this neuron by this stim, keep the responses
                if mask[s, n]:
                    temp_stim_resps.append(responses[s, n])
                # if no response elicited at all, nan placeholder instead
                else:
                    temp_stim_resps.append(torch.full((1, 1), fill_value=torch.nan))

            final_resps.append(temp_stim_resps)