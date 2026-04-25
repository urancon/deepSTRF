import os
import torch
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._crcns_aa_loaders import load_spike_file, parse_cell_name


# TODO (misc.):
#  - find a way to use pre-onset activity ?
#  - make concatenable ?


def get_animals_ids(data_path):
    """
    Takes in the path of the 'CRCNS_AA1/data/' folder, goes through 'Field_L/' and 'MLd', and outputs a list of unique
     animal ids, which are the string preceding the first underscore of each subfolder.
     e.g., 'gg0304_4_B' --> 'gg0304'
    """
    animal_ids = []
    for area_folder in ['Field_L_cells', 'MLd_cells']:
        cell_names = sorted(os.listdir(os.path.join(data_path, area_folder)))
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
        cell_names = sorted(os.listdir(os.path.join(data_path, area_folder)))
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
        stim_names = sorted(os.listdir(os.path.join(data_path, stim_folder)))
        for stim_name in stim_names:
            stim_dict[stim_type].append(stim_name)
    return stim_dict


class CRCNS_AA1_Dataset(AudioNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the CRCNS-AA1 dataset.


    =============== SOURCE ================

    See original papers for details:
     - "Tuning for Spectro-temporal Modulations: a Mechanism for Auditory Discrimination of Natural Sound" by Woolley et al. (2005)
     - "Modulation and phase spectrum of natural sounds enhance neural discrimination performed by single auditory neurons" by Hsu et al. (2004)
     - "Modulation spectra of natural sounds and ethological theories of auditory processing" by Singh and Theunissen (2003)

    Data available at: https://crcns.org/data-sets/aa/aa-1/about


    =============== DETAILS ================

    More details can be found in the dataset source, our dedicated readme file, or in the original papers.
    But in a nutshell:
     - extracellular, spike-sorted single units of anesthetized male zebra finches
     - 50 cells in field L, 50 in MLd
     - 10 clips of conspecific vocalizations and 20 clips of flat ripples, up to 5 s duration.
     - 10 trials on average


    =============== STRUCTURE ================

    Follows the standard deepSTRF data paradigm (see docs/_source/md/data_paradigm.md).
    AA1-specific metadata contents:
     - self.stims                       list of S tensors (1, F, T_s), mel-spectrograms
     - self.responses                   list of S lists of N tensors (R_{s,n}, T_s)
     - self.stim_meta                   list of S dicts {"name", "type",
                                        "sample_rate", "n_samples", "duration_s"}
     - self.neuron_metadata             list of N dicts {"cell_id", "animal_id",
                                        "area", "cell_seq", "rig"} — cell_seq is
                                        the sequential cell index parsed from the
                                        cell folder name (per AA1 readme PDF: the
                                        n-th cell recorded); rig is the single-
                                        letter rig label when present, else None
                                        (cells "4_A" and "4_B" were recorded
                                        simultaneously, possibly in different
                                        brain areas)


    =============== REMARKS ================

    only cell 'pipu1018_2_A' in 'MLd' does not have responses to 'conspecific' stims
    only cell 'pipu1018_2_B' in 'Field_L' does not have responses to 'conspecific' stims

    """

    def __init__(self, path: str, areas=('Field_L', 'MLd'), stimuli=('conspecific', 'flatrip'), animals='all', dt_ms=1, smooth=True, n_mels=32, compression='cubic'):
        """
        Initializes the AA1 Dataset

        Specific units can be selected according to the stimulus they were presented, their animal, and recording site.

        Parameters:
            path (str): Path to the 'CRCNS_AA2/data/' folder containing files as indicated in our readme
            areas (tuple of str): recording sites of interest, can be 'Field_L' or 'MLd'
            stimuli (tuple of str): stimulus types of interest, can be 'conspecific' or 'flatrip'
            dt (float): time step size in ms
            n_mels (int): number of mel frequency bands the stimulus should have in spectrogram form
            compression: compression function to apply to the stimulus spectrogram
        """

        super().__init__(path, dt_ms)

        # general
        self.species = 'zebra finch'
        self.behavioral_state = 'anesthetized'

        # hop_length (samples) | dt (ms)
        # 320 | 10
        # 160 | 5
        # 32  | 1
        self.F = n_mels
        hl = dt_ms * 32
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=10 * hl, hop_length=hl, n_mels=self.F)  # n_fft=800
        self.compression = compression

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

        # filter cells by animal
        if animals == 'all':
            self.animals = ANIMAL_IDs
        else:
            self.animals = animals
        cells = [cell for cell in cells if cell.split('_')[0] in self.animals]
        cell_animals = [cell.split('_')[0] for cell in cells]

        # filter stimuli by type
        stims = []
        stim_types = []
        for stim_type in stimuli:
            stims += STIM_IDs[stim_type]
            stim_types += [stim_type] * len(STIM_IDs[stim_type])

        # filter cells by stimulus type:
        # remove cells if they don't have any stimulus of the required type
        for area in ['Field_L', 'MLd']:
            area_path = os.path.join(path, f'{area}_cells')
            for cell in sorted(os.listdir(area_path)):
                # if the cell is in the current selection of cells, check if it has responses to at least one stim type
                if cell in cells:
                    cell_path = os.path.join(area_path, cell)
                    cell_stim_types = sorted(os.listdir(cell_path))
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

        self.stims = []             # --> list of S tensors of shape (1, F, T_s)
        self.responses = []         # --> list of S lists of N tensors of shape (R_{s,n}, T_s)
        self.stim_meta = []         # --> list of S dicts {name, type}
        stim_meta = list(zip(stims, stim_types))
        # list of N dicts; cell_seq + rig parsed from the documented
        # AA1 cell-name format (<animal>_<cell_seq>[_<rig>], cf. AA1 readme PDF)
        self.neuron_metadata = []
        for c, a, r in zip(cells, cell_animals, cell_areas):
            _, cell_seq, rig = parse_cell_name(c)
            self.neuron_metadata.append({
                "cell_id": c,
                "animal_id": a,
                "area": r,
                "cell_seq": cell_seq,
                "rig": rig,
            })

        for s, (stim_name, stim_type) in enumerate(stim_meta):

            # =========== load the stim ============

            stims_dir = os.path.join(path, f"all_stims/{stim_type}/")

            wav, sr = torchaudio.load(os.path.join(stims_dir, stim_name), normalize=True)  # sample rate: 32 kHz (mono)
            assert sr == 32000, f"found wav sr of {sr}, expected 32000"
            n_samples_wav = wav.shape[-1]
            spec = transform(wav)  # (T,) --> (1, F, T-)
            if self.compression == 'cubic':
                spec = torch.pow(spec, 1.0 / 3)
            elif self.compression == 'log1p':
                spec = torch.log1p(spec)
            elif self.compression == 'none':
                pass
            T = spec.shape[-1]  # nbr of timesteps of current stim, in spectrogram form

            # =========== load the resps ============

            pop_resps = []
            no_data_nrn_idces = []

            for n, nrn in enumerate(self.neuron_metadata):
                cell_name = nrn["cell_id"]
                area = nrn["area"]

                # some cells may not have any response for the current stim type;
                # if that is the case --> null response directly
                if stim_type not in os.listdir(os.path.join(path, f'{area}_cells/', cell_name)):
                    resp = torch.full((1,1), fill_value=float('nan'))
                    no_data_nrn_idces.append(n)

                # if they do have responses to this stim_type:
                else:

                    #  1. find the stim file corresponding to the stim_name
                    #  2. find the spike file corresponding to that stim file, if any
                    spike_dir = os.path.join(path, f'{area}_cells/', cell_name, stim_type)
                    no_stim = True
                    stim_files = sorted(file for file in os.listdir(spike_dir) if 'stim' in file)
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
                        resp = torch.full((1,1), fill_value=float('nan'))
                        no_data_nrn_idces.append(n)

                    #  3. if they do indeed have a response to this specific stim, get the response
                    else:
                        spike_file = os.path.join(spike_dir, f'spike{i}')
                        try:
                            resp = load_spike_file(spike_file, dt_ms=self.dt)  # (R, T)

                            # align response time dim to the stimulus duration T:
                            #   shorter: right-pad with zeros (no spikes)
                            #   longer:  crop (post-stimulus spikes discarded)
                            # TODO: alternatively, keep post-stim spikes and pad the spectrogram to match
                            if resp.shape[-1] <= T:
                                Pt = T - resp.shape[-1]
                                resp = torch.nn.functional.pad(resp, pad=(0, Pt), mode='constant', value=0.)
                            else:
                                resp = resp[:, :T]

                        # some neurons have a 'stimXX' file, but not the corresponding 'spikeXX' file
                        except FileNotFoundError:
                            resp = torch.full((1,1), fill_value=float('nan'))
                            no_data_nrn_idces.append(n)

                # add the cell's response, whether it is null or not, to the population activity for this stim
                # at the end of the for loop on cells, pop_resps is [N * (R_n, T_ns)]
                pop_resps.append(resp)

            # assert population response is well-formed
            assert len(pop_resps) == self.N_neurons

            # if none of the neurons heard this stim at all, skip it
            if len(no_data_nrn_idces) == self.N_neurons:
                continue

            # otherwise keep the stim and its per-neuron responses
            self.stims.append(spec)
            self.responses.append(pop_resps)
            self.stim_meta.append({
                "name": stim_name,
                "type": stim_type,
                "sample_rate": float(sr),
                "n_samples": int(n_samples_wav),
                "duration_s": n_samples_wav / float(sr),
            })

        # smooth PSTHs with a 21 ms Hanning window (Hsu / Borst / Theunissen 2004)
        if smooth:
            self.smooth_responses(window_ms=21.0)

        # self.nrn_masks is a derived @property on the base class — no need
        # to populate it here
        self.validate()
