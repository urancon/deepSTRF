import os
import pandas as pd
import csv
import torch
import torchaudio

from typing import Optional

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._crcns_aa_loaders import load_spike_file, parse_cell_name
from deepSTRF.utils.data_download import (
    crcns_download,
    default_cache_dir,
    untar,
)


# CRCNS-AA2 ships as 3 tar.gz archives on the NERSC mirror, all wrapping
# their content in ``crcns/aa2/`` (strip 2 components on extract). After
# extraction the layout is the flat one CRCNSAA2Dataset expects:
#   <dest>/all_cells/<cell>/<stim_type>/{spike*, stim*}
#   <dest>/all_stims/*.wav
#   <dest>/{cell_regions.csv, cell_stim_classes.csv, stim_data.csv,
#           crcns-aa2-README.txt}
AA2_NERSC_FILES = (
    "aa-2/crcns-aa2-docs.tar.gz",
    "aa-2/crcns-aa2-all_cells.tar.gz",
    "aa-2/crcns-aa2-all_stims.tar.gz",
)


def download_aa2(dest: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None) -> str:
    """Download the CRCNS-AA2 archives from the NERSC mirror into ``dest``.

    Idempotent: skips an archive if already on disk, and skips extraction
    of an archive if its anchor sub-tree (``all_cells/``, ``all_stims/``,
    or ``stim_data.csv``) already exists.

    Parameters
    ----------
    dest : str, optional
        Defaults to ``default_cache_dir('AA2')`` (overridable via
        ``$DEEPSTRF_DATA_DIR``).
    username, password : str, optional
        Default to ``$CRCNS_USERNAME`` / ``$CRCNS_PASSWORD``.
    """
    dest_path = str(default_cache_dir("AA2") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)

    # anchor file/dir per archive — if it exists, the archive has already
    # been extracted, so we skip both download and extraction.
    extraction_anchors = {
        "crcns-aa2-docs.tar.gz":      "stim_data.csv",
        "crcns-aa2-all_cells.tar.gz": "all_cells",
        "crcns-aa2-all_stims.tar.gz": "all_stims",
    }
    for nersc_path in AA2_NERSC_FILES:
        archive_name = os.path.basename(nersc_path)
        archive_path = os.path.join(dest_path, archive_name)
        anchor = extraction_anchors[archive_name]
        if os.path.exists(os.path.join(dest_path, anchor)):
            continue
        if not os.path.exists(archive_path):
            crcns_download(nersc_path, archive_path,
                           username=username, password=password)
        untar(archive_path, dest_path, strip_components=2)

    return dest_path


# TODO (misc.):
#  - some PSTHs have very high peaks (> 20) on some stims (songrips)  --> double check
#  - make concatenable to AA1 Dataset ?


def get_animals_ids(file_path):
    """
    Extracts unique animal identifiers from the first column of the 'cell_stim_classes.csv' file.
    The unique identifier is defined as the substring preceding the first underscore in the first column.
    The output is a list of unique identifiers.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Extract the first column
    first_column = df.iloc[:, 0]

    # Extract the substring preceding the first underscore and find unique values
    unique_identifiers = first_column.str.split('_').str[0].unique()

    return list(unique_identifiers)


def get_stims_ids_from_csv(file_path):
    """
    Extracts .wav file names from the first column of the 'stim_data.csv' file, and classify them into stimulus types.
    The output is a dictionary with categories as keys and lists of .wav file names as values.
    """
    stim_dict = {
        "songrip": [],
        "flatrip": [],
        "conspecific": [],
        "unknown": [],
        "bengalese": []
    }

    stim_count = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # Ensure the row has enough columns
                category = row[-1].strip().lower()
                wav_file = row[0].strip()

                # Match the category with the keys in the dictionary
                if category in stim_dict:
                    stim_dict[category].append(wav_file)
                    stim_count += 1
                else:
                    print(f"Unknown category '{category}' found in the file. Skipping...")

    #print(f"Number or stimuli: {stim_count}")

    return stim_dict


def load_stim_data_csv(file_path):
    """Read CRCNS-AA2 ``stim_data.csv`` into a ``{wav_filename: {...}}`` dict.

    Each value is ``{"sample_rate": Hz, "bit_depth": int, "n_samples": int,
    "duration_s": float}``. Returned even for stims classified as "unknown" /
    "bengalese", since the dataset class will simply not select those by
    default.
    """
    info = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 5:
                continue
            wav, sr_str, bd_str, n_str, _cls = (c.strip() for c in row[:5])
            sr = float(sr_str)
            n_samples = int(n_str)
            info[wav] = {
                "sample_rate": sr,
                "bit_depth": int(bd_str),
                "n_samples": n_samples,
                "duration_s": n_samples / sr if sr > 0 else float("nan"),
            }
    return info


def get_area_cells(file_path):
    """
    From the cell_regions.csv file, returns a dictionary with area labels as keys and lists of cell names as values.
    """
    cell_dict = {
        'L': [],    # e.g., '[pupu2122_2_A, pupu2122_2_B, ...]
        'L1': [],
        'L2a': [],
        'L2b': [],
        'L3': [],
        'mld': [],
        'OV': [],
        'CM': [],
        'None': []
    }

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:  # Ensure the row has enough columns
                site = row[-1].strip()
                cell = row[0].strip()
                cell_dict[site].append(cell)

    return cell_dict

def get_stim_ids_from_folders(cells_path, verbose=False):
    """
        needs the 'all_cells/' path

        returns a dictionary with the three main stim_types 'consepcific', 'songrip' and 'flatrip' as keys, and a
        list of unique wav names for each value

    """
    stim_dict = {
        "songrip": [],
        "flatrip": [],
        "conspecific": []
    }

    for stim_type in stim_dict.keys():

        for cell in sorted(os.listdir(cells_path)):

            if stim_type not in os.listdir(os.path.join(cells_path, cell)):
                if verbose:
                    print(f"no {stim_type} stim for cell {cell}, skipping...")
            else:
                stimfiles = sorted(file for file in os.listdir(os.path.join(cells_path, cell, stim_type)) if 'stim' in file)
                for stimfile in stimfiles:
                    with open(os.path.join(cells_path, cell, stim_type, stimfile)) as f:
                        wavname = f.readlines()[0][:-1]
                        if wavname not in stim_dict[stim_type]:
                            stim_dict[stim_type].append(wavname)

    return stim_dict


class CRCNSAA2Dataset(AudioNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the CRCNS-AA2 dataset and its many recording sites (OV, Mld, Field L, CM)


    =============== SOURCE ================

    See original papers for details:
     - "Sound representation methods for spectro-temporal receptive field estimation" by Patrick Gill et al. (2006)
     - "Role of the Zebra Finch Auditory Thalamus in Generating Complex Representations for Natural Sounds" by Noopur Amin et al. (2010)

    Data available at: https://crcns.org/data-sets/aa/aa-2/about


    =============== DETAILS ================

    More details can be found in the dataset source, the dataset-specific README in the deepSTRF docs, or in the original papers.
    But in a nutshell:
    - 494 extracellular, spike-sorted single units of male zebra finches
    - neurons identified in OV, MLd, Field L, L1, L2a, L2b, L3, OV. Also neurons with unindentified area (None)
    - 3 stimulus classes: conspecific songs (72 stims), flat ripples (20), and song ripples (25)
    - almost all cells were presented conspecific and songrip stimuli, and about half were presented flatrip
    - stimuli were each presented 10-20 times
    - low trial-to-trial variability
    - population fitting-compatible


    =============== STRUCTURE ================

    Follows the standard deepSTRF data paradigm (see docs/_source/md/data_paradigm.md).
    AA2-specific metadata contents:
     - self.stims                       list of S tensors (1, F, T_s), mel-spectrograms
     - self.responses                   list of S lists of N tensors (R_{s,n}, T_s)
     - self.stim_meta                   list of S dicts {"name", "type",
                                        "sample_rate", "n_samples", "duration_s"}
                                        (last three from data/stim_data.csv)
     - self.nrn_meta             list of N dicts {"cell_id", "animal_id",
                                        "area", "cell_seq", "rig"} — see AA1's
                                        docstring for the cell-name format
                                        documentation; rig is often None in AA2

    """
    def __init__(self, path: Optional[str] = None,
                 areas=('Field_L', 'mld', 'OV', 'CM', 'None'),
                 stimuli=('conspecific', 'flatrip', 'songrip'),
                 animals='all', dt_ms=1, smooth=True, n_mels=32,
                 compression='cubic',
                 window_ms: float = 10.0,
                 download: bool = False,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initializes the AA2 Dataset.

        Parameters
        ----------
        path : str, optional
            Path to the AA2 data folder. Defaults to the platformdirs cache.
        areas : tuple of str
            Recording sites of interest: 'Field_L', 'L1', 'L2a', 'L2b',
            'L3', 'mld', 'OV', 'CM', or 'None'.
        stimuli : tuple of str
            Stimulus types of interest: 'conspecific', 'flatrip', 'songrip'.
        dt_ms : float
            Time step size in ms.
        n_mels : int
            Number of mel frequency bands.
        compression : str
            Spectrogram compression ('cubic', 'log1p', 'none').
        window_ms : float, default 10.0
            FFT analysis-window length in ms. ``n_fft`` is computed as
            ``round(window_ms * 1e-3 * sample_rate)`` and is **decoupled
            from ``hop_length``** so phonemic detail is preserved at any
            ``dt_ms``. Earlier versions of this dataset hardcoded
            ``n_fft = 10 * hop_length``, which gave a benign 10 ms FFT
            window at ``dt_ms=1`` but a 500 ms window at ``dt_ms=50``.
            Default ``window_ms=10.0`` preserves bit-identical behaviour
            at ``dt_ms=1`` while fixing the scaling bug at coarser bins.
        download : bool, default False
            If True and the data is missing under ``path``, fetch the
            ~30 MB worth of CRCNS-AA2 archives from the NERSC mirror
            (free CRCNS account required) and extract in place.
        username, password : str, optional
            CRCNS credentials. Default to ``$CRCNS_USERNAME`` /
            ``$CRCNS_PASSWORD``. Prefer env vars over passing literals.
        """

        if path is None:
            path = str(default_cache_dir("AA2"))
        if download:
            download_aa2(path, username=username, password=password)

        super().__init__(path, dt_ms)

        self.species = 'zebra finch'
        # sr = 32 kHz (CRCNS-AA2 wavs). hop_length tracks dt_ms; n_fft is
        # **independent of dt_ms** and pinned to ``window_ms``. At the
        # default (window_ms=10, dt_ms=1) this gives n_fft=320, which
        # equals the legacy ``10 * hl`` value bit-for-bit. At coarser
        # dt_ms the legacy formula scaled the window with the hop and
        # over-smoothed the spec; the new formula caps n_fft at
        # ``window_ms * sr`` (or ``hop_length`` if that's larger — STFT
        # constraint). See ``window_ms`` docstring above.
        sample_rate = 32000
        self.F = n_mels
        self.window_ms = float(window_ms)
        hl = int(dt_ms * 32)
        # See the AA1 spec block for why we derive n_fft from the
        # truncated hop via the ratio ``window_ms / dt_ms`` — preserves
        # bit-identical behaviour at the legacy ``window_ms = 10 * dt_ms``
        # contract for every supported sr.
        n_fft = max(int(round((self.window_ms / float(dt_ms)) * hl)), hl)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hl, n_mels=self.F,
        )
        self.compression = compression

        #######################
        # 1. get metadata
        #######################

        # get all animal ids
        ANIMAL_IDs = get_animals_ids(os.path.join(path, 'cell_stim_classes.csv'))  # list of animal ids

        # get all cells for each areas
        AREA_CELLs = get_area_cells(os.path.join(path, 'cell_regions.csv'))  # dict with areas as keys and list of cell names as values

        # get all stims
        STIM_IDs = get_stim_ids_from_folders(os.path.join(path, 'all_cells/'), verbose=False)  # dict with stim types as keys and list of wav names as values

        # per-wav metadata (sample_rate, n_samples, duration_s, bit_depth)
        STIM_INFO = load_stim_data_csv(os.path.join(path, 'stim_data.csv'))


        ################################
        # 2. cells & stims selection
        ################################

        # pre-selection of cell names, based on stim_type, animal, and area
        cells = []

        # filter cells by area
        cell_areas = []
        if areas == 'all':
            areas = ('Field_L', 'OV', 'CM', 'mld', 'None')
        for area in areas:
            if (area == 'Field_L') or (area == 'L'):
                cells += AREA_CELLs['L'] + AREA_CELLs['L1'] + AREA_CELLs['L2a'] + AREA_CELLs['L2b'] + AREA_CELLs['L3']
                cell_areas += (['L'] * len(AREA_CELLs['L']) +
                               ['L1'] * len(AREA_CELLs['L1']) +
                               ['L2a'] * len(AREA_CELLs['L2a']) +
                               ['L2b'] * len(AREA_CELLs['L2b']) +
                               ['L3'] * len(AREA_CELLs['L3']))
            else:
                cells += AREA_CELLs[area]
                cell_areas += ([area] * len(AREA_CELLs[area]))

        # filter cells by animal
        if animals == 'all':
            self.animals = ANIMAL_IDs
        else:
            self.animals = animals
        cells = [cell for cell in cells if cell.split('_')[0] in self.animals]
        cell_animals = [cell.split('_')[0] for cell in cells]

        # filter stimuli by stimulus type
        stims = []
        stim_types = []
        for stim_type in stimuli:
            stims += STIM_IDs[stim_type]
            stim_types += [stim_type] * len(STIM_IDs[stim_type])

        # filter cells by stimulus type:
        # remove cells if they don't have any stimulus of the required type
        for cell in cells:
            i = 0
            for stim_type in stimuli:
                if stim_type not in os.listdir(os.path.join(path, 'all_cells', cell)):
                    i += 1
            if i >= len(stimuli):
                cell_idx = cells.index(cell)
                cells.remove(cell)
                cell_animals.remove(cell.split('_')[0])
                cell_areas.remove(cell_areas[cell_idx])

        self.N_neurons = len(cells)


        ####################################
        # 3. load stim spectros and resps
        ####################################

        stims_dir = os.path.join(path, f"all_stims/")

        self.stims = []             # --> list of S tensors of shape (1, F, T_s)
        self.responses = []         # --> list of S lists of N tensors of shape (R_{s,n}, T_s)
        self.stim_meta = []         # --> list of S dicts {name, type}
        stim_meta = list(zip(stims, stim_types))
        # list of N dicts; cell_seq + rig parsed from the documented AA1/AA2
        # cell-name format (<animal>_<cell_seq>[_<rig>], cf. AA1 readme PDF —
        # AA2 inherits the convention).
        self.nrn_meta = []
        for c, a, r in zip(cells, cell_animals, cell_areas):
            _, cell_seq, rig = parse_cell_name(c)
            self.nrn_meta.append({
                "cell_id": c,
                "animal_id": a,
                "area": r,
                "cell_seq": cell_seq,
                "rig": rig,
            })

        for s, (stim_name, stim_type) in enumerate(stim_meta):

            # =========== load the stim ============

            wav, sr = torchaudio.load(os.path.join(stims_dir, stim_name), normalize=True)  # sample rate: 32 kHz (mono)
            assert sr == 32000, f"found wav sr of {sr}, expected 32000"
            spec = transform(wav)    # (T,) --> (1, F, T-)
            if self.compression == 'cubic':
                spec = torch.pow(spec, 1.0/3)
            elif self.compression == 'log1p':
                spec = torch.log1p(spec)
            elif self.compression == 'none':
                pass
            T = spec.shape[-1]  # nbr of timesteps of current stim, in spectrogram form


            # =========== load the resps ============

            pop_resps = []
            no_data_nrn_idces = []

            for n, cell in enumerate(cells):

                # some cells may not have any response for the current stim type;
                # if that is the case --> null response directly  --> (1, 1) tensor of NaN
                if stim_type not in os.listdir(os.path.join(path, 'all_cells/', cell)):
                    resp = torch.full((1,1), fill_value=float('nan'))
                    no_data_nrn_idces.append(n)

                # if they do have responses to this stim_type:
                else:

                    #  1. find the stim file corresponding to the stim_name
                    #  2. find the spike file corresponding to that stim file, if any
                    spike_dir = os.path.join(path, 'all_cells/', cell, stim_type)
                    no_stim = True
                    stim_files = sorted(file for file in os.listdir(spike_dir) if 'stim' in file)
                    for stim_file in stim_files:
                        i = int(stim_file[4:])   # e.g., 'stim20'  --> '20'
                        with open(os.path.join(spike_dir, stim_file)) as f:
                            wavname = f.readlines()[0][:-1]
                        if wavname == stim_name:
                            no_stim = False
                            break

                    # some neurons may have responses to stims of this type, but not this one in particular;
                    # in this case --> null response  --> (1, 1) tensor of NaN again
                    if no_stim :
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
                        # in this case --> null response  --> (1, 1) tensor of NaN again
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
            wav_info = STIM_INFO.get(stim_name, {})
            self.stim_meta.append({
                "name": stim_name,
                "type": stim_type,
                "sample_rate": wav_info.get("sample_rate"),
                "n_samples": wav_info.get("n_samples"),
                "duration_s": wav_info.get("duration_s"),
            })

        # smooth PSTHs with a 21 ms Hanning window (Hsu / Borst / Theunissen 2004)
        if smooth:
            self.smooth_responses(window_ms=21.0)

        # self.nrn_masks is a derived @property on the base class — no need
        # to populate it here
        self.validate()
