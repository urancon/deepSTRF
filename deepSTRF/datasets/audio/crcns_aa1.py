import os
from typing import Optional

import torch
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._crcns_aa_loaders import load_spike_file, parse_cell_name
from deepSTRF.utils.audio_io import load_wav
from deepSTRF.utils.data_download import crcns_download, default_cache_dir, unzip


# TODO (misc.):
#  - find a way to use pre-onset activity ?
#  - make concatenable ?


# CRCNS-AA1 ships as a single zip archive on the NERSC mirror.
# After unzip -> 'all_stims/', 'Field_L_cells/', 'MLd_cells/' at the
# directory root (flat — no wrapping top-level folder).
AA1_NERSC_PATH = "aa-1/crcns-aa1.zip"


def download_aa1(dest: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None) -> str:
    """Download the CRCNS-AA1 archive from the NERSC mirror into ``dest``.

    Idempotent: skips the archive if already present, and skips unzipping
    if ``Field_L_cells/`` already exists in ``dest``. Returns the dataset
    directory.

    Parameters
    ----------
    dest : str, optional
        Defaults to the platformdirs cache (overridable via
        ``$DEEPSTRF_DATA_DIR``).
    username, password : str, optional
        Default to ``$CRCNS_USERNAME`` / ``$CRCNS_PASSWORD``. Free CRCNS
        account at https://crcns.org/register.
    """
    dest_path = str(default_cache_dir("AA1") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)

    zip_path = os.path.join(dest_path, "crcns-aa1.zip")
    if not os.path.exists(zip_path):
        crcns_download(AA1_NERSC_PATH, zip_path,
                       username=username, password=password)

    if not os.path.isdir(os.path.join(dest_path, "Field_L_cells")):
        unzip(zip_path, dest_path, strip_root=True)

    return dest_path


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


class CRCNSAA1Dataset(AudioNeuralDataset):
    """PyTorch dataset for the CRCNS-AA1 recordings.

    Extracellular, spike-sorted single units of anesthetized male zebra
    finches: 50 cells in Field L and 50 in MLd, recorded in response to 10
    clips of conspecific vocalizations and 20 clips of flat ripples (up to
    5 s each, ~10 trials on average). Data are available at
    https://crcns.org/data-sets/aa/aa-1/about (free CRCNS account); see the
    AA1 README in the deepSTRF docs for the full notes.

    Notes
    -----
    Follows the standard deepSTRF data paradigm (see
    ``docs/_source/md/data_paradigm.md``). AA1-specific metadata:

    - ``stims`` are mel-spectrograms ``(1, F, T_s)``.
    - ``stim_meta`` dicts hold ``name``, ``type``, ``sample_rate``,
      ``n_samples`` and ``duration_s``.
    - ``nrn_meta`` dicts hold ``cell_id``, ``animal_id``, ``area``,
      ``cell_seq`` and ``rig``. ``cell_seq`` is the sequential cell index
      parsed from the cell folder name (the n-th cell recorded); ``rig`` is
      the single-letter rig label when present, else ``None`` (cells "4_A"
      and "4_B" were recorded simultaneously, possibly in different areas).

    Two cells lack ``conspecific`` responses: ``pipu1018_2_A`` (MLd) and
    ``pipu1018_2_B`` (Field_L).

    References
    ----------
    Woolley et al. (2005). "Tuning for Spectro-temporal Modulations: a
    Mechanism for Auditory Discrimination of Natural Sound."

    Hsu et al. (2004). "Modulation power and phase spectrum of natural
    sounds enhance neural encoding performed by single auditory neurons."

    Singh & Theunissen (2003). "Modulation spectra of natural sounds and
    ethological theories of auditory processing."
    """

    def __init__(self, path: Optional[str] = None, areas=('Field_L', 'MLd'),
                 stimuli=('conspecific', 'flatrip'), animals='all', dt_ms=1,
                 smooth=True, n_mels=32, compression='cubic',
                 window_ms: float = 10.0,
                 return_waveform: bool = False, audio_fs: int = 32000,
                 download: bool = False,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initializes the AA1 Dataset.

        Parameters
        ----------
        path : str, optional
            Path to the AA1 data folder (containing ``Field_L_cells/``,
            ``MLd_cells/``, ``all_stims/``). Defaults to the platformdirs
            cache (``$DEEPSTRF_DATA_DIR`` overrides).
        areas : tuple of str
            Recording sites: 'Field_L' or 'MLd'.
        stimuli : tuple of str
            Stimulus types: 'conspecific' or 'flatrip'.
        dt_ms : float
            Time step size in ms.
        n_mels : int
            Number of mel frequency bands.
        compression : str
            Spectrogram compression ('cubic', 'log1p', 'none'). Ignored when
            ``return_waveform=True``.
        window_ms : float, default 10.0
            FFT analysis-window length in ms. ``n_fft`` is computed as
            ``round(window_ms * 1e-3 * sample_rate)`` and is **decoupled
            from ``hop_length``** so phonemic detail is preserved at any
            ``dt_ms``. Earlier versions of this dataset hardcoded
            ``n_fft = 10 * hop_length`` — benign at the default
            ``dt_ms=1`` (10 ms FFT window), but at ``dt_ms=50`` the same
            formula produced a 500 ms FFT window and over-smoothed every
            spec frame. The default ``window_ms=10.0`` preserves
            bit-identical behaviour at ``dt_ms=1`` while removing the
            scaling bug at coarser bins. Speech-pipeline users may
            prefer ``window_ms=25.0`` (Kaldi default). Ignored when
            ``return_waveform=True``.
        return_waveform : bool, default False
            If True, ``self.stims[s]`` holds the raw audio waveform
            ``(1, T_audio)`` at ``audio_fs`` Hz (grid-locked to ``T_audio =
            T_neural * hop``) instead of the in-loader mel spectrogram. Pair
            with a model whose ``wav2spec`` slot is a waveform front-end (see
            ``deepSTRF.models.wav2spec``); responses are unchanged.
        audio_fs : int, default 32000
            Sample rate for waveform mode. Default 32 kHz is the native rate of
            the AA1 wavs (so no resampling); other values resample and must keep
            ``audio_fs * dt_ms / 1000`` an integer. Ignored unless
            ``return_waveform=True``.
        download : bool, default False
            If True and the data is missing under ``path``, fetch the
            ~17 MB CRCNS-AA1 archive from the NERSC mirror (free CRCNS
            account required; see ``crcns_download``) and unzip in place.
        username, password : str, optional
            CRCNS credentials. Default to ``$CRCNS_USERNAME`` /
            ``$CRCNS_PASSWORD``. Prefer the env vars over passing
            literals — anything in source / a notebook ends up in
            history / logs / VCS.
        """

        if path is None:
            path = str(default_cache_dir("AA1"))
        if download:
            download_aa1(path, username=username, password=password)

        super().__init__(path, dt_ms)

        # general
        self.species = 'zebra finch'
        # Informational hearing range (zebra finch behavioural audiogram ≈ 250 Hz – 8 kHz).
        self.hearing_range_hz = (250.0, 8000.0)
        # Waveform-input mode: store raw audio instead of the in-loader mel spec.
        self.return_waveform = bool(return_waveform)
        self.audio_fs = int(audio_fs) if return_waveform else None
        self.behavioral_state = 'anesthetized'

        # sr = 32 kHz (CRCNS-AA1 wavs). hop_length tracks dt_ms; n_fft is
        # **independent of dt_ms** and pinned to ``window_ms``. At the
        # default (window_ms=10, dt_ms=1) this gives n_fft=320, which
        # equals the legacy ``10 * hl`` value bit-for-bit — no behaviour
        # change at the historical default. At coarser dt_ms the new
        # formula yields a sensible window (n_fft=320 at dt_ms=50 vs the
        # legacy 16000). See ``window_ms`` docstring above.
        sample_rate = 32000
        self.F = n_mels
        self.window_ms = float(window_ms)
        hl = int(dt_ms * 32)
        # ``n_fft`` is derived from ``hop`` (already truncated to an int)
        # via the ratio ``window_ms / dt_ms`` so the default
        # ``window_ms = 10.0`` reproduces the legacy ``n_fft = 10 * hl``
        # value bit-for-bit at every sr supported by AA1 (32 kHz fixed
        # here, but the same trick is used in AA4 where sr varies and
        # ``hop`` is truncated). Floored at ``hl`` so the
        # ``n_fft >= hop_length`` STFT constraint always holds.
        n_fft = max(int(round((self.window_ms / float(dt_ms)) * hl)), hl)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hl, n_mels=self.F,
        )
        self.compression = compression
        # audio-samples-per-neural-bin for waveform mode (== hl when audio_fs is
        # the native 32 kHz). Grid-lock: audio_fs * dt_ms / 1000 must be integer.
        hop = int(round(audio_fs * dt_ms / 1000)) if return_waveform else None

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

            stims_dir = os.path.join(path, f"all_stims/{stim_type}/")

            wav, sr = load_wav(os.path.join(stims_dir, stim_name))  # sample rate: 32 kHz (mono)
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

            for n, nrn in enumerate(self.nrn_meta):
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

            # otherwise keep the stim and its per-neuron responses. In waveform
            # mode store the raw audio (grid-locked to T * hop samples so it
            # aligns with the T response bins) instead of the in-loader mel spec.
            if return_waveform:
                T_audio = T * hop
                w = wav if audio_fs == sr else torchaudio.functional.resample(wav, sr, audio_fs)
                if w.shape[-1] < T_audio:
                    w = torch.nn.functional.pad(w, (0, T_audio - w.shape[-1]))
                else:
                    w = w[..., :T_audio]
                self.stims.append(w.contiguous().float())
            else:
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
