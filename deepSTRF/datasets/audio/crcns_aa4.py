import os
import re
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._crcns_aa_loaders import time_binning
from deepSTRF.utils.data_download import (
    crcns_download,
    default_cache_dir,
    untar,
)


def _get_subgroups(group):
    """Return list of subgroup names under `group` in an h5 File."""
    return [name for name, obj in group.items() if isinstance(obj, h5py.Group)]


def _decode_attr(val) -> str:
    """Decode an h5 string attribute that may be bytes / array-of-bytes / str."""
    try:
        return val.decode()
    except AttributeError:
        try:
            return val[0].decode()
        except (AttributeError, IndexError, TypeError):
            return str(val)


# Filename format from the AA4 PDF:
#   Site<S>_L<Lz>R<Rz>_e<elec>_s<online_sortid>[_ss<offline_sortid>].h5
# (e.g. "Site1_L1400R1400_e10_s0_ss1.h5"); some files omit the trailing _ss<n>.
_AA4_SUBSORT_RE = re.compile(r"_ss(\d+)$")

# Some cells in the data have a typo'd sortType ("singl" instead of "single").
# Normalise so downstream filters don't have to care.
_AA4_SORTTYPE_FIXES = {"singl": "single"}


AA4_ANIMAL_IDS = ('BlaBro09xxF', 'GreBlu9508M', 'LblBlu2028M', 'WhiBlu5396M', 'WhiWhi4522M', 'YelBlu6903F')


def download_aa4(dest: Optional[str] = None,
                 animals: Sequence[str] = AA4_ANIMAL_IDS,
                 username: Optional[str] = None,
                 password: Optional[str] = None) -> str:
    """Download CRCNS-AA4 archives from the NERSC mirror into ``dest``.

    AA4 is split into one ``.tar.gz`` per animal (each is hundreds of MB);
    by default this fetches all 6, but ``animals`` can be narrowed to a
    subset. The CRCNSCode tutorial archive is also fetched (small, ~1 MB).

    Idempotent: skips an archive if its animal directory already exists,
    skips the CRCNSCode archive if ``CRCNSCode/`` already exists.

    Parameters
    ----------
    dest : str, optional
        Defaults to ``default_cache_dir('AA4')`` (``$DEEPSTRF_DATA_DIR``
        overrides).
    animals : sequence of str, default all 6
        Animals to download. Must be a subset of ``AA4_ANIMAL_IDS``.
    username, password : str, optional
        Default to ``$CRCNS_USERNAME`` / ``$CRCNS_PASSWORD``.
    """
    dest_path = str(default_cache_dir("AA4") if dest is None else dest)
    os.makedirs(dest_path, exist_ok=True)

    for animal in animals:
        assert animal in AA4_ANIMAL_IDS, \
            f"Unknown AA4 animal {animal!r}. Valid: {AA4_ANIMAL_IDS}"
        if os.path.isdir(os.path.join(dest_path, animal)):
            continue
        archive_name = f"{animal}.tar.gz"
        archive_path = os.path.join(dest_path, archive_name)
        if not os.path.exists(archive_path):
            crcns_download(f"aa-4/{archive_name}", archive_path,
                           username=username, password=password)
        untar(archive_path, dest_path)  # tarball already wraps in <animal>/

    # CRCNSCode tutorial — small, useful pointer to the original loaders
    code_dir = os.path.join(dest_path, "CRCNSCode")
    if not os.path.isdir(code_dir):
        archive_path = os.path.join(dest_path, "CRCNSCode.tar.gz")
        if not os.path.exists(archive_path):
            crcns_download("aa-4/CRCNSCode.tar.gz", archive_path,
                           username=username, password=password)
        untar(archive_path, dest_path)

    return dest_path


class CRCNSAA4Dataset(AudioNeuralDataset):
    """
    A PyTorch dataset for handling neural data from the CRCNS-AA4 dataset.


    =============== SOURCE ================

    See original papers for details:
     - "Meaning in the avian auditory cortex: Neural representation of communication calls" by Elie JE and Theunissen FE. (2015)
            European Journal of Neuroscience.
     - "Invariant neural responses for sensory categories revealed by the time-varying information for communication calls" by Elie JE and Theunissen FE. (2019)
            Plos Computational Biology.

    Data available at: https://crcns.org/data-sets/aa/aa-4/about-aa-4


    =============== DETAILS ================

    More details can be found in the dataset source, the dataset-specific README in the deepSTRF docs, or in the original papers.
    But in a nutshell:
    - 1401 extracellular, spike-sorted single and multi units of adult zebra finches (4 males, 2 females)
    - Field L, caudolateral and caudomedial mesopallium (CLM and CMM) and caudomedial nidopallium (NCM)
    - units were not precisely assigned one of the above areas
    - 3 stimulus classes: conspecific songs, calls, and ripple noise.
    - stimuli lasted for a few seconds and were each presented ~10 times
    - population fitting-compatible
    - batch-compatible


    =============== STRUCTURE ================

    Follows the standard deepSTRF data paradigm (see docs/_source/md/data_paradigm.md).
    AA4-specific metadata contents:
     - self.stims                       list of S tensors (1, F, T_s), mel-spectrograms
     - self.responses                   list of S lists of N tensors (R_{s,n}, T_s)
     - self.stim_meta                   list of S dicts {"name", "type", "class",
                                        "duration_s"} — "name" is the stimulus md5
                                        (the canonical identifier; the wav filename
                                        is per-animal and not unique across the
                                        corpus); "duration_s" is the stim_duration
                                        attr from the h5 (seconds)
     - self.nrn_meta             list of N dicts with the following keys:
                                          - "cell_id"      basename of the h5 file (no extension)
                                          - "animal_id"    one of AA4_ANIMAL_IDS
                                          - "sex"          'M' or 'F' (last char of animal_id)
                                          - "site"         recording site label, e.g. "Site1"
                                          - "electrode"    int 1-32 — channel index across both
                                                           electrode arrays at this site (each
                                                           array is 16 channels in one hemisphere
                                                           in 5/6 birds; 1 array in the 6th)
                                          - "ldepth"       left-array depth (µm) at this site
                                          - "rdepth"       right-array depth (µm)
                                          - "sort_type"    'single', 'multi', or 'noise'/'tdt'
                                                           (the latter two are filtered out)
                                          - "sort_id"      online-sort id (int)
                                          - "subsort_id"   offline spike-sorting id (int) —
                                                           parsed from the trailing ``_ss<N>`` of
                                                           the filename; ``None`` if absent

    Note: the dataset paper does NOT publish a per-cell brain-area assignment
    (cf. PDF §Methods: "units were not precisely assigned one of the above
    areas") — the depth + electrode-array geometry is the only anatomical
    proxy. The PDF also does not document which electrode IDs (1-16 vs 17-32)
    correspond to the left vs right hemisphere; users wishing to derive
    "hemisphere" from "electrode" should confirm with the dataset authors.

    """

    def __init__(self, path: Optional[str] = None, animals='all',
                 stimuli=('song', 'call', 'mlnoise'),
                 dt_ms=1.0, smooth=True, n_mels=32, compression='cubic',
                 download: bool = False,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initializes the AA4 Dataset.

        Parameters
        ----------
        path : str, optional
            Path to the 'CRCNS_AA4/data/' folder containing one subfolder
            per animal (with `.h5` cell files + a `wavfiles/` directory
            of stimulus `.wav`s). Defaults to the platformdirs cache.
        animals : 'all' or sequence of str
            Animals to load (any subset of `AA4_ANIMAL_IDS`).
        stimuli : sequence of str
            Stimulus types to keep; subset of {'song', 'call', 'mlnoise'}.
        dt_ms : float
            Time-bin width in ms.
        smooth : bool
            If True, smooth PSTHs in place with a 21 ms Hanning window
            (Hsu, Borst & Theunissen 2004).
        n_mels : int
            Number of mel frequency bands of the stimulus spectrogram.
        compression : {'cubic', 'log1p', 'none'}
            Compression applied to the spectrogram (saturation effect of hair cells).
        download : bool, default False
            If True and an animal's data is missing under ``path``, fetch
            its tarball (~hundreds of MB per animal) from the NERSC mirror
            and untar in place. Only the animals listed in ``animals`` are
            downloaded — useful for quick iteration on a subset.
        username, password : str, optional
            CRCNS credentials. Default to ``$CRCNS_USERNAME`` /
            ``$CRCNS_PASSWORD``. Prefer env vars over passing literals.
        """

        if path is None:
            path = str(default_cache_dir("AA4"))

        animals_to_load = AA4_ANIMAL_IDS if animals == 'all' else tuple(animals)
        if download:
            download_aa4(path, animals=animals_to_load,
                         username=username, password=password)

        super().__init__(path, dt_ms)

        # general
        self.species = 'zebra finch'
        self.F = n_mels
        self.compression = compression
        self.animals = animals_to_load
        self.stim_types = set(stimuli)

        ###########################################
        # 1. preload mel-spectrograms per animal
        ###########################################

        # hop_length (samples) | dt (ms)   — at sr = stim wav's sr
        # the wav sample rate varies across animals so hop = sr * dt_ms / 1000
        wav_specs_by_animal = {}
        for animal in self.animals:
            wav_dir = os.path.join(path, animal, 'wavfiles')
            specs = {}
            for fname in sorted(os.listdir(wav_dir)):
                if not fname.endswith('.wav'):
                    continue
                sid = os.path.splitext(fname)[0]    # e.g. 'stim85'
                waveform, sr = torchaudio.load(os.path.join(wav_dir, fname))
                hop = max(1, int(sr * self.dt / 1000))
                n_fft = hop * 10
                mel_tf = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr, n_mels=self.F, n_fft=n_fft, hop_length=hop,
                )
                spec = mel_tf(waveform)  # (1, F, T) for mono wav

                if self.compression == 'cubic':
                    spec = torch.pow(spec, 1.0 / 3)
                elif self.compression == 'log1p':
                    spec = torch.log1p(spec)
                elif self.compression == 'none':
                    pass

                if spec.ndim == 2:
                    spec = spec.unsqueeze(0)
                specs[sid] = spec
            wav_specs_by_animal[animal] = specs

        ###########################################
        # 2. walk h5 cell files per animal
        ###########################################

        # ordered list of unique stim md5s (corpus-wide canonical id)
        stim_uids = []
        stim_meta_map = {}   # md5 -> {"name", "type", "class"}
        stim_spec_map = {}   # md5 -> spectrogram tensor (1, F, T)

        # per-neuron accumulator
        units_data = []  # list of dicts: {'meta': nrn_meta_dict, 'responses': {md5: (R, T) tensor}}

        for animal in self.animals:
            sex = animal[-1]
            animal_path = os.path.join(path, animal)
            for fname in sorted(os.listdir(animal_path)):
                if not fname.endswith('.h5'):
                    continue
                h5_path = os.path.join(animal_path, fname)
                cell_id = os.path.splitext(fname)[0]
                with h5py.File(h5_path, 'r') as celldata:
                    sort_type = _decode_attr(celldata.attrs.get('sortType', b''))
                    if sort_type in ('tdt', 'noise'):
                        continue
                    sort_type = _AA4_SORTTYPE_FIXES.get(sort_type, sort_type)

                    subsort_match = _AA4_SUBSORT_RE.search(cell_id)
                    subsort_id = int(subsort_match.group(1)) if subsort_match else None

                    nrn_meta = {
                        'cell_id': cell_id,
                        'animal_id': animal,
                        'sex': sex,
                        'site': _decode_attr(celldata.attrs.get('site', b'')),
                        'electrode': int(celldata.attrs.get('electrode', 0)),
                        'ldepth': float(celldata.attrs.get('ldepth', np.nan)),
                        'rdepth': float(celldata.attrs.get('rdepth', np.nan)),
                        'sort_type': sort_type,
                        'sort_id': int(celldata.attrs.get('sortid', 0)),
                        'subsort_id': subsort_id,
                    }
                    responses = {}

                    # iterate stim classes (skip metadata groups)
                    for cls in sorted(_get_subgroups(celldata)):
                        if cls in ('class_info', 'extra_info'):
                            continue
                        cls_grp = celldata[cls]
                        for stim_key in sorted(_get_subgroups(cls_grp)):
                            stim_grp = cls_grp[stim_key]

                            stim_type = _decode_attr(stim_grp.attrs.get('stim_type', b''))
                            if stim_type not in self.stim_types:
                                continue

                            stim_md5 = _decode_attr(stim_grp.attrs.get('stim_md5', b''))
                            stim_class = _decode_attr(stim_grp.attrs.get('stim_class', b''))
                            stim_dur_s = float(stim_grp.attrs.get('stim_duration', np.nan))

                            # register unique stimulus on first encounter
                            if stim_md5 not in stim_meta_map:
                                spec = wav_specs_by_animal[animal].get(f'stim{stim_key}')
                                if spec is None:
                                    # wav missing for this animal — skip the stim altogether
                                    continue
                                stim_uids.append(stim_md5)
                                stim_meta_map[stim_md5] = {
                                    'name': stim_md5,
                                    'type': stim_type,
                                    'class': stim_class,
                                    'duration_s': stim_dur_s,
                                }
                                stim_spec_map[stim_md5] = spec

                            T_stim = stim_spec_map[stim_md5].shape[-1]

                            # bin spike times (in seconds in h5) into (R, T_stim)
                            trial_tensors = []
                            for trial_key in sorted(_get_subgroups(stim_grp)):
                                raw_times = stim_grp[trial_key]['spike_times'][()]
                                raw_times = raw_times[raw_times >= 0]    # post-onset only
                                if raw_times.size == 0:
                                    continue
                                times_ms = (raw_times * 1000.0).tolist()
                                trial_tensors.append(time_binning(times_ms, dt_ms=self.dt))
                            if not trial_tensors:
                                continue

                            # align each trial to T_stim (right-pad with 0, or crop)
                            aligned = []
                            for t in trial_tensors:
                                if t.shape[-1] < T_stim:
                                    t = torch.nn.functional.pad(
                                        t, (0, T_stim - t.shape[-1]), mode='constant', value=0.0,
                                    )
                                elif t.shape[-1] > T_stim:
                                    t = t[..., :T_stim]
                                aligned.append(t)
                            counts = torch.stack(aligned, dim=0)    # (R, T_stim)

                            if torch.all(counts == 0):
                                continue

                            responses[stim_md5] = counts

                    if responses:
                        units_data.append({'meta': nrn_meta, 'responses': responses})

        ###########################################
        # 3. assemble core dataset attributes
        ###########################################

        self.N_neurons = len(units_data)
        self.stims = [stim_spec_map[uid] for uid in stim_uids]
        self.stim_meta = [stim_meta_map[uid] for uid in stim_uids]
        self.nrn_meta = [u['meta'] for u in units_data]

        # responses[s][n] = (R, T) tensor or (1, 1) NaN sentinel
        self.responses = []
        for uid in stim_uids:
            row = []
            for u in units_data:
                if uid in u['responses']:
                    row.append(u['responses'][uid])
                else:
                    row.append(torch.full((1, 1), float('nan')))
            self.responses.append(row)

        # smooth PSTHs with a 21 ms Hanning window (Hsu / Borst / Theunissen 2004)
        if smooth:
            self.smooth_responses(window_ms=21.0)

        # self.nrn_masks is a derived @property on the base class — no need
        # to populate it here
        self.validate()
