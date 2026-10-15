import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torchaudio

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.datasets.audio._crcns_aa_loaders import time_binning
from deepSTRF.utils.audio_io import load_wav
from deepSTRF.utils.data_download import (
    crcns_download,
    crcns_resolve,
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


# Stimulus identity table, computed once from the audio of the whole corpus (the CRCNS release
# is fixed). The md5 of a wav cannot be trusted as the identity of a sound: the 20 songs and 10
# ripple noises played to every bird were saved in three slightly different copies (one shared
# by four birds, LblBlu2028M's copy with ~420 samples changed at the onset, WhiBlu5396M's copy
# re-quantised), and 3 calls in two. And call stimuli are 2.5-s bouts of three renditions drawn
# from a pool per caller and call type, so different bouts can contain the same rendition.
# Entry per md5: [canonical md5, overlap group id, stim type, caller, call type, identity id].
#   - canonical: the copy that duplicates of the same sound are merged into (same length and
#     log-mel r > 0.999 on the louder half of the bins; the copy heard by most birds);
#   - overlap group: stimuli sharing any identical audio, the duplicates included. Renditions
#     are separated by digital silence and a reused rendition is bit-identical, so each
#     stimulus is cut into events at its silences and every event is slid along every other
#     stimulus: linked at waveform r > 0.9, or, for WhiBlu5396M's re-quantised quiet calls,
#     at log-mel r > 0.999 at the best waveform lag. Different recordings of one caller's
#     stereotyped call stay below waveform r 0.4. 594 md5s -> 539 sounds -> 224 groups (at
#     most 18 stimuli); no group mixes stimulus types or callers.
#   - caller / call type: from the original wav name. Calls 'Caller_[XYZ_]Ty_<renditions>'
#     -> (Caller, Ty); named songs -> (singer, 'So'); the 20 'STRFxx0000' songs (singer not
#     given) -> (None, 'So'); ripple noises -> (None, 'Ri').
#   - identity: what is held out together by group_split(holdout='caller_type'). Union of the
#     overlap group, the caller x call type (songs: singer), the bird's-own / tutor / brother
#     song trio (learned copies of one song), and the two call pairs of different callers with
#     whole-call log-mel r > 0.90 (Th WhiWhi1415 ~ WhiGra0114, LT LblBlu1630 ~ LblBlu2028; all
#     other cross-caller pairs stay <= 0.90). A zebra finch has its own version of each call
#     type (Elie & Theunissen 2018, Nat Commun 9:4026): 15% of the acoustic variance of the
#     call stimuli lies within a caller x call type, 43% between callers of a type and 42%
#     between call types. 539 sounds -> 136 identities.
_STIM_GROUPS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '_crcns_aa4_stim_groups.json')
_STIM_GROUPS: Optional[Dict[str, Tuple[str, str, str, Optional[str], str, str]]] = None


def aa4_stim_groups() -> Dict[str, Tuple[str, str, str, Optional[str], str, str]]:
    """``{md5: (canonical md5, overlap group id, stim type, caller, call type, identity id)}``
    for every AA4 stimulus wav (caller is None for the ripples and the 20 unnamed songs)."""
    global _STIM_GROUPS
    if _STIM_GROUPS is None:
        with open(_STIM_GROUPS_FILE) as f:
            _STIM_GROUPS = {k: tuple(v) for k, v in json.load(f).items()}
    return _STIM_GROUPS


AA4_ANIMAL_IDS = ('BlaBro09xxF', 'GreBlu9508M', 'LblBlu2028M', 'WhiBlu5396M', 'WhiWhi4522M', 'YelBlu6903F')


def download_aa4(dest: Optional[str] = None,
                 animals: Sequence[str] = AA4_ANIMAL_IDS,
                 username: Optional[str] = None,
                 password: Optional[str] = None) -> str:
    """Download CRCNS-AA4 archives from the CRCNS download server into ``dest``.

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
            crcns_download(crcns_resolve("aa-4", archive_name, username=username,
                                         password=password),
                           archive_path, username=username, password=password)
        untar(archive_path, dest_path)  # tarball already wraps in <animal>/

    # CRCNSCode tutorial — small, useful pointer to the original loaders
    code_dir = os.path.join(dest_path, "CRCNSCode")
    if not os.path.isdir(code_dir):
        archive_path = os.path.join(dest_path, "CRCNSCode.tar.gz")
        if not os.path.exists(archive_path):
            crcns_download(crcns_resolve("aa-4", "CRCNSCode.tar.gz", username=username,
                                         password=password),
                           archive_path, username=username, password=password)
        untar(archive_path, dest_path)

    return dest_path


class CRCNSAA4Dataset(AudioNeuralDataset):
    """PyTorch dataset for the CRCNS-AA4 recordings.

    1401 extracellular, spike-sorted single and multi units of adult zebra
    finches (4 males, 2 females) in Field L, caudolateral and caudomedial
    mesopallium (CLM, CMM) and caudomedial nidopallium (NCM) — though units
    were not precisely assigned to one of these areas. Three stimulus classes
    (conspecific songs, calls, ripple noise), each a few seconds long and
    presented ~10 times. Population- and batch-compatible. Data are available
    at https://crcns.org/data-sets/aa/aa-4/about-aa-4 (free CRCNS account).

    Notes
    -----
    Follows the standard deepSTRF data paradigm (see
    ``docs/_source/md/data_paradigm.md``). AA4-specific metadata:

    - ``stims`` are mel-spectrograms ``(1, F, T_s)``.
    - ``stim_meta`` dicts hold ``name`` (the stimulus md5 — the canonical
      identifier, since the wav filename is per-animal and not unique across
      the corpus), ``type``, ``class``, ``duration_s`` (the
      ``stim_duration`` attr from the h5, in seconds), ``md5s`` (the md5s of
      every wav merged into this stimulus, see ``merge_duplicates``) and
      ``overlap_group`` (stimuli that share any audio carry the same id),
      ``caller`` (the bird that produced a call or song; ``None`` for the
      ripples and the 20 songs whose singer is not given), ``call_type``
      (``'Ag'``, ``'Be'``, ``'DC'``, ``'Di'``, ``'LT'``, ``'Ne'``, ``'Te'``,
      ``'Th'``, ``'Wh'``; ``'So'`` for songs, ``'Ri'`` for ripples) and
      ``identity`` (one caller's call type, merged with any stimulus it shares
      audio with). :meth:`group_split` splits by these keys; they are there to
      build any other split too.
    - ``nrn_meta`` dicts hold: ``cell_id`` (h5 basename, no extension),
      ``animal_id``, ``sex`` (``'M'`` / ``'F'``), ``site`` (e.g. ``"Site1"``),
      ``electrode`` (int 1-32, channel index across both 16-channel arrays at
      a site), ``ldepth`` / ``rdepth`` (left / right array depth in µm),
      ``sort_type`` (``'single'`` / ``'multi'``; ``'noise'`` / ``'tdt'`` are
      filtered out), ``sort_id`` (online-sort int) and ``subsort_id``
      (offline-sort int parsed from the trailing ``_ss<N>``; ``None`` if
      absent).

    The dataset paper does not publish a per-cell brain-area assignment, so
    the depth + electrode-array geometry is the only anatomical proxy; nor
    does it document which electrode IDs (1-16 vs 17-32) map to the left vs
    right hemisphere — confirm with the dataset authors before deriving a
    hemisphere from ``electrode``.

    References
    ----------
    Elie & Theunissen (2015). "Meaning in the avian auditory cortex: Neural
    representation of communication calls." *European Journal of
    Neuroscience*.

    Elie & Theunissen (2019). "Invariant neural responses for sensory
    categories revealed by the time-varying information for communication
    calls." *PLoS Computational Biology*.
    """

    def __init__(self, path: Optional[str] = None, animals='all',
                 stimuli=('song', 'call', 'mlnoise'),
                 dt_ms=1.0, smooth=True, n_mels=32, compression='cubic',
                 window_ms: float = 10.0,
                 return_waveform: bool = False, audio_fs: int = 24000,
                 merge_duplicates: bool = True,
                 download: bool = False,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initializes the AA4 Dataset.

        Parameters
        ----------
        path : str, optional
            Path to the ``CRCNS_AA4/data/`` folder containing one subfolder
            per animal (with ``.h5`` cell files + a ``wavfiles/`` directory
            of stimulus ``.wav`` files). Defaults to the platformdirs cache.
        animals : 'all' or sequence of str
            Animals to load (any subset of ``AA4_ANIMAL_IDS``).
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
            Compression applied to the spectrogram (saturation effect of hair
            cells). Ignored when ``return_waveform=True``.
        window_ms : float, default 10.0
            FFT analysis-window length in ms. ``n_fft`` is computed
            per-stim as ``round(window_ms * 1e-3 * sample_rate)`` and is
            **decoupled from ``hop_length``** so phonemic detail is
            preserved at any ``dt_ms``. Earlier versions of this dataset
            hardcoded ``n_fft = hop * 10`` — at ``dt_ms=50`` that gave a
            500 ms FFT window and over-smoothed every spec frame.
            Default ``window_ms=10.0`` preserves bit-identical
            behaviour at ``dt_ms=1`` (n_fft=320 at sr=32 kHz) while
            fixing the scaling bug at coarser bins. Ignored when
            ``return_waveform=True``.
        return_waveform : bool, default False
            If True, ``self.stims[s]`` holds the raw audio waveform
            ``(1, T_audio)`` at ``audio_fs`` Hz (grid-locked to ``T_audio =
            T_neural * hop``) instead of the in-loader mel spectrogram. Pair
            with a model whose ``wav2spec`` slot is a waveform front-end;
            responses are unchanged.
        audio_fs : int, default 24000
            Sample rate for waveform mode. The AA4 wavs are 24414 Hz, which
            gives a non-integer hop at dt=1 ms; the default 24 kHz resamples to
            a clean ``hop = 24`` (exactly dt=1 ms bins, slightly better than the
            native spec's 0.983 ms). Other values must keep
            ``audio_fs * dt_ms / 1000`` an integer. Ignored unless
            ``return_waveform=True``.
        merge_duplicates : bool, default True
            If True, wav files that hold the same sound are merged into one
            stimulus: the songs and ripple noises played to every bird exist in
            up to three copies with different md5s (differences ~50 dB below the
            signal), and 3 calls in two. Each bird heard only one copy, so the
            neurons of all birds share the merged stimulus (539 stimuli instead
            of 594). The audio is the copy heard by most birds among the loaded
            ones. ``name`` is the canonical md5 and ``md5s`` lists the merged
            ones. If False, every md5 is its own stimulus (the previous
            behaviour); duplicates then still share an ``overlap_group``.
        download : bool, default False
            If True and an animal's data is missing under ``path``, fetch
            its tarball (~hundreds of MB per animal) from the CRCNS download server
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
        # Informational hearing range (zebra finch ≈ 250 Hz – 8 kHz).
        self.hearing_range_hz = (250.0, 8000.0)
        self.F = n_mels
        self.compression = compression
        # Waveform-input mode: store raw audio (resampled to a single audio_fs,
        # since AA4 wavs are 24414 Hz which gives a non-integer hop at dt=1 ms)
        # instead of the in-loader mel spec. 24 kHz -> integer hop=24 at dt=1 ms.
        self.return_waveform = bool(return_waveform)
        self.audio_fs = int(audio_fs) if return_waveform else None
        self._wav_hop = int(round(audio_fs * self.dt / 1000)) if return_waveform else None
        self.animals = animals_to_load
        self.stim_types = set(stimuli)

        ###########################################
        # 1. preload mel-spectrograms per animal
        ###########################################

        # hop_length (samples) | dt (ms)   — at sr = stim wav's sr
        # the wav sample rate varies across animals so hop = sr * dt_ms / 1000.
        # ``n_fft`` is decoupled from ``hop`` and pinned to
        # ``window_ms * 1e-3 * sr`` (with a floor at ``hop`` so the STFT
        # constraint ``n_fft >= hop_length`` is satisfied). See the
        # ``window_ms`` docstring above for the rationale and the
        # bit-identical-at-default contract.
        self.window_ms = float(window_ms)
        wav_specs_by_animal = {}
        wav_audio_by_animal = {}
        for animal in self.animals:
            wav_dir = os.path.join(path, animal, 'wavfiles')
            specs = {}
            wavs = {}
            for fname in sorted(os.listdir(wav_dir)):
                if not fname.endswith('.wav'):
                    continue
                sid = os.path.splitext(fname)[0]    # e.g. 'stim85'
                waveform, sr = load_wav(os.path.join(wav_dir, fname))
                hop = max(1, int(sr * self.dt / 1000))
                # Derive n_fft from the (already-truncated) hop via the
                # ratio ``window_ms / dt_ms``. At the default
                # ``window_ms = 10 * dt_ms`` this collapses to the legacy
                # ``hop * 10`` regardless of sr — bit-identical on the
                # 32 kHz and 44.1 kHz wavs that ship with this dataset.
                # Floored at ``hop`` so MelSpectrogram's
                # ``n_fft >= hop_length`` constraint always holds.
                n_fft = max(int(round((self.window_ms / float(self.dt)) * hop)), hop)
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

                if return_waveform:
                    # store the raw audio, resampled to the dataset's single
                    # audio_fs and grid-locked to T_neural * hop samples so it
                    # aligns with the T_neural spec frames (= response bins). The
                    # spec is still kept (it sets T_neural / the response length).
                    T_audio = spec.shape[-1] * self._wav_hop
                    w = (waveform if sr == audio_fs
                         else torchaudio.functional.resample(waveform, sr, audio_fs))
                    if w.shape[0] > 1:
                        w = w.mean(dim=0, keepdim=True)   # downmix to mono
                    if w.shape[-1] < T_audio:
                        w = torch.nn.functional.pad(w, (0, T_audio - w.shape[-1]))
                    else:
                        w = w[..., :T_audio]
                    wavs[sid] = w.contiguous().float()
            wav_specs_by_animal[animal] = specs
            wav_audio_by_animal[animal] = wavs

        ###########################################
        # 2. walk h5 cell files per animal
        ###########################################

        # ordered list of unique stim md5s (corpus-wide canonical id)
        stim_uids = []
        stim_meta_map = {}   # md5 -> {"name", "type", "class"}
        stim_spec_map = {}   # md5 -> spectrogram tensor (1, F, T)
        stim_wav_map = {}    # md5 -> waveform tensor (1, T_audio)  [waveform mode]
        src_rank = {}        # md5 -> rank of the copy whose audio is stored (0 = canonical)
        groups = aa4_stim_groups()
        self.merge_duplicates = bool(merge_duplicates)

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

                            spec = wav_specs_by_animal[animal].get(f'stim{stim_key}')
                            if spec is None:
                                # wav missing for this animal — skip the stim altogether
                                continue
                            canon, group, _, caller, call_type, ident = groups.get(
                                stim_md5, (stim_md5, stim_md5, stim_type, None, None, stim_md5))
                            uid = canon if merge_duplicates else stim_md5
                            # Rank of this copy as the audio of the merged stimulus: the
                            # canonical one, then any copy but WhiBlu5396M's re-quantised one.
                            rank = 0 if stim_md5 == canon else 2 if animal == 'WhiBlu5396M' else 1
                            if uid not in stim_meta_map:
                                stim_uids.append(uid)
                                stim_meta_map[uid] = {
                                    'name': uid,
                                    'type': stim_type,
                                    'class': stim_class,
                                    'duration_s': stim_dur_s,
                                    'md5s': [],
                                    'overlap_group': group,
                                    'caller': caller,
                                    'call_type': call_type,
                                    'identity': ident,
                                }
                            meta = stim_meta_map[uid]
                            if stim_md5 not in meta['md5s']:
                                meta['md5s'] = sorted(meta['md5s'] + [stim_md5])
                            if uid not in stim_spec_map or rank < src_rank[uid]:
                                src_rank[uid] = rank
                                stim_spec_map[uid] = spec
                                if return_waveform:
                                    stim_wav_map[uid] = wav_audio_by_animal[animal].get(f'stim{stim_key}')

                            T_stim = stim_spec_map[uid].shape[-1]

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

                            # one copy per bird, so a unit never has two trial sets for one sound
                            assert uid not in responses, (cell_id, uid)
                            responses[uid] = counts

                    if responses:
                        units_data.append({'meta': nrn_meta, 'responses': responses})

        ###########################################
        # 3. assemble core dataset attributes
        ###########################################

        self.N_neurons = len(units_data)
        _stim_map = stim_wav_map if self.return_waveform else stim_spec_map
        self.stims = [_stim_map[uid] for uid in stim_uids]
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

    _HOLDOUTS = ('caller_type', 'caller', 'call_type', 'audio')

    def group_split(self, seed: Optional[int] = None,
                    fractions: Sequence[float] = (0.7, 0.1, 0.2),
                    holdout: str = 'caller_type') -> Dict[str, List[int]]:
        """Split the stimuli into train / val / test, keeping related sounds together.

        Stimuli that share any audio (copies of one song or ripple noise, call
        bouts that contain the same recorded call) always land in the same set,
        whatever ``holdout``. ``holdout`` says what else a val or test stimulus
        must be new in:

        - ``'caller_type'`` (default): a call type of a caller never heard in
          training (all of one bird's begging calls, say, go to one set; the
          bird can be in training with its other call types). Songs: all songs
          of one singer go to one set. Units = ``identity``. Every call type is
          drawn separately, so each set gets about ``fractions`` of each call
          type, of the songs and of the ripples. With few callers a call type
          can miss a set: LT, Th and Wh have no val stimuli at ``seed=0``.
        - ``'caller'``: a bird never heard in training, in any of its calls and
          songs. Units = identities merged per caller.
        - ``'call_type'``: a call type never heard in training: all calls of a
          type go to one set (9 call types, so the shares are coarse). Songs and
          ripples are split as under ``'caller_type'``.
        - ``'audio'``: only shared audio is kept together (``overlap_group``);
          other calls of the same bird and call type can be in train and test.
          Drawn per stimulus type (song, call, mlnoise).

        Each unit goes, in a shuffled order, to the set furthest below its
        target share of that stratum's stimuli. The draw is over the whole AA4
        corpus (the stimulus table shipped with the loader), not over the
        stimuli of this instance: a given ``seed`` and ``holdout`` put a sound
        in the same set whichever birds or stimulus types are loaded. For
        other splits (e.g. new callers AND new call types), use the
        ``caller``, ``call_type``, ``identity`` and ``overlap_group`` keys of
        ``stim_meta``, or intersect two of these splits.

        Parameters
        ----------
        seed : int, optional
            Seed of the draw. ``None`` draws a new random split at each call.
        fractions : sequence of 3 floats, default (0.7, 0.1, 0.2)
            Target train / val / test shares, in stimuli (after merging
            duplicates).
        holdout : {'caller_type', 'caller', 'call_type', 'audio'}
            What a val / test stimulus is new in (see above).

        Returns
        -------
        dict
            ``{'train': [...], 'val': [...], 'test': [...]}``: absolute stimulus
            indices (into ``stim_meta``), whatever the current selection. Use
            :meth:`stim_to_iter_indices` to build a ``torch.utils.data.Subset``.
        """
        if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(f"fractions must be 3 shares summing to 1, got {fractions}")
        if holdout not in self._HOLDOUTS:
            raise ValueError(f"holdout must be one of {self._HOLDOUTS}, got {holdout!r}")
        keys = ("train", "val", "test")
        table = aa4_stim_groups()
        # union-find over the corpus: md5 -> unit
        parent: Dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for md5, (canon, group, stim_type, caller, call_type, ident) in table.items():
            union(md5, 'g' + group)
            if holdout != 'audio':
                union(md5, 'i' + ident)
            if holdout == 'caller' and caller is not None:
                union(md5, 'c' + caller)
            if holdout == 'call_type' and stim_type == 'call':
                union(md5, 't' + call_type)
        sounds: Dict[str, set] = {}          # unit -> canonical md5s
        types: Dict[str, set] = {}           # unit -> stim types, or call types
        first: Dict[str, str] = {}           # unit -> smallest overlap group id (draw order)
        for md5, (canon, group, stim_type, caller, call_type, ident) in table.items():
            u = find(md5)
            sounds.setdefault(u, set()).add(canon)
            types.setdefault(u, set()).add(call_type if holdout == 'caller_type' else stim_type)
            first[u] = min(first.get(u, group), group)
        # stratum: the call type ('caller_type'), else the stimulus type; a bird with calls
        # and songs ('caller') is drawn with the calls
        stratum = {u: 'call' if 'call' in t else min(t) for u, t in types.items()}
        rng = random.Random(seed)
        assign: Dict[str, str] = {}
        for st in sorted(set(stratum.values())):
            us = sorted((u for u, x in stratum.items() if x == st), key=first.get)
            rng.shuffle(us)
            total = sum(len(sounds[u]) for u in us)
            count = dict.fromkeys(keys, 0)
            for u in us:
                k = max(keys, key=lambda k: fractions[keys.index(k)] * total - count[k])
                assign[u] = k
                count[k] += len(sounds[u])
        split: Dict[str, List[int]] = {k: [] for k in keys}
        for i, m in enumerate(self.stim_meta):
            split[assign[find(m["name"])] if m["name"] in table else "train"].append(i)
        return split
