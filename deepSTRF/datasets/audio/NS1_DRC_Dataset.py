import os

import numpy as np
import scipy.io as sio
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset


# Stimulus duration is 4995 ms = 999 bins at the 5 ms binning the original
# authors used. The provided spectrogram tensor (test_data_5ms.mat) is also
# at this temporal resolution.
NS1_RAW_LEN_MS = 4995
NS1_NAT_SOUNDS = 20

# Indices of the 4 natural-speech stimuli, per Rahman et al. (2020) Fig. S2.
# 0-indexed; the original (1-indexed) sound numbers are 9, 10, 11, 12.
NS1_SPEECH_INDICES = (8, 9, 10, 11)
# Index 0 (sound 1) and index 19 (sound 20) are water sounds; index 3 is a
# ferret vocalization; index 6 is insects buzzing. All remaining indices have
# no published category and are left as "unknown".
NS1_TYPE_OVERRIDES = {
    0: "water_sounds",
    3: "ferret_vocalization",
    6: "insects_buzzing",
    19: "water_sounds",
    **{i: "human_speech" for i in NS1_SPEECH_INDICES},
}

# Train/test split used by Rahman et al. (2020). Kept on the class as a
# convenience for downstream notebooks; we do NOT enforce this split inside
# the dataset (callers can sub-select by stim index).
NS1_RAHMAN_TRAINVAL_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
NS1_RAHMAN_TEST_INDICES = [3, 6, 9, 19]


class NS1_Dataset(AudioNeuralDataset):
    """A PyTorch dataset for the NS1 (Harper et al. 2016, Rahman et al. 2020) data.


    =============== SOURCE ================

    See original papers for details:
     - "Network receptive field modeling reveals extensive integration and
       multi-feature selectivity in auditory cortical neurons", Harper et al.
       PLoS Computational Biology (2016).
     - "Simple transformations capture auditory input to cortex", Rahman et al.
       PNAS (2020).

    Data freely available at https://osf.io/ayw2p/ (no account required).


    =============== DETAILS ================

    - 119 multi/single units from primary auditory cortex (A1) of deeply-
      anesthetized ferrets. Of those, 73 pass the "single-unit at known depth"
      filter the original authors used (``singleT in {'Yes', 'Maybe'}`` and
      ``depth >= 0``); ``select_pop_by_nrn_attr`` over ``single_t``/``depth``
      reproduces this subset.
    - 20 natural sound clips of 4.995 s each, presented 20 times per neuron
      (every neuron heard every clip — the response grid is fully dense, no
      NaN sentinels). DRC stimuli are NOT loaded here: their stimulus
      spectrograms are not packaged with the OSF release.
    - The spectrogram tensor is precomputed at ``dt = 5 ms`` (F = 34 frequency
      bands, T = 999 bins). The ``dt_ms`` constructor arg is currently
      validated against this resolution; varying ``dt`` would require
      re-computing spectrograms from raw wavs (TODO).


    =============== STRUCTURE ================

    Follows the standard deepSTRF data paradigm (see docs/_source/md/data_paradigm.md).
    NS1-specific metadata contents:
     - self.stims                       list of S=20 tensors (1, F=34, T=999)
     - self.responses                   list of S lists of N tensors (R=20, T=999)
     - self.stim_meta                   list of S dicts {"name", "type"}
     - self.neuron_metadata             list of N dicts {"cell_id", "area",
                                        "depth_um", "noise_ratio", "single_n",
                                        "single_t", "n_electrodes",
                                        "electrode_number"}
                                        — ``noise_ratio`` is the Sahani-Linden
                                        normalised-noise-power; lower = cleaner
                                        (NOT SNR despite the legacy field name
                                        in the .mat). ``single_n`` is the
                                        single-unit flag from spike-snippet
                                        clustering (0/1); ``single_t`` is the
                                        manual triage label ('Yes'/'Maybe'/'No').

    """

    def __init__(self, path: str, dt_ms: float = 5.0, smooth: bool = True):
        """
        Parameters
        ----------
        path : str
            Path to the ``NS1_DRC/data/`` folder containing
            ``test_data_5ms.mat``, ``MetadataSHEnCneurons.mat``, and the
            ``spikesandwav/`` subdirectory of per-neuron spike .mat files.
        dt_ms : float, default 5.0
            Time-bin width in ms. Must equal 5.0 — the bundled spectrogram is
            precomputed at this resolution. Other values would require
            re-spectrogramming the wavs (not implemented).
        smooth : bool, default True
            If True, smooth PSTHs in place with a 21 ms Hanning window
            (Hsu, Borst & Theunissen 2004).
        """

        super().__init__(path, dt_ms)
        assert dt_ms == 5.0, (
            f"NS1 spectrograms are precomputed at 5 ms; got dt_ms={dt_ms}. "
            f"Re-binning the spike trains is straightforward but the spectrogram "
            f"would need to be recomputed from raw wavs (TODO)."
        )

        self.species = "ferret"

        # ----------- 1. load the precomputed spectrograms -----------
        # X_nfht: (S=20, F=34, 1, T=999) at dt=5 ms
        spec_data = sio.loadmat(os.path.join(path, "test_data_5ms.mat"))
        X = spec_data["X_nfht"]
        S, F, _, T = X.shape
        assert S == NS1_NAT_SOUNDS, f"expected {NS1_NAT_SOUNDS} stims, got {S}"
        self.F = int(F)

        self.stims = [
            torch.from_numpy(X[s, :, 0, :]).float().unsqueeze(0)  # (1, F, T)
            for s in range(S)
        ]
        self.stim_meta = [
            {"name": f"nat{s + 1:02d}", "type": NS1_TYPE_OVERRIDES.get(s, "unknown")}
            for s in range(S)
        ]

        # ----------- 2. load per-neuron metadata + spike data -----------
        meta = sio.loadmat(os.path.join(path, "MetadataSHEnCneurons.mat"))
        neurons = meta["neuron"][0]

        self.responses = [[] for _ in range(S)]
        self.neuron_metadata = []
        spikes_root = os.path.join(path, "spikesandwav")

        for neuron in neurons:
            uid = str(neuron[0].item())[:-4]  # drop trailing '.mat' from filename-as-uid

            spike_path = os.path.join(spikes_root, str(neuron["path"].item()))
            try:
                temp = sio.loadmat(spike_path)
            except FileNotFoundError:
                # The OSF release has a few neurons with a "_1" suffix mismatch
                # between the metadata path and the actual file; try the fallback.
                fallback = spike_path[:-4] + "_1.mat"
                try:
                    temp = sio.loadmat(fallback)
                except FileNotFoundError:
                    print(f"NS1: skipping neuron {uid!r} (spike file not found at {spike_path} or {fallback})")
                    continue

            self.neuron_metadata.append({
                "cell_id": uid,
                "area": "A1",
                "depth_um": int(neuron["depth"].item()),
                "noise_ratio": float(neuron["NoiseRatio"].item()),
                "single_n": int(neuron["singleN"].item()),
                "single_t": str(neuron["singleT"].item()),
                "n_electrodes": int(neuron["NrOfElectrodes"].item()),
                "electrode_number": int(neuron["ElectrodeNumber"].item()),
            })

            for s in range(S):
                repeats_grp = temp["data"]["set"][0, 0]["repeats"][0, s]
                R = int(repeats_grp.shape[1])
                # bin spike times to the requested dt
                bin_size = int(round(dt_ms))
                T_resp = NS1_RAW_LEN_MS // bin_size  # = 999 for 5 ms / 4995 ms
                spike_matrix = np.zeros((R, T_resp), dtype=np.float32)
                for r in range(R):
                    spiketimes = np.round(repeats_grp[0]["t"][r][0]).astype(np.int64)
                    spiketimes = spiketimes[(spiketimes >= 0) & (spiketimes < NS1_RAW_LEN_MS)]
                    one_hot = np.zeros(NS1_RAW_LEN_MS, dtype=np.float32)
                    one_hot[spiketimes] = 1.0
                    spike_matrix[r] = one_hot.reshape(-1, bin_size).sum(axis=1)
                self.responses[s].append(torch.from_numpy(spike_matrix))

        self.N_neurons = len(self.neuron_metadata)

        # smooth PSTHs with a 21 ms Hanning window (Hsu / Borst / Theunissen 2004)
        if smooth:
            self.smooth_responses(window_ms=21.0)

        # self.nrn_masks is a derived @property on the base class — no need
        # to populate it here
        self.validate()
