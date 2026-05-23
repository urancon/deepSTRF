import os
import scipy.io as sio
import torch

from deepSTRF.datasets.audio.audio_dataset import AudioNeuralDataset
from deepSTRF.metrics.performance import compute_CCmax, compute_TTRC


WEHR_VALID_NEURONS = (0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)


WEHR_NEURONS_SPLIT_NATURAL = [
    [7, 2, 2],      # neuron 0
    [5, 1, 1],      #        1      --> one of the 3 unresponsive neurons ? (cf. Machens et al.'s paper)
    [3, 1, 1],      #        2      --> one of the 3 unresponsive neurons ? (cf. Machens et al.'s paper)
    [7, 1, 1],      #        3
    [1, 1, 1],      #        4      --> not enough data
    [6, 1, 1],      #        5
    [3, 1, 1],      #        6
    [1, 1, 1],      #        7
    [4, 1, 1],      #        8      --> one of the 3 unresponsive neurons ? (cf. Machens et al.'s paper) | confirmed !
    [7, 1, 2],      #        9
    [11, 2, 3],     #        10
    [25, 4, 7],     #        11
    [7, 1, 2],      #        12
    [25, 4, 7],     #        13
    [7, 1, 1],      #        14
    [14, 3, 5],     #        15
    [25, 4, 7],     #        16
    [17, 3, 5],     #        17
    [5, 1, 1],      #        18
    [11, 2, 3],     #        19
    [5, 1, 1],      #        20
    [12, 2, 4],     #        21
    [7, 2, 2],      #        22
    [44, 6, 13],    #        23
    [4, 1, 1]       # ..     24
]


class WehrDataset(AudioNeuralDataset):
    """PyTorch class for the "Wehr" subset of the CRCNS-ac1 dataset.

    Intracellular recordings: 25 neurons in rat A1 (deeply anesthetized),
    membrane potential recorded at 4 kHz under whole-cell patch clamp, in
    response to natural and synthetic stimuli. Spectrograms are simple
    short-term Fourier amplitudes, 1 ms resolution by default, spanning
    100 Hz - 25.6 kHz at 6 bins/octave (49 logarithmically spaced bands).

    Set the temporal resolution via the ``dt`` constructor argument (the
    duration of spectrogram / response time bins, in ms). Because the
    stimulus/response pairs differ in duration, use ``batch_size=1``.

    .. note::

       Legacy NEMS-based loader, retained for backward compatibility and
       slated for removal. New code should prefer the modernized audio
       datasets.

    References
    ----------
    Machens, Wehr & Zador (2004). "Linearity of Cortical Receptive Fields
    Measured with Natural Sounds." *J. Neurophysiol.* 102(5): 2638-56.

    Dataset: Asari, Wehr, Machens & Zador (2009). "Auditory cortex and
    thalamic neuronal responses to various natural and synthetic sounds."
    CRCNS.org. http://dx.doi.org/10.6080/K0KW5CXR — data at
    http://crcns.org/data-sets/ac/ac-1/about
    """

    def __init__(self, path: str, neuron_indices=tuple(range(25)), stimuli=('natural', 'tone', 'whitenoise'), dt=1):

        # load the data
        self.data = []

        # iterate through neurons (1 file per neuron)
        neuron_index = 0
        for file in sorted(os.listdir(path)):

            # preprocess selected neurons only
            if neuron_index in neuron_indices:

                filepath = os.path.join(path, file)
                single_neuron_data = sio.loadmat(filepath)

                # iterate through stimuli
                n_sounds = single_neuron_data['spectros_to_save'].size
                spectrograms = []
                responses = []
                ccmaxes = []
                ttrcs = []
                stim_type = []
                for sound_index in range(n_sounds):

                    # load arrays
                    spectro = torch.from_numpy(single_neuron_data['spectros_to_save'][0, sound_index])
                    response = torch.from_numpy(single_neuron_data['responses_to_save'][0, sound_index])

                    # resample to given temporal resolution
                    spectro = torch.nn.functional.interpolate(spectro.unsqueeze(0), scale_factor=1/dt, mode='linear')[0]
                    response = torch.nn.functional.interpolate(response.unsqueeze(0), scale_factor=1/(4*dt), mode='linear')[0]
                    n_timebins = response.size(1)
                    spectro = spectro[:, :n_timebins]   # remove excess time bins in spectrogram
                    spectro = spectro.unsqueeze(0)      # add channel dimension

                    # compute normalization factors (noise)
                    ccmax = compute_CCmax(response.unsqueeze(0), max_iters=126)
                    ttrc = compute_TTRC(response.unsqueeze(0))

                    spectrograms.append(spectro.float())
                    responses.append(response.float())
                    ccmaxes.append(ccmax.item())
                    ttrcs.append(ttrc.item())
                    stim_type.append('natural')  # TODO: pure tones stimuli

                if neuron_index == 12:
                    # there is some drift from the middle of response #10 of neuron #12, so remove this half
                    neuron12_clip10_T = responses[10].shape[-1]
                    responses[10] = responses[10][:, :neuron12_clip10_T//2]
                    spectrograms[10] = spectrograms[10][:, :, :neuron12_clip10_T//2]
                    # response  #11 of neuron #12 is out of the distribution (possible recording problem)
                    responses.pop(11)
                    ccmaxes.pop(11)
                    ttrcs.pop(11)
                    spectrograms.pop(11)
                    stim_type.pop(11)

                # normalize responses to have a min of 0 and a max of 1 for the neuron
                min_resp = float('inf')
                max_resp = -float('inf')
                for response in responses:                  # find min and max values over this neuron
                    if torch.min(response) < min_resp:
                        min_resp = torch.min(response)
                    if torch.max(response) > max_resp:
                        max_resp = torch.max(response)
                for i in range(len(responses)):             # do the normalization
                    responses[i] = (responses[i] - min_resp) / (max_resp - min_resp)

                neuron_dict = {"spectrograms": spectrograms,
                               "responses": responses,
                               "stim_type": stim_type,
                               "ccmaxes": ccmaxes,
                               "ttrcs": ttrcs}
                self.data.append(neuron_dict)

            # filter out unwanted neurons
            else:
                pass

            neuron_index += 1

        self.N_neurons = len(self.data)

        # TODO: select stimuli, like this for instance
        '''
        for neuron_data in self.data:
            
            # possibility 1
            neuron_data['spectro'] = [neuron_data['spectro'][sound_index] for sound_index in range(len(neuron_data['spectro'])) if neuron_data['stim_type'][sound_index] in stimuli]

            # possibility 2
            stim_mask = []
            neuron_data['spectro'] = neuron_data['spectro'][stim_mask]
            neuron_data['response'] = neuron_data['response'][stim_mask]
            neuron_data['stim_type'] = neuron_data['stim_type'][stim_mask]
        '''

        self.F = self.data[0]['spectrograms'][0].shape[1]   # nb of spectrogram frequency bands
        self.I = 0      # select neuron #0 by default

    def __len__(self):
        neuron_data = self.data[self.I]                 # select neuron according to current index
        n_sounds = len(neuron_data['spectrograms'])
        return n_sounds

    def __getitem__(self, sound_index):
        neuron_data = self.data[self.I]                         # select neuron according to current index
        spectro = neuron_data['spectrograms'][sound_index]      # (1, F, T)
        response = neuron_data['responses'][sound_index]        # (N_repeats, T)
        ccmax = neuron_data['ccmaxes'][sound_index]
        ttrc = neuron_data['ttrcs'][sound_index]
        return spectro, response, ccmax, ttrc

    def select_neuron(self, neuron_index):
        assert (neuron_index >= 0) and (neuron_index < self.N_neurons), "neuron_index must be positive and < to the # neurons"
        self.I = neuron_index

    def get_F(self):
        return self.F

    def get_N(self):
        return self.N_neurons


class WehrDatasetPop(AudioNeuralDataset):
    """PyTorch class for the "Wehr" subset of the CRCNS-ac1 dataset.

    Intracellular recordings: 25 neurons in rat A1 (deeply anesthetized),
    membrane potential recorded at 4 kHz under whole-cell patch clamp, in
    response to natural and synthetic stimuli. Spectrograms are simple
    short-term Fourier amplitudes, 1 ms resolution by default, spanning
    100 Hz - 25.6 kHz at 6 bins/octave (49 logarithmically spaced bands).

    Set the temporal resolution via the ``dt`` constructor argument (the
    duration of spectrogram / response time bins, in ms). Because the
    stimulus/response pairs differ in duration, use ``batch_size=1``.

    .. note::

       Legacy NEMS-based loader, retained for backward compatibility and
       slated for removal. New code should prefer the modernized audio
       datasets.

    References
    ----------
    Machens, Wehr & Zador (2004). "Linearity of Cortical Receptive Fields
    Measured with Natural Sounds." *J. Neurophysiol.* 102(5): 2638-56.

    Dataset: Asari, Wehr, Machens & Zador (2009). "Auditory cortex and
    thalamic neuronal responses to various natural and synthetic sounds."
    CRCNS.org. http://dx.doi.org/10.6080/K0KW5CXR — data at
    http://crcns.org/data-sets/ac/ac-1/about
    """

    def __init__(self, path: str, neuron_indices=tuple(range(25)), stimuli=('natural', 'tone', 'whitenoise'), dt=1):
        super().__init__(path)

        # load the data
        self.data = []

        # iterate through neurons (1 file per neuron)
        neuron_index = 0
        for file in sorted(os.listdir(path)):

            # preprocess selected neurons only
            if neuron_index in neuron_indices:

                filepath = os.path.join(path, file)
                single_neuron_data = sio.loadmat(filepath)

                # iterate through stimuli
                n_sounds = single_neuron_data['spectros_to_save'].size
                spectrograms = []
                responses = []
                ccmaxes = []
                ttrcs = []
                stim_type = []
                for sound_index in range(n_sounds):
                    # load arrays
                    spectro = torch.from_numpy(single_neuron_data['spectros_to_save'][0, sound_index])
                    response = torch.from_numpy(single_neuron_data['responses_to_save'][0, sound_index])

                    # resample to given temporal resolution
                    spectro = torch.nn.functional.interpolate(spectro.unsqueeze(0), scale_factor=1 / dt, mode='linear')[
                        0]
                    response = \
                    torch.nn.functional.interpolate(response.unsqueeze(0), scale_factor=1 / (4 * dt), mode='linear')[0]
                    n_timebins = response.size(1)
                    spectro = spectro[:, :n_timebins]  # remove excess time bins in spectrogram
                    spectro = spectro.unsqueeze(0)  # add channel dimension

                    # compute normalization factors (noise)
                    ccmax = compute_CCmax(response.unsqueeze(0), max_iters=126)
                    ttrc = compute_TTRC(response.unsqueeze(0))

                    spectrograms.append(spectro.float())
                    responses.append(response.float())
                    ccmaxes.append(ccmax.item())
                    ttrcs.append(ttrc.item())
                    stim_type.append('natural')  # TODO: pure tones stimuli

                if neuron_index == 12:
                    # there is some drift from the middle of response #10 of neuron #12, so remove this half
                    neuron12_clip10_T = responses[10].shape[-1]
                    responses[10] = responses[10][:, :neuron12_clip10_T // 2]
                    spectrograms[10] = spectrograms[10][:, :, :neuron12_clip10_T // 2]
                    # response  #11 of neuron #12 is out of the distribution (possible recording problem)
                    responses.pop(11)
                    ccmaxes.pop(11)
                    ttrcs.pop(11)
                    spectrograms.pop(11)
                    stim_type.pop(11)

                # normalize responses to have a min of 0 and a max of 1 for the neuron
                min_resp = float('inf')
                max_resp = -float('inf')
                for response in responses:  # find min and max values over this neuron
                    if torch.min(response) < min_resp:
                        min_resp = torch.min(response)
                    if torch.max(response) > max_resp:
                        max_resp = torch.max(response)
                for i in range(len(responses)):  # do the normalization
                    responses[i] = (responses[i] - min_resp) / (max_resp - min_resp)

                neuron_dict = {"spectrograms": spectrograms,
                               "responses": responses,
                               "stim_type": stim_type,
                               "ccmaxes": ccmaxes,
                               "ttrcs": ttrcs}
                self.data.append(neuron_dict)

            # filter out unwanted neurons
            else:
                pass

            neuron_index += 1

        self.N_neurons = len(self.data)

        # TODO: select stimuli, like this for instance
        '''
        for neuron_data in self.data:

            # possibility 1
            neuron_data['spectro'] = [neuron_data['spectro'][sound_index] for sound_index in range(len(neuron_data['spectro'])) if neuron_data['stim_type'][sound_index] in stimuli]

            # possibility 2
            stim_mask = []
            neuron_data['spectro'] = neuron_data['spectro'][stim_mask]
            neuron_data['response'] = neuron_data['response'][stim_mask]
            neuron_data['stim_type'] = neuron_data['stim_type'][stim_mask]
        '''

        self.F = self.data[0]['spectrograms'][0].shape[-2]  # nb of spectrogram frequency bands
        self.I = 0  # select neuron #0 by default

    def __len__(self):
        neuron_data = self.data[self.I]  # select neuron according to current index
        n_sounds = len(neuron_data['spectrograms'])
        return n_sounds

    def __getitem__(self, sound_index):
        neuron_data = self.data[self.I]  # select neuron according to current index
        spectro = neuron_data['spectrograms'][sound_index]  # (1, F, T)
        response = neuron_data['responses'][sound_index]    # (R, T)
        response = response.unsqueeze(0)                    # (N=1, R, T)
        nrn_mask = torch.ones(1).bool()                     # (N=1,)
        return spectro, response, nrn_mask

    def select_neuron(self, neuron_index):
        assert (isinstance(neuron_index, int)) and (neuron_index >= 0) and (neuron_index < self.N_neurons), \
            "neuron_index must be positive and < to the # neurons"
        self.I = neuron_index

    def select_population(self, neuron_indices):
        raise NotImplementedError("Population training not available for this dataset")

    def get_F(self):
        """ Returns the number of frequency bins in the spectrograms. """
        return self.data[0]['spectrograms'][0].shape[-2]

    def get_T(self):
        """ Returns the number of time bins in the spectrograms. """
        return self.data[0]['spectrograms'][0].shape[-1]
