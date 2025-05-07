import os
import scipy.io as sio
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d

import torch
from torch.utils.data.dataset import Dataset


class Asari_Dataset(Dataset):
    """
    Pytorch class to manipulate the subset of the CRCNS-ac1 dataset known as "Asari"

    TODO: edit docstring


    signal_type = 'raw', 'subtreshold', 'psth'/'suprathreshold'

    if max_clip_length = 'min' then max_clip_length is set to that of the lowest clip.
      ==> As a result, all clips will have the same length and will be therefore batchable.


    ============= ABOUT ==============

    data acquisition:
    - 25 neurons recorded in A1 in deeply anesthetized rats
    - standard blind whole-cell patch-clamp --> membrane potential recorded at 4kHz in current-clamp
    - presentation of natural and synthetic stimuli

    spectrograms:
    - simple amplitude short-term fourier transform
    - spectrograms with a temporal resolution of 1 msec (default)
    - 100 Hz to 25.6 kHz, 6 bins by octave (default) --> 49 logarithmically scaled frequency bands


    ============= USAGE ==============

    you can select specific neurons, either by their index, or by masking on the following features:
    - TODO: mask here ?

    you can select specific stimuli, by applying the following mask
     - TODO: also include white noise and pure tone data ---> possible masks: 'natural', 'whitenoise', 'puretone'

    after class instanciation, you can select a particular neuron to train a model

    TODO: example code + because stimuli/response pairs do not all have the same duration, batchsize must be = 1 at all
     times

    set a temporal resolution of your choice (> 1ms) with the 'dt' argument in the constructor. dt is the duration of
    spectrogram / response time bins, in ms


    ============= STRUCTURE ==============

    data is contained in the class attribute 'self.data', which contains a list of N_neurons dictionaries  of the
    following structure:
    e.g.
        {"spectrograms": list,      # N_sounds * torch.tensor(1, F, T)
         "responses": list,         # N_sounds * torch.tensor(N_repeats, T)
         "stim_type": list          # str, e.g. 'natural'/'tone'/'whitenoise'
        }


    ============= SOURCE ==============

    Original data freely available at:
        http://crcns.org/data-sets/ac/ac-1/about

    Original paper:
        C. K. Machens, M. S. Wehr a,d A. M. Zador (2004), "Linearity of Cortical Receptive Fields Measured with
        Natural Sounds", J. Neurophysiol. 102(5):2638-56

    Dataset citation:
        Asari, Hiroki; Wehr, Michael; Machens, Christian; Zador, Anthony M. (2009): Auditory cortex and thalamic
        neuronal responses to various natural and synthetic sounds. CRCNS.org, http://dx.doi.org/10.6080/K0KW5CXR

    """

    def __init__(self, path: str,
                 sites=('A1', 'MGB'),
                 neuron_indices='all',
                 stimuli=('natural', 'tone', 'whitenoise', 'randomchords', 'dmr', 'torc', 'synthetic'),
                 dt=1,
                 min_clip_length=1,                 # int, in number of samples
                 max_clip_length=99999,             # idem
                 signal_type='raw'):

        print("loading dataset...")

        dt_factor = dt

        self.data = []

        # iterate through neurons (1 file per neuron)
        neuron_indices = tuple(range(159)) if neuron_indices == 'all' else neuron_indices
        for neuron_index, file in enumerate(sorted(os.listdir(path))):

            # neuron not in target site --> skip
            if neuron_index not in neuron_indices:
                continue

            # load neuron data
            filepath = os.path.join(path, file)
            try:
                single_neuron_data = sio.loadmat(filepath)
            except ValueError:
                continue

            # neuron not in target site --> skip
            rec_site = 'A1' if 'cortex' in str(single_neuron_data['rec_site'][0]) else 'MGB'
            if rec_site not in sites:
                continue

            freq_axis = torch.from_numpy(single_neuron_data['freq_axis']).float()
            stims_metadata = single_neuron_data['current_unit_stims_meta'][0]
            stims_data = single_neuron_data['current_unit_stims'][0]     # stims[stim_idx] --> shape: (F, T)
            resps_data = single_neuron_data['current_unit_resps'][0]     # resps[stim_idx] --> shape: (T, R)

            stims2save = []
            resps2save = []
            metas2save = []
            ccmax2save = []
            ttrc2save = []

            for stim_idx in range(len(stims_data)):

                stim_meta = stims_metadata[stim_idx]
                stim_type = stim_meta[0, 0][0][0]
                stim_type = 'natural' if stim_type == 'naturalsound' else stim_type

                if stim_type in stimuli:

                    # stimulus-response pair (with trials)
                    stim = torch.from_numpy(stims_data[stim_idx]).float()
                    stim = stim.unsqueeze(0)
                    resp = torch.from_numpy(resps_data[stim_idx]).float()
                    resp = resp.permute(1, 0)   # (T, R) --> (R, T)

                    # change the nature of the signal
                    if signal_type == 'raw':
                        # TODO: super detrend
                        for r in range(len(resp)):  # tentative solution
                            resp[r] = resp[r] - torch.from_numpy(medgauss_low_pass_filter(resp[r], med_size=501, gauss_sig=40.)).float()
                        # reduce temporal resolution of response
                        resp = torch.nn.functional.avg_pool1d(resp, kernel_size=dt_factor, stride=dt_factor)

                    elif signal_type in ['spikes', 'psth', 'PSTH', 'supra', 'suprathresh', 'suprathreshold']:
                        # for each repeat / response trial
                        # resp = torch.from_numpy(median_filter(resp, 10)).float()    # TODO CAUTION: FOR N-DIM ARRAYS, median_filter applies the median to all dims !!!
                        for r in range(len(resp)):  # tentative solution
                            # only keep spikes with high-pass filtering = original signal - low-pass filtered version
                            # apply median filter of window length 10 ms (cf. Asari et al. paper) and substract output
                            resp[r] = resp[r] - torch.from_numpy(median_filter(resp[r], 10)).float()
                            # thresholding
                            spk_detection_threshold = 2.5 * resp[r].std(dim=-1)
                            resp[r] = (resp[r] > spk_detection_threshold).float()
                            # TODO: convolve full temp. res. PSTH with a 21 ms hanning window to smooth it ?
                            resp[r] = torch.from_numpy(signal.convolve(resp[r], signal.windows.hann(21), mode='same', method='direct'))

                        # time binning at desired resol; multiply by kernel size to effectively obtain sum pooling instead of avg
                        resp = torch.nn.functional.avg_pool1d(resp, kernel_size=dt_factor, stride=dt_factor) * dt_factor

                    elif signal_type in ['potential', 'sub', 'subthresh', 'subthreshold']:
                        # remove spikes through low-pass filtering
                        # apply median filter of window length 10 ms (cf. Asari et al. paper)
                        #resp = torch.from_numpy(median_filter(resp, 10)).float()    # TODO CAUTION: FOR N-DIM ARRAYS, median_filter applies the median to all dims !!!
                        for r in range(len(resp)):  # tentative solution
                            resp[r] = torch.from_numpy(median_filter(resp[r], 10)).float()
                            # TODO: super detrend
                            resp[r] = resp[r] - torch.from_numpy(medgauss_low_pass_filter(resp[r], med_size=1001, gauss_sig=40.)).float()
                        # reduce temporal resolution of response
                        resp = torch.nn.functional.avg_pool1d(resp, kernel_size=dt_factor, stride=dt_factor)

                    else:
                        raise ValueError(f"received invalid argument 'signal_type' {signal_type}, please try another")

                    # change spectrogram temporal resolution
                    stim = torch.nn.functional.interpolate(stim, scale_factor=1 / dt_factor, mode='linear')

                    '''
                    # TODO: ok to downsample time this way in a spectrogram ? --> I think so !
                    # change temporal resolution (by discarding one sample every dt_factor along temporal dimension)
                    dt_factor = dt
                    #stim = stim[:, :, ::dt_factor]
                    #resp = resp[:, ::dt_factor]
                    # another way to downsample: linear interpolation
                    stim = torch.nn.functional.interpolate(stim, scale_factor=1 / dt_factor, mode='linear')
                    resp = torch.nn.functional.interpolate(resp.unsqueeze(0), scale_factor=1 / dt_factor, mode='linear')[0]
                    '''

                    # TODO: response normalization
                    # per-response normalization in [0, 1]
                    # resp = torch.from_numpy(detrend(resp, axis=-1, type='linear')).float()
                    # ----
                    #psth = resp.mean(0)                         # avg across trials
                    #resp = (resp - psth.mean(0))/psth.std(0)    # temporal avg/std
                    #resp = (resp - psth.min())/(psth.max() - psth.min())
                    '''
                    # TODO: SUPER detrend
                    for r in range(len(resp)):
                        resp[r] = resp[r] - torch.from_numpy(
                            medgauss_low_pass_filter(resp[r], med_size=1001, gauss_sig=40.)).float()
                    '''

                    # cut into smaller clips of predefined lengths
                    stim = stim.split(max_clip_length, dim=-1)  # tuple of N_subclips * (1, F, T<max_clip_length)
                    resp = resp.split(max_clip_length, dim=-1)  # tuple of N_subclips * (R, T<max_clip_length)

                    for stim_subclip, resp_subclip in zip(stim, resp):
                        T = stim_subclip.shape[-1]
                        if T < min_clip_length:
                            continue

                        # compute noise-related normalization factors
                        #ccmax_subclip = compute_CCmax(resp_subclip.unsqueeze(0), max_iters=126).squeeze()
                        #ttrc_subclip = compute_TTRC(resp_subclip.unsqueeze(0)).squeeze()

                        # get some metadata
                        meta_subclip = {}
                        if stim_type == 'tone':
                            meta_subclip['stim_freq'] = stim_meta[0, 0][1][0].item()
                            meta_subclip['stim_amp'] = stim_meta[0, 0][2][0].item()
                        elif stim_type == 'natural':
                            meta_subclip['descr'] = stim_meta[0, 0][1][0]
                        else:
                            pass

                        # register these data
                        stims2save.append(stim_subclip)
                        resps2save.append(resp_subclip)
                        #ccmax2save.append(ccmax_subclip)
                        #ttrc2save.append(ttrc_subclip)
                        metas2save.append(meta_subclip)

            if len(stims2save) == 0:
                print(f"no stimulus available for this neuron (#{neuron_index}) given these input masks ! skipping... ")
                continue

            # TODO: normalize responses between 0 and 1
            '''
            # per-neuron normalization
            min_activ_nrn = min([resps2save[i].min() for i in range(len(resps2save))])
            max_activ_nrn = max([resps2save[i].max() for i in range(len(resps2save))])
            resps2save = [(resps2save[i]-min_activ_nrn)/(max_activ_nrn-min_activ_nrn) for i in range(len(resps2save))]
            '''

            neuron_dict = {"spectrograms": stims2save,
                           "responses": resps2save,
                           "metadata": metas2save,
                           #"ccmaxes": ccmax2save,
                           #"ttrcs": ttrc2save
                           }
            self.data.append(neuron_dict)

        assert len(self.data) > 0, "no neuron available given these input masks ! Please try less strict conditions."
        self.N_neurons = len(self.data)
        self.F = self.data[0]['spectrograms'][0].shape[-2]   # nb of spectrogram frequency bands
        self.freq_axis = freq_axis      # vector of spectrogram frequencies, in Hz
        self.I = 0      # select neuron #0 by default
        self.species = 'rat'
        self.sites = sites
        self.stims = stimuli
        self.dt = dt            # ms
        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
        self.signal_type = signal_type

        print("finished!")

    def __len__(self):
        neuron_data = self.data[self.I]                 # select neuron according to current index
        n_sounds = len(neuron_data['spectrograms'])
        return n_sounds

    def __getitem__(self, sound_index):
        neuron_data = self.data[self.I]                         # select neuron according to current index
        spectro = neuron_data['spectrograms'][sound_index]      # (1, F, T)
        response = neuron_data['responses'][sound_index]        # (N_repeats, T)
        response = response.unsqueeze(0)                        # (N_neurons=1, R, T)
        mask = torch.ones(1).bool()
        #ccmax = neuron_data['ccmaxes'][sound_index]
        #ttrc = neuron_data['ttrcs'][sound_index]
        return spectro, response, mask  #, ccmax, ttrc

    def select_neuron(self, neuron_index):
        assert (neuron_index >= 0) and (neuron_index < self.N_neurons), "neuron_index must be positive and < to the # neurons"
        self.I = neuron_index

    def get_frequency_axis(self):
        return self.freq_axis

    def get_F(self):
        return self.F

    def get_N(self):
        return self.N_neurons

    def __str__(self):
        descr = (f"AsariDataset("
                 f"N_neurons={self.N_neurons}, "
                 f"sites={self.sites}, "
                 f"stims={self.stims}, "
                 f"dt={self.dt}ms, "
                 f"min_clip_length={self.min_clip_length}, "
                 f"max_clip_length={self.max_clip_length})")
        return descr


# TODO: make available to other datasets ?
def medgauss_low_pass_filter(sig_raw, med_size, gauss_sig):
    return gaussian_filter1d(median_filter(sig_raw, med_size), sigma=gauss_sig, order=0, mode='reflect', axis=-1)
