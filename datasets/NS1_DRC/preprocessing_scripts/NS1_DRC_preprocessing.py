import scipy.io as sio
import numpy as np
import torch
from scipy.signal.windows import hann


# Constants
CCHALF_MAX_ITERS = 126
VALID_NEURONS = 73
NAT_SOUNDS = 20
NAT_REPEATS = 20
DRC_SOUNDS = 12
DRC_REPEATS = 5

# Hanning window
window_duration_ms = 21
timebin_duration_ms = 5
window_duration_bins = round(window_duration_ms / timebin_duration_ms)
hanning_window = hann(window_duration_bins)

# Load the data from original MATLAB file
data = sio.loadmat('../data/MetadataSHEnCneurons.mat')
neuron = data['neuron'][0]

# Go through the data and only select the 73 valid neurons
good_neurons = []
for i in range(len(neuron)):
    if (neuron[i]['singleT'][0] == 'Yes' or neuron[i]['singleT'][0] == 'Maybe') and (neuron[i]['depth'][0] >= 0):
        good_neurons.append(neuron[i])

PSTH = np.zeros((len(good_neurons), 32, 5200 // 5))
nat_responses = np.zeros((len(good_neurons), 20, 20, 5200 // 5))  # (N, S, R, T)
drc_responses = np.zeros((len(good_neurons), 12, 5, 5200 // 5))   # idem
cchalftot = np.zeros((len(good_neurons), 32, 1))
ccmax = np.zeros((len(good_neurons), 32, 1))

# loop through neurons and process them
for i, good_neuron in enumerate(good_neurons):
    temp = sio.loadmat('../data/spikesandwav/' + good_neuron['path'][0])

    # loop through sounds/stimuli
    for j in range(32):

        n_repetitions = int(temp['data']['set'][0, 0]['repeats'][0, j].shape[1])     # 20 for natural stims, 12 for drc

        # spike matrix containing spiking responses for each repeat of each sound
        spike_matrix = np.zeros((n_repetitions, 5200 // 5))     # (R, T)

        # if the sound is natural (first 20/32 sounds)
        if j < 20:

            # loop through repeats (20 for natural stims) to create the (R, T) response matrix of the current sound
            for k in range(n_repetitions):
                spiketimes = np.round(temp['data']['set'][0, 0]['repeats'][0, j][0]['t'][k][0])
                spiketimes = spiketimes[spiketimes < 5200]

                input_array = np.zeros(5200)                        # Init matrix of zeros: 0 --> no spike, 1 --> spike
                input_array[spiketimes.astype(int)] = 1             # Toggle elements at spike times
                temp_array = input_array.reshape(-1, 5).sum(axis=1) # time binning of spikes: 5 bins --> 1 bin

                # store each response trial
                spike_matrix[k, :] = temp_array

            # save repeats
            nat_responses[i, j] = spike_matrix

        # if the sound is a dynamic random chord (drc), i.e. the last 12/32 sounds
        else:

            # loop through repeats (5 for natural stims) to create the (R, T) response matrix of the current sound
            for k in range(n_repetitions):
                spiketimes = np.round(temp['data']['set'][0, 0]['repeats'][0, j][0]['t'][k][0])
                spiketimes = spiketimes[spiketimes < 5200]

                input_array = np.zeros(5200)
                input_array[spiketimes.astype(int)] = 1
                temp_array = input_array.reshape(-1, 5).sum(axis=1)
                spike_matrix[k, :] = temp_array

            # save repeats
            drc_responses[i, j-20] = spike_matrix   # -20 = -N_nat_stims

nat_responses = torch.from_numpy(nat_responses)[:, :, :, :999].to(torch.float32)  # (N, S, R, T)
drc_responses = torch.from_numpy(drc_responses)[:, :, :, :999].to(torch.float32)  # (N, S, R, T)

data2save = []
for neuron_idx in range(VALID_NEURONS):

    input_stims = []
    responses = []
    psth = []
    ccmaxes = []
    for sound_idx in range(NAT_SOUNDS + DRC_SOUNDS):
        if sound_idx < NAT_SOUNDS:
            responses.append(nat_responses[neuron_idx, sound_idx])
        else:
            responses.append(drc_responses[neuron_idx, sound_idx - NAT_SOUNDS])

    neuron_dict = {'responses': responses}
    data2save.append(neuron_dict)

torch.save(data2save, "../data/ns1_drc_responses.pt")
print("Dataset preprocessed & saved !")
