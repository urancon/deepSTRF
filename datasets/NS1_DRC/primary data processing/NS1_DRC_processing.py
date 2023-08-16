import scipy.io as sio
import numpy as np
import torch
from scipy.signal.windows import hann
from itertools import combinations

# Load the MATLAB data
data = sio.loadmat('MetadataSHEnCneurons.mat')
neuron = data['neuron'][0]

# matrice de sortie
good_neurons = []

# Parcours de la structure neuron pour prendre que les 73 neurons
for i in range(len(neuron)):
    if (neuron[i]['singleT'][0] == 'Yes' or neuron[i]['singleT'][0] == 'Maybe') and (neuron[i]['depth'][0] >= 0):
        # Ajout de l'unité à la matrice de sortie si bonne valeur
        good_neurons.append(neuron[i])

PSTH = np.zeros((len(good_neurons), 32, 5200 // 5))
cchalftot = np.zeros((len(good_neurons), 32, 1))
ccmax = np.zeros((len(good_neurons), 32, 1))

for i, good_neuron in enumerate(good_neurons):
    temp = sio.loadmat('spikesandwav/' + good_neuron['path'][0])
    for j in range(32):  # Les 32 sons
        # Créer une matrice pour stocker les signaux de spike de chaque répétition
        spike_matrix = np.zeros((32, 5200 // 5))
        cchalf = None

        n_repetitions = int(temp['data']['set'][0, 0]['repeats'][0,j].shape[1])
        if j < 20:  # si le son est de NS1_DRC
            # matrice pour stocker les 126 paires de 10 et 10 elements :
            cchalf = np.zeros(126)
            for h in range(len(cchalf)):
                for k in range(n_repetitions):  # Les 20 répétitions d'un son
                    # spiketimes = np.round(temp['data']['set'][0, j]['repeats'][0, k]['t'][0]) MATLAB version
                    spiketimes = np.round(temp['data']['set'][0,0]['repeats'][0,j][0]['t'][k][0])
                    spiketimes = spiketimes[spiketimes < 5200]

                    # Conversion en tableau de zéros
                    input_array = np.zeros(5200)

                    # Remplissage avec des 1 aux indices correspondants
                    input_array[spiketimes.astype(int)] = 1

                    # Regrouper les 1 de input_array par timebins de 5
                    temp_array = input_array.reshape(-1, 5).sum(axis=1)

                    # Stocker le signal de spike de chaque répétition dans la matrice
                    spike_matrix[k, :] = temp_array

                spike_matrix_shuffle = spike_matrix[np.random.permutation(spike_matrix.shape[0]), :]

                # Séparer en 2 le dataset
                spike_matrix_1 = spike_matrix_shuffle[:10, :]
                spike_matrix_2 = spike_matrix_shuffle[10:, :]

                # Calculer la moyenne des signaux de spike pour chaque répétition
                if h ==0:
                    average_spike_signal = np.mean(spike_matrix_shuffle, axis=0)  # 1*1040

                average_spike_signal_1 = np.mean(spike_matrix_1, axis=0)  # 1*1040
                average_spike_signal_2 = np.mean(spike_matrix_2, axis=0)  # 1*1040

                # Fenêtre de hanning
                window_duration_ms = 21
                timebin_duration_ms = 5

                window_duration_bins = round(window_duration_ms / timebin_duration_ms)

                # Créer la fenêtre de Hanning
                hanning_window = hann(window_duration_bins)

                # Appliquer la fenêtre de Hanning à average_spike_signal
                if h == 0:
                    average_spike_signal_windowed = np.convolve(average_spike_signal, hanning_window, mode='same')

                average_spike_signal_windowed_1 = np.convolve(average_spike_signal_1, hanning_window, mode='same')
                average_spike_signal_windowed_2 = np.convolve(average_spike_signal_2, hanning_window, mode='same')

                coefficient_correlation = np.corrcoef(average_spike_signal_windowed_1, average_spike_signal_windowed_2)

                if h == 0:
                    PSTH[i,j] = average_spike_signal_windowed

                cchalf[h] = coefficient_correlation[0, 1]

        else: #Si le son est de DRC
            # matrice pour stocker les 126 paires de 10 et 10 elements :
            cchalf = np.zeros(10)
            spiketimes = temp['data']['set'][0, 0]['repeats'][0, j][0]['t'].tolist() #[k][0]
            couples = [
                ((1, 2), (3, 4, 5)),
                ((1, 3), (2, 4, 5)),
                ((1, 4), (2, 3, 5)),
                ((1, 5), (2, 3, 4)),
                ((2, 3), (1, 4, 5)),
                ((2, 4), (1, 3, 5)),
                ((2, 5), (1, 3, 4)),
                ((3, 4), (1, 2, 5)),
                ((3, 5), (1, 2, 4)),
                ((4, 5), (1, 2, 3))
            ]

            for c,couple in enumerate(couples):
                indice1, indice2 = couple[0]
                indice3, indice4, indice5 = couple[1]

                pair_1 = spiketimes[indice1-1],spiketimes[indice2-1]
                pair_2 = spiketimes[indice3-1],spiketimes[indice4-1],spiketimes[indice5-1]

                psth = np.zeros(((len(couple[0]) + len(couple[1])),1040))
                psth_1 = np.zeros((len(couple[0]),1040))
                psth_2 = np.zeros((len(couple[1]),1040))

                for e,element in enumerate(pair_1):
                    temp_array_drc = np.zeros((1,5200))
                    spiketime = np.squeeze(np.round(pair_1[e]))
                    spiketime = spiketime.astype(int)
                    spiketime = spiketime[spiketime < 5200]

                    temp_array_drc[0][spiketime] = 1
                    for w in range(5200 // 5):
                        start_index = w * 5
                        end_index = (w + 1) * 5
                        psth_1[e][w] = np.sum(temp_array_drc[0][start_index:end_index])
                    psth[0:2] = psth_1

                for e,element in enumerate(pair_2):
                    temp_array_drc = np.zeros((1, 5200))
                    spiketime = np.squeeze(np.round(pair_2[e]))
                    spiketime = spiketime.astype(int)
                    spiketime = spiketime[spiketime < 5200]

                    temp_array_drc[0][spiketime] = 1
                    for w in range(5200 // 5):
                        start_index = w * 5
                        end_index = (w + 1) * 5
                        psth_2[e][w] = np.sum(temp_array_drc[0][start_index:end_index])
                    psth[2:5] = psth_2

                # Calculer la moyenne des signaux de spike
                if c == 0:
                    average_pair = np.mean(psth, axis=0)  # 1*1040

                average_pair_1 = np.mean(psth_1, axis=0)  # 1*1040
                average_pair_2 = np.mean(psth_2, axis=0)  # 1*1040

                # Fenêtre de hanning
                window_duration_ms = 21
                timebin_duration_ms = 5

                window_duration_bins = round(window_duration_ms / timebin_duration_ms)

                # Créer la fenêtre de Hanning
                hanning_window = hann(window_duration_bins)

                # Appliquer la fenêtre de Hanning à average_spike_signal
                if c == 0:
                    average_pair_windowed = np.convolve(average_pair, hanning_window, mode='same')

                average_pair_windowed_1 = np.convolve(average_pair_1, hanning_window, mode='same')
                average_pair_windowed_2 = np.convolve(average_pair_2, hanning_window, mode='same')

                coefficient_correlation_pair = np.corrcoef(average_pair_windowed_1, average_pair_windowed_2)

                if c == 0:
                    PSTH[i, j] = average_pair_windowed

                cchalf[c] = coefficient_correlation_pair[0, 1]

        cchalftot[i, j] = np.mean(cchalf)

ccmax = np.sqrt(2. / (1 + np.sqrt(1. / pow(cchalftot,2))))
ccmax = np.mean(ccmax, axis=(1, 2))
ccmax = torch.tensor(ccmax)

data = torch.load('data.pt')
spectrograms = data['spectrograms']
responses = PSTH

torch.save({'spectrograms': spectrograms, 'responses': responses, 'ccmax':ccmax}, "../NS1_DRC/ns1_new.pt")
print("Dataset Sauvegardé !")

torch.save(ccmax,'ccmax.pt')