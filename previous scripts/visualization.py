import torch
import matplotlib.pyplot as plt

# Charger le fichier "validation.pt"
# data = torch.load("validation.pt")
# data = torch.load("datasets/NAT4/estimation.pt")
data = torch.load("datasets/NAT4/validation.pt")
spectrograms = data['spectrograms']
responses = data['responses']

a= 2

def visualize_epoch(neuron_index, epoch_idx):
    # Select data according to neuron and epoch
    spectrogram = spectrograms[epoch_idx][0]
    response = responses[neuron_index]
    response = response[epoch_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plots
    ax1.imshow(spectrogram, origin='lower')
    ax1.set_title(f"Sound {epoch_idx+1} - Spectrogram")
    ax1.set_ylabel("Frequency")

    ax2.plot(response, 'r-')
    ax2.set_xlabel("Time")

for i in range(18):
        visualize_epoch(3, i)
        plt.close()

for i in range(777):
    for j in range(18):
        visualize_epoch(1, j)
        plt.close()


a = 3