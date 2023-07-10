import time
import torch
import matplotlib.pyplot as plt
import csv
import wandb
import numpy as np
from torch.utils.data import DataLoader

from utils import set_random_seed
from interpret.metrics import correlation_coefficient
from datasets.auditory_datasets import NS1Dataset
from network.PSTH_models import GRU_RRF1d_Net


def train_and_plot(neuron_index, run):
    # Set random seed
    set_random_seed(run)

    # Load the data
    data = torch.load('datasets/NS1/ns1a.pt')
    dataset = NS1Dataset(data, neuron_index)

    # Define the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"\nselected device: {device}\n")

    # Model parameters
    T = 1
    K = 7
    S = 3
    C = 7

    # Optimization parameters
    n_epochs = 300
    learning_rate = 0.001
    weight_decay = 0.02

    # Initialize wandb
    config = {
        "temporal_window_size": T,
        "Kernel Size": K,
        "Stride": S,
        "Hidden Channels": C,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }
    wandb.init(
        project='GRU_rrf1d', entity='stage-cerco-corentin-ne',
        config=config
    )

    # Initialize the model
    n_bands = dataset.get_F()
    net = GRU_RRF1d_Net(n_bands=n_bands, temporal_window_size=T, kernel_size=K, stride=S, hidden_channels=C).to(device)
    print(f"Model: {net.__class__.__name__}, # params: {net.count_trainable_params()}")
    wandb.log({"Minisobel - a": net.minisobel.a.clone().detach(), "Minisobel - w": net.minisobel.w.clone().detach()})

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load the dataset used in Rahman et al
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [14, 2, 4])
    batchsize = 1
    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

    # Training loop
    w_history = []
    a_history = []
    for epoch in range(n_epochs):
        epoch_train_loss = 0.
        epoch_train_cc = 0.
        net.train()
        for spectrogram, response, ccmax in train_dataloader:
            spectrogram = spectrogram.to(device)
            response = response.to(device)
            ccmax = ccmax.to(device)

            prediction = net(spectrogram)

            loss = criterion(prediction, response)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            net.detach()

            cc = correlation_coefficient(prediction, response)

            epoch_train_loss += loss.item()
            epoch_train_cc += cc.item()

        epoch_train_loss /= len(train_dataloader)
        epoch_train_cc /= len(train_dataloader)

        epoch_val_loss = 0.
        epoch_val_cc = 0.

        net.eval()
        with torch.no_grad():
            for spectrogram, response, ccmax in valid_dataloader:
                spectrogram = spectrogram.to(device)
                response = response.to(device)
                ccmax = ccmax.to(device)

                prediction = net(spectrogram)

                loss = criterion(prediction, response)
                cc = correlation_coefficient(prediction, response)

                epoch_val_loss += loss.item()
                epoch_val_cc += cc.item()

        epoch_val_loss /= len(valid_dataloader)
        epoch_val_cc /= len(valid_dataloader)

        wandb.log({"Minisobel - a": net.minisobel.a.clone().detach(),
                   "Minisobel - w": net.minisobel.w.clone().detach()})

        # Log w and a values
        w_history.append(net.minisobel.w.clone().detach().cpu().numpy().copy())
        a_history.append(net.minisobel.a.clone().detach().numpy().copy())

        # Log metrics to wandb
        wandb.log({"Train Loss": epoch_train_loss, "Train CC": epoch_train_cc,
                   "Validation Loss": epoch_val_loss, "Validation CC": epoch_val_cc})

    # Plot w and a evolution
    w_history = np.array(w_history)
    a_history = np.array(a_history)
    epochs = np.arange(1, n_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    for i in range(w_history.shape[1]):
        plt.plot(epochs, w_history[:, i], label=f"w{i + 1}")
    plt.xlabel("Epochs")
    plt.ylabel("w values")
    plt.title("Evolution of w values")
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(a_history.shape[1]):
        plt.plot(epochs, a_history[:, i], label=f"a{i + 1}")
    plt.xlabel("Epochs")
    plt.ylabel("a values")
    plt.title("Evolution of a values")
    plt.legend()

    plt.tight_layout()
    plt.show()

train_and_plot(neuron_index=0, run=0)

