import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.neurophysio_datasets import NAT4Dataset, NAT4
from interpret.metrics import correlation_coefficient
from network.PSTH_models import StatelessConvNet


def plot_spectrogram_and_signals(spectrogram, response, prediction):
    # Create a figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Plot the cochleagram using imshow
    axs[0].imshow(spectrogram[0][0], cmap='viridis')
    axs[0].set_title('Cochleagram')

    # Plot the response and prediction curves
    axs[1].plot(response[0], color='grey', alpha=0.7, label='Response')
    axs[1].plot(prediction[0].detach().numpy(), color='blue', alpha=0.7, label='Prediction')
    axs[1].set_title('Acquired Signal and Predicted Signal')
    axs[1].legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Display the plot
    plt.show()


# Select the device (GPU if available, otherwise CPU)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
print(f"\nSelected device: {device}\n")

final_test_loss = 0.
final_test_cc = 0.

train_losses = []
eval_losses = []

# Load datasets
estimation = NAT4Dataset('datasets/NAT4/estimation.pt')
validation = NAT4Dataset('datasets/NAT4/validation.pt')

data = NAT4(estimation, validation)

# Open a CSV file for data saves
with open('nat4_convnet.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["neuron_id", "Train CC", "Eval CC", "Test CC"])

    for neuron_index in tqdm(range(data.neuron_count)):
        # Initialize array for CSV
        array_cc = np.zeros(11)
        array_cc[0] = neuron_index

        #######################
        # 0. INIT
        #######################

        start_time = time.time()

        ##########################
        # I. PREPARE TRAIN + VALID
        ##########################

        est_data = data.prepare_nat4(neuron_index=400, choose_dataset="est")
        # Load the dataset used in Rahman et al.
        n_bands = data.estimation.get_F()
        n_timesteps = data.estimation.get_T()
        train_set, valid_set = torch.utils.data.random_split(est_data, [460, 115])
        batchsize = 16
        train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
        valid_dataloader = DataLoader(valid_set, batch_size=batchsize, shuffle=True)

        # Instantiate bio-plausible SNN model
        T = 1
        K = 5
        S = 2
        C = 7
        net = StatelessConvNet(n_bands=n_bands, temporal_window_size=15, n_hidden=100).to(device)
        print(f"Model: {net.__class__.__name__}, # params: {net.count_trainable_params()}")

        # Optimization
        n_epochs = 300
        learning_rate = 0.001
        weight_decay = 0.02
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #######################
        # II. LEARN
        #######################

        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            # #### TRAIN ##### #
            epoch_train_loss = 0.
            epoch_train_cc = 0.
            net.train()
            for spectrogram, response in train_dataloader:
                # Data preprocessing
                spectrogram = spectrogram.to(device)
                response = response.to(device)

                # Feed
                prediction = net(spectrogram)

                # Gradient descent step
                loss = criterion(prediction, response)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                net.detach()

                # Correlation coefficient
                cc = correlation_coefficient(prediction, response)

                # Logging
                epoch_train_loss += loss.item()
                epoch_train_cc += cc.item()

            epoch_train_loss /= len(train_dataloader)
            epoch_train_cc /= len(train_dataloader)
            train_losses.append(epoch_train_loss)

            # #### EVAL ##### #
            epoch_val_loss = 0.
            epoch_val_cc = 0.
            net.eval()
            for spectrogram, response in valid_dataloader:
                spectrogram = spectrogram.to(device)
                response = response.to(device)

                prediction = net(spectrogram)

                loss = criterion(prediction, response)
                cc = correlation_coefficient(prediction, response)

                net.detach()

                epoch_val_loss += loss.item()
                epoch_val_cc += cc.item()

            epoch_val_loss /= len(valid_dataloader)
            epoch_val_cc /= len(valid_dataloader)
            eval_losses.append(epoch_val_loss)

            # Save trained model if it has improved
            if epoch_val_loss < best_val_loss:
                torch.save(net.state_dict(), "./results/response_predictor_snn.pth")
                best_val_loss = epoch_val_loss

        #######################
        # III. PREDICT
        #######################

        val_data = data.prepare_nat4(neuron_index=400, choose_dataset="val")
        test_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

        # Load saved model
        net.load_state_dict(torch.load("./results/response_predictor_snn.pth"))

        test_loss = 0.
        test_cc = 0.
        test_losses= []

        for spectrogram, response in test_dataloader:
            # Data preprocessing
            spectrogram = spectrogram.to(device)
            response = response.to(device)

            # Feed to the SNN network
            with torch.no_grad():
                prediction = net(spectrogram)

            plot_spectrogram_and_signals(spectrogram,response,prediction)

            loss = criterion(prediction, response)
            test_loss += loss.item()

            cc = correlation_coefficient(prediction, response)
            test_cc += cc.item()

        test_loss /= len(test_dataloader)
        test_cc /= len(test_dataloader)

        # CSV preparation
        array_cc[3] = test_cc

        #######################
        # IV. AVERAGE OVER RUNS
        #######################

        fig = plt.plot(train_losses)
        plt.plot(eval_losses)
        plt.plot(train_losses)
        plt.xlabel("epochs")
        plt.legend()

        run_time = time.time() - start_time

        # Write to CSV
        writer.writerow(array_cc)

        print(f"Result for neuron n°{neuron_index + 1}")
        print(f"Test loss: {test_loss}, test correlation coefficient: {test_cc}")

    file.close()
