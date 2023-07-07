import time
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from spikingjelly.clock_driven import functional
import matplotlib.pyplot as plt
import csv
import wandb

from utils import set_random_seed
from interpret.metrics import correlation_coefficient
from datasets.neurophysio_datasets import NS1Dataset
from network.PSTH_models import StatelessConvNet, RahmanDynamicNet, GRU_RRF1d_Net, GRU_RRF1dplus_Net, RRF_GRU_Learnable_Delays


def plot_spectrogram_and_signals(spectrogram, response, prediction):
    # Create a figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Plot the cochleagram using imshow
    axs[0].imshow(spectrogram[0][0], cmap='viridis')
    axs[0].set_title('Cochleagram')

    # Plot the response and prediction curves
    axs[1].plot(response[0], color='grey',alpha=0.7, label='Response')
    axs[1].plot(prediction[0].detach().numpy(), color='blue', alpha=0.7, label='Prediction')
    axs[1].set_title('Acquired Signal and Predicted Signal')
    axs[1].legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Display the plot
    plt.show()

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"\nselected device: {device}\n")

data = torch.load('datasets/NS1/ns1a.pt')

# instanciate bio-plausible SNN model
Max_Delay = 19
K = 7
S = 3
C = 7

nomentre = 0

# optimization
n_epochs = 300
learning_rate = 0.001
weight_decay = 0.02

# wandb logging
config = {
    "Max Delay": Max_Delay,
    "Kernel Size": K,
    "Stride": S,
    "Hidden Channels": C,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay
}
wandb.init(
    project='GRU_rrf1d', entity='stage-cerco-corentin-ne',
    config=config)

ccraw_array = []
ccnorm_array = []

with open('ns1a_GRU_correlations.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["neuron_id", "Run 1", "Run 2", "Run 3", "Run 4", "Run 5", "Run 6", "Run 7", "Run 8", "Run 9", "Run 10"])

    for neuron_index in tqdm(range(73)):
        # Init array for CSV
        array_cc = np.zeros(21)
        array_cc[0] = neuron_index
        dataset = NS1Dataset(data,neuron_index)

        final_test_loss = 0.
        final_test_cc = 0.
        final_test_cc_norm = 0.

        for seed, i in enumerate(seeds):
            #######################
            # 0. INIT
            #######################

            set_random_seed(seed)
            start_time = time.time()

            #######################
            # I. PREPARE
            #######################

            # load the dataset used in Rahman et al
            n_bands = dataset.get_F()
            n_timesteps = dataset.get_T()
            train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [14, 2, 4])
            batchsize = 1
            train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
            valid_dataloader = DataLoader(valid_set, batch_size=batchsize, shuffle=True)
            test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)


            # net = StatelessConvNet(temporal_window_size=T, n_hidden=H).to(device)
            # net = LIF_RRF1dplus_Net(temporal_window_size=T, kernel_size=7, stride=3, hidden_channels=7).to(device)
            # net = RahmanDynamicNet(temporal_window_size=T, n_hidden=H).to(device)
            # net = StatelessConvNet(n_bands=n_bands, temporal_window_size=41, n_hidden=20).to(device)
            # net = GRU_RRF1d_Net(n_bands=n_bands,temporal_window_size=T, kernel_size=K, stride=S, hidden_channels=C).to(device)
            net = RRF_GRU_Learnable_Delays(max_delay=Max_Delay, rrf_kernel_size=K, rrf_stride=S, hidden_channels=C).to(device)
            print(f"Model: {net.__class__.__name__}, # params: {net.count_trainable_params()}")
            if nomentre==0:
                wandb.config.update({"model": net.__class__.__name__, "Nb of parameters" : net.count_trainable_params()})
                nomentre+=1

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
                for spectrogram, response, ccmax in train_dataloader:
                    # data preprocessing
                    spectrogram = spectrogram.to(device)
                    response = response.to(device)
                    ccmax = ccmax.to(device)

                    # feed to the ANN network
                    prediction = net(spectrogram)

                    # gradient descent step
                    loss = criterion(prediction, response)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # detach stateful variables from graph
                    net.detach()

                    # correlation coefficient
                    cc = correlation_coefficient(prediction, response)

                    # logging
                    epoch_train_loss += loss.item()
                    epoch_train_cc += cc.item()

                epoch_train_loss /= len(train_dataloader)
                epoch_train_cc /= len(train_dataloader)

                # #### EVAL ##### #

                epoch_val_loss = 0.
                epoch_val_cc = 0.
                net.eval()
                for spectrogram, response, ccmax in valid_dataloader:
                    spectrogram = spectrogram.to(device)
                    response = response.to(device)
                    ccmax = ccmax.to(device)

                    prediction = net(spectrogram)

                    loss = criterion(prediction, response)
                    cc = correlation_coefficient(prediction, response)

                    net.detach()

                    epoch_val_loss += loss.item()
                    epoch_val_cc += cc.item()

                epoch_val_loss /= len(valid_dataloader)
                epoch_val_cc /= len(valid_dataloader)

                # save trained model if it has improved
                if epoch_val_loss < best_val_loss:
                    torch.save(net.state_dict(), "./results/response_predictor_snn.pth")
                    best_val_loss = epoch_val_loss

            #######################
            # III. PREDICT
            #######################

            # load saved model
            net.load_state_dict(torch.load("./results/response_predictor_snn.pth"))

            test_loss = 0.
            test_cc = 0.

            for spectrogram, response, ccmax in test_dataloader:
                # data preprocessing
                spectrogram = spectrogram.to(device)
                response = response.to(device)
                ccmax = ccmax.to(device)

                # feed to the SNN network
                with torch.no_grad():
                    prediction = net(spectrogram)

                # plot_spectrogram_and_signals(spectrogram,response,prediction)

                loss = criterion(prediction, response)
                test_loss += loss.item()

                cc = correlation_coefficient(prediction, response)
                test_cc += cc.item()

            test_loss /= len(test_dataloader)
            test_cc /= len(test_dataloader)
            test_cc_norm = test_cc/ccmax.item()

            # CSV preparation
            array_cc[i+1] = test_cc
            array_cc[i+11] = test_cc_norm

            #######################
            # IV. AVERAGE OVER RUNS
            #######################

            final_test_loss += test_loss
            final_test_cc += test_cc
            final_test_cc_norm += test_cc_norm
            run_time = time.time() - start_time

            print(f"Run: {i+1}, loss: {test_loss}, ccraw: {test_cc}, ccnorm: {test_cc_norm}, time: {run_time} s")

        #Write in CSV
        writer.writerow(array_cc)

        final_test_loss /= len(seeds)
        final_test_cc /= len(seeds)
        final_test_cc_norm /= len(seeds)

        ccnorm_array.append(final_test_cc_norm)
        ccraw_array.append(final_test_cc)

        wandb.log({"CC Raw": final_test_cc, "CC Norm": final_test_cc_norm,"Loss":final_test_loss})

        print(f"Result for the neuron n°{neuron_index+1}")
        print(f"Loss: {final_test_loss}, CCraw: {final_test_cc}, CCnorm: {final_test_cc_norm} (average over {len(seeds)} runs)")
        if neuron_index==0:
            torch.save(net.state_dict(), "saved_model.pt")

    mean_ccnorm = np.mean(ccnorm_array)
    mean_ccraw = np.mean(ccraw_array)

    wandb.log({"Mean CC Norm": mean_ccnorm, "Mean CC Raw":mean_ccraw})

    file.close()


