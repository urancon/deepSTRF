import time
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.NAT4.NAT4_Dataset import NAT4Dataset
from models.PSTH_models import *
from metrics.metrics import correlation_coefficient
from utils.utils import set_random_seed
import torch.utils

seeds = range(10)
first_run = True

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
print(f"\nSelected device: {device}\n")

dataset = NAT4Dataset('../datasets/NAT4/nat4.pt')

# Parameters of the model
T = 1       # Temporal window size
K = 7       # Kernel size
S = 3       # Stride
C = 7       # Hidden channels

# optimization
n_epochs = 300
learning_rate = 0.001
weight_decay = 0.02

# Weights & Biases logging
config = {
    "dataset": dataset.__class__.__name__,
    "temporal_window_size": T,
    "Kernel Size": K,
    "Stride": S,
    "Hidden Channels": C,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay
}

wandb.init(
    project='PROJECT_NAME_PLACEHOLDER', entity='ENTITY_PLACEHOLDER',
    config=config)

ccraw_array = []
ccnorm_array = []
cc_norm_all_seeds_array = np.zeros((len(seeds), dataset.n))

for neuron_index in tqdm(range(dataset.n)):

    dataset.set_actual_neuron(neuron_index)
    best_test_ccnorm = 0.
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

        # load the dataset used in Rahman et al.
        n_bands = dataset.get_F()
        n_time_steps = dataset.get_T()

        train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [14, 2, 4])
        batch_size = 1
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

        # The model
        net = GRU_RRF1d_Net(n_bands=n_bands,temporal_window_size=T, kernel_size=K, stride=S, hidden_channels=C).to(device)
        if first_run:
            wandb.config.update({"model": net.__class__.__name__, "Nb of parameters": net.count_trainable_params()})
            first_run = False

        print(f"Model: {net.__class__.__name__}, # params: {net.count_trainable_params()}")

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
            epoch_train_cc_norm = 0.
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

                # adding the metrics values for one sound
                epoch_train_loss += loss.item()
                epoch_train_cc += cc.item()
                epoch_train_cc_norm += (cc.item() / ccmax.item())

            # averaging the total of the metrics over all the sounds to get the mean values for 1 epoch
            epoch_train_loss /= len(train_dataloader)
            epoch_train_cc /= len(train_dataloader)
            epoch_train_cc_norm /= len(train_dataloader)

            # #### EVAL ##### #
            epoch_val_loss = 0.
            epoch_val_cc = 0.
            epoch_val_cc_norm = 0.

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
                epoch_val_cc_norm += (cc.item() / ccmax.item())

            epoch_val_loss /= len(valid_dataloader)
            epoch_val_cc /= len(valid_dataloader)
            epoch_val_cc_norm /= len(valid_dataloader)

            # save trained model if it has improved
            if epoch_val_loss < best_val_loss:
                torch.save(net.state_dict(), "./results/response_model.pth")
                best_val_loss = epoch_val_loss
                best_val_cc = epoch_val_cc
                best_val_cc_norm = epoch_val_cc_norm
                best_train_cc = epoch_train_cc
                best_train_cc_norm = epoch_train_cc_norm
                best_train_loss = epoch_train_loss

        #######################
        # III. PREDICT
        #######################

        # load saved model
        net.load_state_dict(torch.load("./results/response_predictor_snn.pth"))

        test_loss = 0.
        test_cc = 0.
        test_cc_norm = 0.
        son = 0

        for spectrogram, response, ccmax in test_dataloader:
            # data preprocessing
            spectrogram = spectrogram.to(device)
            response = response.to(device)
            ccmax = ccmax.to(device)
            # feed to the SNN network
            with torch.no_grad():
                prediction = net(spectrogram)

            loss = criterion(prediction, response)
            test_loss += loss.item()

            cc = correlation_coefficient(prediction, response)
            test_cc += cc.item()
            test_cc_norm += (cc.item() / ccmax.item())

        test_loss /= len(test_dataloader)
        test_cc /= len(test_dataloader)
        test_cc_norm /= len(test_dataloader)

        #######################
        # IV. AVERAGE OVER RUNS
        #######################

        final_test_loss += test_loss
        final_test_cc += test_cc
        final_test_cc_norm += test_cc_norm
        run_time = time.time() - start_time

        cc_norm_all_seeds_array[i, neuron_index] = test_cc_norm

        print(f"Run: {i + 1}, loss: {test_loss}, ccraw: {test_cc}, ccnorm: {test_cc_norm}, time: {run_time} s")

    final_test_loss /= len(seeds)
    final_test_cc /= len(seeds)
    final_test_cc_norm /= len(seeds)

    ccnorm_array.append(final_test_cc_norm)
    ccraw_array.append(final_test_cc)

    wandb.log({"CC Raw": final_test_cc, "CC Norm": final_test_cc_norm, "Loss": final_test_loss})

    print(f"Result for the neuron n°{neuron_index + 1}")
    print(f"Loss: {final_test_loss}, CCraw: {final_test_cc}, CCnorm: {final_test_cc_norm} (average over {len(seeds)} runs)")
    if neuron_index == 0:
        torch.save(net.state_dict(), "saved_model.pt")

mean_cc_norm = np.mean(ccnorm_array)
mean_cc_raw = np.mean(ccraw_array)

cc_norm_all_seeds_array = np.mean(cc_norm_all_seeds_array, 1)

wandb.log({"Mean CC Norm": mean_cc_norm, "Mean CC Raw":mean_cc_raw,
           "Best Val Loss" : best_val_loss,"Best Val CCraw" : best_val_cc,"Best Val CCnorm" : best_val_cc_norm,
           "Best Train Loss" : best_val_loss,"Best Train CCraw" : best_train_cc,"Best Train CCnorm" : best_train_cc_norm,"CCnorm All seeds" : cc_norm_all_seeds_array})

