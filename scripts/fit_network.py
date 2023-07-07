import time
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import wavfile
from spikingjelly.clock_driven import functional
import matplotlib.pyplot as plt
import wandb

from utils import set_random_seed
from interpret.metrics import correlation_coefficient
from datasets.neurophysio_datasets import RahmanDataset
from network.PSTH_models import StatelessConvNet, RahmanDynamicNet, GRU_RRF1d_Net, LIF_RRF1dplus_Net

#######################
# 0. INIT
#######################

seed = 0 
if seed is not None:
    set_random_seed(seed)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"\nselected device: {device}\n")


#######################
# I. PREPARE
#######################

# load the dataset used in Rahman et al
dataset = RahmanDataset("datasets/Rahman/")
dataset.responses = dataset.responses / dataset.responses.max()  # normalize responses between 0 and 1
n_bands = dataset.get_F()
n_timesteps = dataset.get_T()

a = 2

train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [14, 2, 4])
batchsize = 1
train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=batchsize, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

# instanciate a deep ANN model
T = 1
H = 70
#net = StatelessConvNet(temporal_window_size=T, n_hidden=H).to(device)
#net = LIF_RRF1dplus_Net(temporal_window_size=T, kernel_size=7, stride=3, hidden_channels=7).to(device)
net = GRU_RRF1d_Net(temporal_window_size=T, kernel_size=7, stride=3, hidden_channels=7).to(device)
print(f"Model: {net.__class__.__name__}, # params: {net.count_trainable_params()}")


# optimization
n_epochs = 300
learning_rate = 0.001
weight_decay = 0.02
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


# wandb logging
config = {
    "model": net.__class__.__name__,
    "temporal_window_size": T,
    "n_hidden": H,
    "n_params": net.count_trainable_params(),

    "seed": seed,
    "batch_size": batchsize,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay
}
wandb.init(
    entity="st",
    project="Rahman_Neural_Response_Fitting",
    name=f"SNN_minisobel_T{T}_H{H}_seed{seed}",
    config=config)


#######################
# II. LEARN
#######################

best_val_cc = 0.

for epoch in range(n_epochs):

    start_time = time.time()

    # #### TRAIN ##### #

    epoch_train_loss = 0.
    epoch_train_cc = 0.
    net.train()
    for spectrogram, response in train_dataloader:

        # data preprocessing
        spectrogram = spectrogram.to(device)
        response = response.to(device)

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

    epoch_time = time.time() - start_time

    print(f"Epoch: {epoch}, train_loss: {epoch_train_loss}, train_cc: {epoch_train_cc}, val_loss: {epoch_val_loss}, "
          f"val_cc: {epoch_val_cc}, time: {epoch_time} s")

    wandb.log({"train_loss": epoch_train_loss, "train_cc": epoch_train_cc, "val_loss": epoch_val_loss,
        "val_cc": epoch_val_cc, "minisobel_w": net.minisobel.get_w()})

    # save trained model if it has improved
    if epoch_val_cc > best_val_cc:
        print("saving model...")
        torch.save(net.state_dict(), f"./results/response_predictor_snn_T{1}_H{H}.pth")
        best_val_cc = epoch_val_cc


#######################
# III. PREDICT
#######################

# load saved model
net.load_state_dict(torch.load(f"./results/response_predictor_snn_T{1}_H{H}.pth"))

test_loss = 0.
test_cc = 0.
i = 1

for spectrogram, response in test_dataloader:

    # data preprocessing
    spectrogram = spectrogram.to(device)
    response = response.to(device)

    # feed to the SNN network
    with torch.no_grad():
        prediction = net(spectrogram)

    loss = criterion(prediction, response)
    test_loss += loss.item()

    cc = correlation_coefficient(prediction, response)
    test_cc += cc.item()

    # show prediction
    fig = plt.figure()
    plt.subplot(211)
    plt.title("input spectrogram")
    plt.imshow(spectrogram.squeeze().cpu())
    plt.subplot(212)
    plt.title("actual vs predicted response")
    plt.plot(response.squeeze().cpu())
    plt.plot(prediction.squeeze().cpu())
    plt.legend(["actual", "prediction"])
    plt.show()
    wandb.log({f"test_pred{i}": fig})

    i += 1

test_loss /= len(test_dataloader)
test_cc /= len(test_dataloader)
print(f"test loss: {test_loss}, test correlation coefficient: {test_cc}")
wandb.log({"test_loss": test_loss, "test_cc": test_cc})

wandb.finish()
