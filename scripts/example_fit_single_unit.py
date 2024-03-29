# TODO:
#   - choose save dir
#   - wandb args (entity, project, run)
#   - verbose

# TODO (others):
#   - argparse version of this script for easier usage
#   - multiprocess version (1 seed = 1 process) ?


import os
import time
from tqdm import tqdm
import numpy as np
import wandb
import torch.utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.Wehr_Dataset import WehrDataset, WEHR_NEURONS_SPLIT_NATURAL, WEHR_VALID_NEURONS
from datasets.NS1_DRC_Dataset import NS1_DRC_Dataset, RAHMAN_TRAINVAL_SET_INDICES, RAHMAN_TEST_SET_INDICES
from models.models import Linear, LinearNonlinear, NetworkReceptiveField, DNet
from utils.training import set_random_seed, optimize_one_seed, optimize_multiple_seeds


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"\nSelected device: {device}\n")


# dataset
root = '/home/ulysse/Desktop/PhD2/Research/Ulysse/Code/deepSTRF/'
neuron_indices = tuple(range(73))
dataset = NS1_DRC_Dataset(path=root + 'datasets/NS1_DRC/data', stimuli=('nat',), neuron_indexes=neuron_indices)
F = dataset.get_F()     # number of spectrogram frequency bands

def data_split_func(dataset):
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [14, 2, 4])
    return train_set, valid_set, test_set


# optimization
seeds = list(range(10))
n_epochs_early_stop = 50
batch_size = 1
learning_rate = 0.001
weight_decay = 0.
criterion = torch.nn.MSELoss()

# prefiltering / parameterization (if applies) / model architecture hyperparameters
T = 15  # Temporal window size
prefilt_dict = {'type': 'AdapTrans', 'dt': 1.0, 'min_freq': 500, 'max_freq': 20000, 'scale': 'mel'}
param_dict = {'type': 'DCLS', 'num_gauss': 10}
def model_init_fn():
    """ change it as you like """
    model = Linear(n_frequency_bands=F, temporal_window_size=T, prefiltering=prefilt_dict, parameterization=param_dict)
    # net = LinearNonlinear(n_frequency_bands=49, temporal_window_size=T, prefiltering=prefilt_dict, parameterization=param_dict)
    # net = NetworkReceptiveField(n_frequency_bands=49, temporal_window_size=T, n_hidden=20, prefiltering=prefilt_dict, parameterization=param_dict)
    # net = DNet(n_frequency_bands=F, temporal_window_size=T, n_hidden=20, init_tau=2., prefiltering=prefilt_dict, parameterization=param_dict)
    model = model.to(device)
    return model
net = model_init_fn()

# Weights & Biases logging
os.environ["WANDB_MODE"] = "offline"  # comment out for online logging
project_name = 'deepSTRF'
entity_name = 'urancon'
config = {
    "seeds": seeds,
    "temporal_window_size": T,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "model": net.__class__.__name__,
    "Nb of parameters": net.count_trainable_params(),
    "Dataset": dataset.__class__.__name__
}
wandb.init(project=project_name, entity=entity_name, config=config)
print(f"Model: {net.__class__.__name__}, # params: {net.count_trainable_params()}")

# create folder for model saves
root_path_abs = '/home/ulysse/Desktop/PhD2/Research/Ulysse/Code/deepSTRF/scripts'
savedir = os.path.join(root_path_abs, f'results/{dataset.__class__.__name__}/{net.__class__.__name__}/')
if not os.path.exists(savedir):
    os.makedirs(savedir)

# metrics averaged over neurons and over seeds
global_res_dict = {
    'best_epoch': 0,
    'train_loss': 0., 'train_CCraw': 0., 'train_CCnorm': 0.,
    'val_loss': 0., 'val_CCraw': 0., 'val_CCnorm': 0.,
    'test_loss': 0., 'test_CCraw': 0., 'test_CCnorm': 0.
}

for neuron_idx in neuron_indices:

    print(f"\n#### NEURON {neuron_idx} ####\n")
    dataset.select_neuron(neuron_idx)

    # train model on multiple splits of this neuron's data and average results
    nrn_res_dict = optimize_multiple_seeds(neuron_idx, seeds, dataset,
                                           data_split_func, model_init_fn,
                                           criterion, learning_rate, weight_decay, batch_size, n_epochs_early_stop, device, savedir
                                           )

    # log results
    wandb.log(nrn_res_dict)

    # for later averaging over seeds
    for key in global_res_dict:
        global_res_dict[key] += nrn_res_dict[key]

# average metrics over seeds
for key in global_res_dict:
    global_res_dict[key] /= len(neuron_indices)

# report average metrics over neurons and over seeds
wandb.log(global_res_dict)

print("job done !")
