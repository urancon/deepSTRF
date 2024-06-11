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
from datasets.NAT4_Dataset import NAT4Dataset, NAT4Dataset_pop, NAT4_A1_AUDITORY_NEURONS, NAT4_PEG_AUDITORY_NEURONS
from models.models import Linear, LinearNonlinear, NetworkReceptiveField, DNet, ConvNet2D
from utils.training_pop import set_random_seed, optimize_one_seed, optimize_multiple_seeds


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"\nSelected device: {device}\n")


# dataset
root = '/home/ulysse/Desktop/PhD2/Research/Ulysse/Code/deepSTRF/'
area = 'PEG'
neuron_indices = tuple(range(NAT4_A1_AUDITORY_NEURONS)) if area == 'A1' else tuple(range(NAT4_PEG_AUDITORY_NEURONS))
est_set = NAT4Dataset_pop(path=root +'datasets/NAT4/data', area=area, set='est', neuron_indexes='auditory')
val_set = NAT4Dataset_pop(path=root + 'datasets/NAT4/data', area=area, set='val', neuron_indexes='auditory')
datasets = [est_set, val_set]
F = est_set.get_F()     # number of spectrogram frequency bands
N = est_set.get_N()     # total number of available neurons to fit in the dataset

def data_split_func(datasets):
    est_set = datasets[0]
    train_set, valid_set = torch.utils.data.random_split(est_set, [int(0.8 * len(est_set)), len(est_set) - int(0.8 * len(est_set))])
    test_set = datasets[1]
    return train_set, valid_set, test_set


# optimization
seeds = list(range(1))
n_epochs_early_stop = 50
batch_size = 16
learning_rate = 0.001
weight_decay = 0.
criterion = torch.nn.MSELoss()

# prefiltering / parameterization (if applies) / model architecture hyperparameters
T = 15  # Temporal window size
prefilt_dict = {'type': 'AdapTrans', 'dt': 10.0, 'min_freq': 500, 'max_freq': 20000, 'scale': 'mel'}
param_dict = {'type': 'DCLS', 'num_gauss': 10}
def model_init_fn():
    """ change it as you like """
    # model = Linear(n_frequency_bands=F, temporal_window_size=T, out_neurons=N, prefiltering=prefilt_dict, parameterization=param_dict)
    # model = LinearNonlinear(n_frequency_bands=49, temporal_window_size=T, prefiltering=prefilt_dict, parameterization=param_dict)
    # model = NetworkReceptiveField(n_frequency_bands=49, temporal_window_size=T, n_hidden=20, prefiltering=prefilt_dict, parameterization=param_dict)
    # model = DNet(n_frequency_bands=F, temporal_window_size=T, n_hidden=20, init_tau=2., prefiltering=prefilt_dict, parameterization=param_dict)
    model = ConvNet2D(n_frequency_bands=F, kernel_size=(3, 7), c_hidden=10, n_hidden=90, out_neurons=1, prefiltering=None)
    model = model.to(device)
    return model
net = model_init_fn()
print(net)

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
    "Dataset": est_set.__class__.__name__
}
wandb.init(project=project_name, entity=entity_name, config=config)
print(f"Model: {net.__class__.__name__}, # params: {net.count_trainable_params()}")

# create folder for model saves
savedir = os.path.join(root, f'results/{est_set.__class__.__name__}/{net.__class__.__name__}/')
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
    datasets[0].select_neuron(neuron_idx)
    datasets[1].select_neuron(neuron_idx)

    # train model on multiple splits of this neuron's data and average results
    nrn_res_dict = optimize_multiple_seeds(neuron_idx, seeds, datasets,
                                           data_split_func, model_init_fn,
                                           criterion, learning_rate, weight_decay, batch_size, n_epochs_early_stop, device, savedir
                                           )

    # log results
    wandb.log(nrn_res_dict)

    # for later averaging over seeds
    for key in global_res_dict:
        global_res_dict[key] += nrn_res_dict[key]

# average metrics over neurons
for key in global_res_dict:
    global_res_dict[key] /= len(neuron_indices)

# report average metrics over neurons and over seeds
wandb.log(global_res_dict)

print("job done !")



"""

print("\n#### POPULATION ####\n")
print("Neuron indices:\n", neuron_indices)
est_set.select_population(neuron_indices)
val_set.select_population(neuron_indices)

# train model on multiple splits of this population's data and average results
pop_res_dict = optimize_multiple_seeds(0, seeds, datasets,
                                       data_split_func, model_init_fn,
                                       criterion, learning_rate, weight_decay, batch_size, n_epochs_early_stop, device, savedir
                                       )

# log per-neuron metrics
for i in range(N):
    wandb.log({
        'train_loss': pop_res_dict['train_loss'].item(),
        #'train_CCraw': pop_res_dict['train_CCraw'][i].item(),
        #'train_CCnorm': pop_res_dict['train_CCnorm'][i].item(),
        'val_loss': pop_res_dict['val_loss'].item(),
        #'val_CCraw': pop_res_dict['val_CCraw'][i].item(),
        #'val_CCnorm': pop_res_dict['val_CCnorm'][i].item(),
        'test_loss': pop_res_dict['test_loss'].item(),
        'test_CCraw': pop_res_dict['test_CCraw'][i].item(),
        'test_CCnorm': pop_res_dict['test_CCnorm'][i].item()
    })

# log population metrics (i.e. metrics averaged over neurons)
wandb.log(
{
        'final_train_loss': pop_res_dict['train_loss'].item(),
        #'final_train_CCraw': pop_res_dict['train_CCraw'].mean().item(),
        #'final_train_CCnorm': pop_res_dict['train_CCnorm'].mean().item(),
        'final_val_loss': pop_res_dict['val_loss'].item(),
        #'final_val_CCraw': pop_res_dict['val_CCraw'].mean().item(),
        #'final_val_CCnorm': pop_res_dict['val_CCnorm'].mean().item(),
        'final_test_loss': pop_res_dict['test_loss'].item(),
        'final_test_CCraw': pop_res_dict['test_CCraw'].mean().item(),
        'final_test_CCnorm': pop_res_dict['test_CCnorm'].mean().item()
    }
)

print("job done !")
"""