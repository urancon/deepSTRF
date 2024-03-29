import os
import io
import random
import torch
import numpy as np
import torch.backends.cudnn
from torch.utils.data import DataLoader

from metrics.performance import *


#############
# 	RNG 	#
#############

def set_random_seed(seed):
    random.seed(seed)                           # Python
    np.random.seed(seed)                        # NumPy
    torch.manual_seed(seed)                     # PyTorch
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)


#################
# 	TRAINING 	#
#################


def optimize_multiple_seeds(nrn_idx, seeds, dataset, data_split_func, model_init_func, criterion, learning_rate, weight_decay, batch_size, n_epochs_early_stop, device, savedir):
    """ Train, validate during several epochs until convergence, for multiple seeds, and report average metrics """

    # per-neuron metrics (averaged over seeds for each neuron)
    nrn_res_dict = {
        'best_epoch': 0,
        'train_loss': 0., 'train_CCraw': 0., 'train_CCnorm': 0.,
        'val_loss': 0., 'val_CCraw': 0., 'val_CCnorm': 0.,
        'test_loss': 0., 'test_CCraw': 0., 'test_CCnorm': 0.
    }

    for seed in seeds:

        print(f"\n==== SEED {seed} ====")
        set_random_seed(seed)

        # split dataset into train/val/test
        train_set, val_set, test_set = data_split_func(dataset)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

        # initialize new model
        model = model_init_func()

        # instanciate optimizer for the new model
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # training / validation / test process
        modeldir = os.path.join(savedir, f"nrn{nrn_idx}_seed{seed}.pt")
        seed_res_dict = optimize_one_seed(train_dataloader, val_dataloader, test_dataloader,
                                          model, criterion, optimizer, n_epochs_early_stop, device, modeldir)

        # for later averaging over seeds
        for key in nrn_res_dict:
            nrn_res_dict[key] += seed_res_dict[key]

    # average metrics over seeds
    for key in nrn_res_dict:
        nrn_res_dict[key] /= len(seeds)

    return nrn_res_dict


def optimize_one_seed(train_dataloader, val_dataloader, test_dataloader, model, criterion, optimizer, n_epochs_early_stop, device, savedir):
    """ Train, validate during several epochs until convergence, for one seed """

    best_val_loss = float('inf')
    epoch = 0
    best_epoch = -1
    n_epochs_no_improvement = 0

    while n_epochs_no_improvement < n_epochs_early_stop:

        # train 1 epoch on training set, then evaluate model on validation set
        epoch_train_loss, epoch_train_cc, epoch_train_cc_norm = train_one_epoch(train_dataloader, model, criterion, optimizer, device)
        epoch_val_loss, epoch_val_cc, epoch_val_cc_norm = evaluate(val_dataloader, model, criterion, device)

        # save trained model if it has improved
        if epoch_val_loss < best_val_loss:
            torch.save(model.state_dict(), savedir)
            best_epoch = epoch
            n_epochs_no_improvement = 0
            best_val_loss = epoch_val_loss
            best_val_cc = epoch_val_cc
            best_val_cc_norm = epoch_val_cc_norm
            best_train_cc = epoch_train_cc
            best_train_cc_norm = epoch_train_cc_norm
            best_train_loss = epoch_train_loss
        else:
            n_epochs_no_improvement += 1

        epoch += 1

    # finally load the best saved model and evaluate it on the test set
    model.load_state_dict(torch.load(savedir))
    test_loss, test_cc, test_cc_norm = evaluate(test_dataloader, model, criterion, device)

    # save and possibly print final results for this seed
    seed_res_dict = {
        'best_epoch': best_epoch,
        'train_loss': best_train_loss, 'train_CCraw': best_train_cc, 'train_CCnorm': best_train_cc_norm,
        'val_loss': best_val_loss, 'val_CCraw': best_val_cc, 'val_CCnorm': best_val_cc_norm,
        'test_loss': test_loss, 'test_CCraw': test_cc, 'test_CCnorm': test_cc_norm
    }
    print(seed_res_dict)

    return seed_res_dict


def train_one_epoch(train_dataloader, model, criterion, optimizer, device):
    """ Optimization through gradient descent, each training sample seen once (1 epoch) """

    # #### TRAIN ##### #
    epoch_train_loss = 0.
    epoch_train_cc = 0.
    epoch_train_cc_norm = 0.
    model.train()

    for spectrogram, responses, ccmax, ttrc in train_dataloader:

        # data preprocessing
        spectrogram = spectrogram.to(device).float()
        responses = responses.to(device)
        ccmax = ccmax.to(device)
        psth = responses.mean(dim=1)  # (B, R, T) --> (B, T)  # TODO: (B, N, T)

        # feed to the ANN network
        prediction = model(spectrogram)
        prediction = prediction.squeeze(-1)  # (B, T)  # TODO: (B, N, T)

        # gradient descent step
        loss = criterion(prediction, psth)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if any, detach stateful variables from graph
        model.detach()

        # correlation coefficient
        cc = correlation_coefficient(prediction, psth)
        ccnorm = normalized_correlation_coefficient(prediction, responses, method='hsu', precomputed_ccmaxes=ccmax)

        # adding the metrics values for one sound
        epoch_train_loss += loss.item()
        epoch_train_cc += cc.item()
        epoch_train_cc_norm += ccnorm.item()

    # averaging the total of the metrics over all the sounds to get the mean values for 1 epoch
    epoch_train_loss /= len(train_dataloader)
    epoch_train_cc /= len(train_dataloader)
    epoch_train_cc_norm /= len(train_dataloader)

    return epoch_train_loss, epoch_train_cc, epoch_train_cc_norm


def evaluate(valtest_dataloader, model, criterion, device):
    """ Just for evaluation, no gradient descent here """

    # #### EVAL ##### #
    epoch_valtest_loss = 0.
    epoch_valtest_cc = 0.
    epoch_valtest_cc_norm = 0.
    model.eval()

    for spectrogram, responses, ccmax, ttrc in valtest_dataloader:

        # data preprocessing
        spectrogram = spectrogram.to(device).float()
        responses = responses.to(device)
        ccmax = ccmax.to(device)
        psth = responses.mean(dim=1)  # (B, R, T) --> (B, T)  # TODO: (B, N, T)

        # feed to the ANN network
        prediction = model(spectrogram)
        prediction = prediction.squeeze(-1)  # (B, T)  # TODO: (B, N, T)

        # no gradient descent step for evaluation = validation / testing
        loss = criterion(prediction, psth)

        # if any, detach stateful variables from graph
        model.detach()

        # correlation coefficient
        cc = correlation_coefficient(prediction, psth)
        ccnorm = normalized_correlation_coefficient(prediction, responses, method='hsu', precomputed_ccmaxes=ccmax)

        # adding the metrics values for one sound
        epoch_valtest_loss += loss.item()
        epoch_valtest_cc += cc.item()
        epoch_valtest_cc_norm += ccnorm.item()

    # averaging the total of the metrics over all the sounds to get the mean values for 1 epoch
    epoch_valtest_loss /= len(valtest_dataloader)
    epoch_valtest_cc /= len(valtest_dataloader)
    epoch_valtest_cc_norm /= len(valtest_dataloader)

    return epoch_valtest_loss, epoch_valtest_cc, epoch_valtest_cc_norm
