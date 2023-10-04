import gc
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import pandas as pd

from nems0 import get_setting
from nems0.recording import load_recording
from nems0 import xforms, preprocessing, modelspec, epoch

from metrics import compute_CCmax, compute_TTRC


def prepare_nat4(area: str, data_dir: str):

    # =========  LOAD THE DATA  ===========

    signals_dir = "file://"
    datafile = os.path.join(signals_dir, data_dir, f'/{area}_NAT4_ozgf.fs100.ch18.tgz')
    rec = load_recording(datafile)

    context = {'rec': rec}
    context.update(xforms.normalize_sig(sig='stim', norm_method='minmax', log_compress=1, **context))  # normalize spectrograms (log-compression, important)
    context.update(xforms.normalize_sig(sig='resp', norm_method='minmax', **context))  # normalize responses
    context.update(preprocessing.split_pop_rec_by_mask(**context))

    cells = context['rec']['resp'].chans
    val_sounds = epoch.epoch_names_matching(context['rec']['resp'].epochs, "^STIM_00cat")
    est_sounds = epoch.epoch_names_matching(context['rec']['resp'].epochs, "^STIM_cat")


    # =========  EXTRACT STIMULUS SPECTROGRAMS  ===========

    val_spectros = []
    for val_sound in val_sounds:
        val_spectro = context['rec']['stim'].extract_epoch(val_sound)  # (1, F=18, T=150)
        val_spectros.append(torch.from_numpy(val_spectro))
    val_spectros = torch.stack(val_spectros)  # (18, 1, 18, 150) = (S, 1, F, T)

    est_spectros = []
    for est_sound in est_sounds:
        est_spectro = context['rec']['stim'].extract_epoch(est_sound)
        est_spectros.append(torch.from_numpy(est_spectro))
    est_spectros = torch.stack(est_spectros)  # (575, 1, 18, 150) = (S, 1, F, T)


    # =========  EXTRACT CORRESPONDING RESPONSE TRIALS (ESTIMATION SET) ===========

    est_responses = []
    for est_sound in est_sounds:
        est_resp = context['rec']['resp'].extract_epoch(est_sound)  # (1, N=398, T=150)
        est_responses.append(torch.from_numpy(est_resp))
    est_responses = torch.cat(est_responses)  # (575, 398, 150) = (S, N, T)
    est_responses = est_responses.permute(1, 0, 2)  # (N, S, T)
    est_responses = est_responses.unsqueeze(2)  # (N, S, R=1, T)


    # =========  EXTRACT CORRESPONDING RESPONSE TRIALS (VALIDATION SET)  ===========

    del rec
    del context
    gc.collect()

    val_responses = []

    PEG_FILES_LIST = os.listdir(f'../data/{area}_single_sites/')

    for cell in tqdm(cells):
        site_name = cell.split('-')[0]

        for filename in PEG_FILES_LIST:
            if site_name not in filename:
                pass
            else:
                datafile = os.path.join(signals_dir, f'../data/{area}_single_sites/', filename)
                single_site_rec = load_recording(datafile)

                responses = []

                for val_sound in val_sounds:
                    resp = single_site_rec['resp'].rasterize()
                    try:
                        single_cell_resp = resp.extract_channels([cell])
                    except ValueError:
                        if area == 'A1' and 'TNC' in cell:
                            cell = cell[:8] + '0' + cell[8:]
                        single_cell_resp = resp.extract_channels([cell])
                    r = single_cell_resp.extract_epoch(val_sound)  # (20, 1, 1500)
                    r = torch.from_numpy(r).squeeze(1).to(torch.float32)  # (N_repeats, N_timebins)
                    r = r.reshape(20, -1, 10).sum(axis=-1)  # (N_repeats, N_timebins/10): dt=1ms --> dt=10ms
                    responses.append(r)

                    del resp
                    del single_cell_resp
                    gc.collect()

                del single_site_rec
                gc.collect()

                responses = torch.stack(responses)  # (N_sounds, N_repeats, N_timebins)

                break

        val_responses.append(responses)

    val_responses = torch.stack(val_responses)  # (N_neurons, N_sounds, N_repeats, N_timebins)


    # =========  COMPUTE NORMALIZATION FACTORS  ===========

    # 'estimation' stimuli only have R=1 repeat, so normalization factors are set to 1
    N, S, R, T = est_responses.shape
    est_ccmaxes = torch.ones(N, S)
    est_ttrcs = torch.ones(N, S)

    # compute normalization factors for 'validation' stimuli
    N, S, R, T = val_responses.shape
    val_r_temp = torch.flatten(val_responses, start_dim=0, end_dim=1)  # (N*S, R, T)
    val_ccmaxes = compute_CCmax(val_r_temp, max_iters=126)  # (N*S,)
    val_ccmaxes = torch.unflatten(val_ccmaxes, 0, (N, S))  # (N, S)
    val_ttrcs = compute_TTRC(val_r_temp)  # (N*S,)
    val_ttrcs = torch.unflatten(val_ttrcs, 0, (N, S))  # (N, S)


    # =========  MASK FOR 'AUDITORY RESPONSIVE' NEURONS  ===========

    # filter auditory responsive (valid) neurons
    list_neurons = pd.read_csv(f'../data/{area}_pred_correlation.csv')
    auditory_neurons_indices = []

    for i, cell in enumerate(tqdm(cells)):
        cell_auditory = list_neurons.loc[list_neurons['cellid'] == cell]['sig_auditory'].item()
        if cell_auditory == True:
            auditory_neurons_indices.append(i)

    auditory_neurons_indices = torch.tensor(auditory_neurons_indices)  # (N_audi,)


    # =========  WRAP THINGS UP  ===========

    save_dict = {'est_spectrograms': est_spectros,  # (S, 1, F, T)
                 'est_responses': est_responses,  # (N, S, R=1, T)
                 'est_ccmaxes': est_ccmaxes,  # (N, S)
                 'est_ttrcs': est_ttrcs,  # (N, S)
                 'val_spectrograms': val_spectros,  # (S, 1, F, T)
                 'val_responses': val_responses,  # (N, S, R=20, T)
                 'val_ccmaxes': val_ccmaxes,  # (N, S)
                 'val_ttrcs': val_ttrcs,  # (N, S)
                 'auditory': auditory_neurons_indices}  # (N_audi,)

    torch.save(save_dict, os.path.join(data_dir, f'/nat4_{area}.pt'))
    print("Dataset preprocessed & saved !")


if __name__ == "__main__":
    prepare_nat4(area='A1', data_dir='./datasets/NAT4/data/')
    prepare_nat4(area='PEG', data_dir='./datasets/NAT4/data/')
