import os
import tqdm
import numpy as np
import torch
import pandas as pd

from nems0 import epoch
from nems0 import preprocessing
from nems0.recording import load_recording


A1_FILES_LIST = [
    'A1_TNC015a_abf3608afd9b5c83bf8ed17a5ac61040e9c83261.tgz',
    'A1_TNC020a_2f0624a284401ad0e40680435fc57de3873e0a2c.tgz',
    'A1_ARM030a_9b0586bd962b60ab7b02a21771c98695134b8be8.tgz',
    'A1_TNC008a_2eb34ed73bb45f087a72652b5fcea810a9f2e9c9.tgz',
    'A1_ARM029a_aefada7c479857e55fdafe3e00e94df189250651.tgz',
    'A1_TNC021a_89668bbfb0d89473b8efff14b7b6625bb58060dd.tgz',
    'A1_TNC017a_2d5a6c489bf573e2f762eb2e17ca5478271fcc64.tgz',
    'A1_TNC012a_c677f8485dfd121765cd31de50230196a8a4bcc1.tgz',
    'A1_TNC010a_2f50864a77d8ec2f1786ebd8b0161114a9e905b1.tgz',
    'A1_TNC014a_a0ebe478cdce9612658c29d6766859fd66f1a3b6.tgz',
    'A1_TNC016a_13decf694dbf004e611c8e362e23b6fa7852ee4b.tgz',
    'A1_DRX006b_adb77e6b989c6b08bf7fce0c19ab1a94ba124399.tgz',
    'A1_DRX007a_5409c3cb81c2745cec8af5f9a1402ed24ea7ec7d.tgz',
    'A1_DRX008b_240d4f51148b270f6dc099eaccd4f316ce3021f5.tgz',
    'A1_ARM032a_d28371c918efb9917c560f41e51ff15efdf516c5.tgz',
    'A1_CRD017c_07010f7f2781fc649e4f1f90679212e60a7ff3b4.tgz',
    'A1_CRD016d_f9cf97eab58415d6187e5f104e9df6bf06a9fd99.tgz',
    'A1_TNC013a_cc40dccc6e141410c8b0c16e403b763ad368b170.tgz',
    'A1_ARM031a_e73a3420ba4e26d680d9a8adc5bef1c32f6d9617.tgz',
    'A1_ARM033a_8bc7cdda34517574d7973a1b6352d52d873bad7b.tgz',
    'A1_TNC009a_91819235d1188908cee2787e5769d3613fbd756f.tgz',
    'A1_TNC018a_2d5a31aeb27af29f52739c37061da96fcb058e96.tgz'
]

# 'PEG_TNC022a_5da19f65eab678631540a3074cefc14e80dd076c.tgz' was removed to have only the 398 neurons of the study
PEG_FILES_LIST = [
    'PEG_ARM017a_78f8cd0f60b245f8dd137e6b00306c27f0cae9af.tgz',
    'PEG_ARM018a_5e04645168a032cf445a5246a5865c1977a2378c.tgz',
    'PEG_ARM019a_54195bae38f79360e61248914f2f035ead3c0279.tgz',
    'PEG_ARM021b_ee2e2266509d78ceadee0da7906dadeb049c8909.tgz',
    'PEG_ARM022b_5e0509b8a4bf6dd7ad890b620f1313941612e49a.tgz',
    'PEG_ARM023a_55b620d7fd2958d3201d347d75c6b1be9aa757d9.tgz',
    'PEG_ARM024a_b97ebd6d2780914105a1562ce19942dacaf8abb5.tgz',
    'PEG_ARM025a_b6cabae20ac9a9d2863fa68569037b29fb1b62ed.tgz',
    'PEG_ARM026b_f230bfe4a990c4a8eadcb843bbcb4deb79c837c9.tgz',
    'PEG_ARM027a_c36811efdff0d4136be330e77953f6c6e69c7aec.tgz',
    'PEG_ARM028b_f2005ed600d22a854579b7735758144c15c56f46.tgz',
]

VAL_EPOCHS = [
    "STIM_00cat172_rec1_geese_excerpt1.wav",
    "STIM_00cat221_rec1_laughing_excerpt1.wav",
    "STIM_00cat232_rec1_man_speaking_excerpt1.wav",
    "STIM_00cat28_rec1_bigband_bbc-big-band_american-patrol_0sec_excerpt1.wav",
    "STIM_00cat337_rec1_siren_excerpt1.wav",
    "STIM_00cat401_rec1_walking_with_heels_excerpt1.wav",
    "STIM_00cat414_rec1_woman_speaking_excerpt1.wav",
    "STIM_00cat516_rec1_stream_excerpt1.wav",
    "STIM_00cat634_rec1_drumming-jazz_various-artists_fusion-drum-solo_0sec_excerpt1.wav",
    "STIM_00cat668_rec1_ferret_fights_Athena-Violet001_excerpt1.wav",
    "STIM_00cat668_rec3_ferret_kits_51p9-8-10-11-12-13-14_excerpt1.wav",
    "STIM_00cat668_rec7_ferret_oxford_male_chopped_excerpt1.wav",
    "STIM_00cat669_rec11_marmoset_twitter_excerpt1.wav",
    "STIM_00cat669_rec2_marmoset_chirp_excerpt1.wav",
    "STIM_00cat669_rec4_marmoset_phee_2_excerpt1.wav",
    "STIM_00cat78_rec1_chimes_in_the_wind_excerpt1.wav",
    "STIM_00cat83_rec1_cicadas_excerpt1.wav",
    "STIM_00cat92_rec1_country_song_excerpt1.wav"
]

N_NEURONS_A1 = 849
N_NEURONS_PEG = 398


def save_response_trials_nat4(data_folder_path, file_list, out_path):
    """
    Extracts and saves the response trials for the validation set of NAT4

    Therefore creates and saves a (N_neuron, N_repeats, N_sounds) tensor

    """

    #  LOAD THE DATA
    rng = np.random.default_rng()
    signals_dir = "file://"
    recs = []           # list to store all the recordings
    N_neurons = 0       # total number of neurons

    for filebase in file_list:
        datafile = os.path.join(signals_dir, data_folder_path, filebase)
        rec = load_recording(datafile)
        recs.append(rec)
        N_neurons += len(rec['resp'].chans)

    val_responses_all_neurons = []

    for rec in tqdm.tqdm(recs):
        cells = rec['resp'].chans
        resp = rec['resp'].rasterize()
        sounds = epoch.epoch_names_matching(resp.epochs, "^STIM_00")

        for neuron_idx, cell in enumerate(cells):
            single_cell_resp = resp.extract_channels([cell])
            val_responses_neuron = []

            for id, sound in enumerate(sounds):
                # extract responses to a single validation stimulus
                r = single_cell_resp.extract_epoch(sound)
                r = torch.from_numpy(r).squeeze(1).to(torch.float32)    # (N_repeats, N_timebins)
                r = r.reshape(20, -1, 10).sum(axis=-1)                   # (N_repeats, N_timebins/10): dt=1ms --> dt=10ms
                val_responses_neuron.append(r)

            val_responses_neuron = torch.stack(val_responses_neuron)    # (N_sounds, N_repeats, N_timebins)
            val_responses_all_neurons.append(val_responses_neuron)

    val_responses_all_neurons = torch.stack(val_responses_all_neurons)  # (N_neurons, N_sounds, N_repeats, N_timebins)

    torch.save(val_responses_all_neurons, out_path)
    print("Computed & Saved validation Response Trials !")


def preprocess_dataset_nat4(data_folder_path, csv_path, out_path, val_trials_path=None):
    """
    Creates a dictionary containing estimation and test spectrograms and responses.


    """

    # ##################################
    # ####### SPLITTING ################
    # ##################################

    # Define data path
    signals_dir = "file://"

    # Load the recording
    datafile = os.path.join(signals_dir, data_folder_path)
    rec = load_recording(datafile)

    # Split recording into estimation and validation sets
    recs = preprocessing.split_pop_rec_by_mask(rec=rec)
    estimation = recs['est']
    validation = recs['val']

    # ##################################
    # ####### PROCESSING ###############
    # ##################################

    ##### 'VALIDATION' SET ######

    stim = validation['stim'].as_matrix(VAL_EPOCHS)
    val_spectrograms = torch.tensor(stim, dtype=torch.float32)

    val_responses = torch.load(val_trials_path)

    # filter auditory responsive (valid) neurons
    list_neurons = pd.read_csv(csv_path)
    good_neurons = list_neurons.loc[list_neurons["sig_auditory"] == True].index
    good_idx = torch.tensor(good_neurons, dtype=torch.int64)

    ##### 'ESTIMATION' SET ######

    # only keep 'estimation' sounds
    epochs = epoch.epoch_occurrences(estimation.epochs, regex="^STIM_").index.to_list()
    for sound in VAL_EPOCHS:
        epochs.remove(sound)

    est_spectrograms = []
    est_responses = []

    for ep in epochs:
        stim = estimation['stim'].as_matrix(ep)
        resp = estimation['resp'].as_matrix(ep)

        spectrogram = torch.tensor(stim, dtype=torch.float32)
        response = torch.tensor(resp, dtype=torch.float32)

        est_spectrograms.append(spectrogram)
        est_responses.append(response)

    est_spectrograms = torch.stack(est_spectrograms, dim=0)
    est_spectrograms = est_spectrograms.squeeze(1)      # (N_sounds, 1, N_bands, N_timebins)
    est_spectrograms = est_spectrograms.to(dtype=torch.float32)

    responses_tensor = torch.stack(est_responses, dim=0)
    # responses_tensor = torch.index_select(responses_tensor, dim=3, index=good_idx)    # select good neurons only
    responses_tensor = responses_tensor.squeeze()           # (N_sounds, N_neurons, N_timebins)
    responses_tensor = responses_tensor.permute(1, 0, 2)    # (N_neurons, N_sounds, N_timebins)
    responses_tensor = responses_tensor.unsqueeze(2)        # (N_neurons, N_sounds, N_repeats=1, N_timebins)
    est_responses = responses_tensor.to(dtype=torch.float32)

    save_dict = {'est_spectrograms': est_spectrograms,
                 'est_responses': est_responses,
                 'val_spectrograms': val_spectrograms,
                 'val_responses': val_responses,
                 'auditory': good_idx}

    torch.save(save_dict, out_path)
    print("Dataset preprocessed & saved !")


if __name__ == "__main__":
    # preprocess A1 data
    #save_response_trials_nat4("../data/A1_single_sites", A1_FILES_LIST, out_path="../data/val_trials_a1.pt")
    preprocess_dataset_nat4(data_folder_path="../data/A1_NAT4_ozgf.fs100.ch18.tgz",
                            csv_path="../data/A1_pred_correlation.csv",
                            out_path="../data/nat4_a1.pt",
                            val_trials_path="../data/val_trials_a1.pt")

    # preprocess PEG data
    save_response_trials_nat4("../data/PEG_single_sites", PEG_FILES_LIST, out_path="../data/val_trials_peg.pt")
    preprocess_dataset_nat4(data_folder_path="../data/PEG_NAT4_ozgf.fs100.ch18.tgz",
                            csv_path="../data/PEG_pred_correlation.csv",
                            out_path="../data/nat4_peg.pt",
                            val_trials_path="../data/val_trials_peg.pt")
