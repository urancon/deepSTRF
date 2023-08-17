import os

import numpy as np
import pandas as pd
import torch
from nems0 import epoch
from nems0 import preprocessing
from nems0.recording import load_recording

# ##################################
# ####### SPLITTING ################
# ##################################

# Define data path
signals_dir = "file://"

# Load the recording
datafile = os.path.join(signals_dir, 'data_files_here/A1_NAT4_ozgf.fs100.ch18.tgz')
rec = load_recording(datafile)

# Prepare context
context = {'rec': rec}

# Split recording into estimation and validation sets
recs = preprocessing.split_pop_rec_by_mask(rec=rec)
estimation = recs['est']
validation = recs['val']

"""# Save sets
estimation.save(os.path.join(signals_dir, 'recordings/a1_est/'))
validation.save(os.path.join(signals_dir, 'recordings/a1_val/'))"""

# ##################################
# ####### PROCESSING ###############
# ##################################

rng = np.random.default_rng()

epochs = [
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

stim = validation['stim'].as_matrix(epochs)
resp = validation['resp'].as_matrix(epochs)

val_spectrograms = torch.tensor(stim, dtype=torch.float32)

responses = torch.tensor(resp, dtype=torch.float32)
responses1 = torch.squeeze(responses)
val_responses = torch.permute(responses1, (1,0,2))

list_neurons = pd.read_csv("A1_pred_corr.csv")
good_neurons = list_neurons.loc[list_neurons["sig_auditory"] == True, "idx"].values
good_idx = indices = torch.from_numpy(good_neurons)

epochs = epoch.epoch_occurrences(estimation.epochs, regex="^STIM_").index.to_list()
epochs_to_remove = [
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

for sound in epochs_to_remove:
    epochs.remove(sound)

spectrograms = []
responses = []

for ep in epochs:
    stim = estimation['stim'].as_matrix(ep)
    resp = estimation['resp'].as_matrix(ep)

    spectrogram = torch.tensor(stim, dtype=torch.float32)
    response = torch.tensor(resp, dtype=torch.float32)

    spectrograms.append(spectrogram)
    responses.append(response)

spectrograms_tensor = torch.stack(spectrograms, dim=0)
spectrograms_tensor = spectrograms_tensor.view(575, 1, 18, 150)
est_spectrograms = spectrograms_tensor.to(dtype=torch.float32)

responses_tensor = torch.stack(responses, dim=0)
# responses_tensor = torch.index_select(responses_tensor, dim=3, index=good_idx)
responses_tensor = responses_tensor.view(849, 575, 150)
est_responses = responses_tensor.to(dtype=torch.float32)

ccmax = torch.load('Single/ccmax_a1.pt')
# ccmax = torch.mean(ccmax,dim=1)

auditory = good_idx

torch.save({'est_spectrograms': est_spectrograms, 'est_responses': est_responses, 'val_spectrograms': val_spectrograms, 'val_responses': val_responses, 'ccmax': ccmax, 'auditory': auditory}, "../nat4_a1.pt")
print("Dataset Saved !")
