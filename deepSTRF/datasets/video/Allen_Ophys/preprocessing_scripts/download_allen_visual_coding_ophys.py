# Freely adapted from the Allen SDK's documentation
#
# Resources:
# - https://allenswdb.github.io/physiology/ophys/visual-coding/vc2p-stimuli.html


import os
import numpy as np
import torch
import allensdk
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from pathlib import Path
import json
from IPython.display import display
from PIL import Image

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


# ==========================================
# OPHYS RESPONSE DATA
# ==========================================

# Example cache directory path, it determines where downloaded data will be stored
output_dir = "../data/"

boc =  BrainObservatoryCache(manifest_file='boc/manifest.json')
# boc.get_all_targeted_structures() --> ['VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl']
# boc.get_all_stimuli() --> ['drifting_gratings', ..., 'natural_movie_one', 'natural_movie_three', 'natural_movie_two', 'natural_scenes', ...]

# get experiments for neurons from all areas and from any Cre
structures = boc.get_all_targeted_structures()

for structure in structures:
    print(f"\n ======== Downloading {structure} responses ========\n")

    # 540 experiments for all Cre lines and structures
    exps = boc.get_experiment_containers(targeted_structures=[structure], cre_lines=boc.get_all_cre_lines())

    # response folder
    resps_path = os.path.join(output_dir, f'responses/{structure}/')
    if structure not in os.listdir(os.path.join(output_dir, f'responses/')):
        os.mkdir(resps_path)

    N_neurons = 0
    movie1_resps_pop = []
    movie2_resps_pop = []
    movie3_resps_pop = []

    for exp in exps:
        experiment_container_id = exp['id']
        print(f"\nprocessing experiment {experiment_container_id}...")

        # always 3 sessions per exp container (see doc here:
        # https://allenswdb.github.io/physiology/ophys/visual-coding/vc2p-dataset.html)
        sessions = boc.get_ophys_experiments(experiment_container_ids=[experiment_container_id])

        session_A = boc.get_ophys_experiments(experiment_container_ids=[experiment_container_id], session_types=['three_session_A'])                        # movies 1 & 3
        session_B = boc.get_ophys_experiments(experiment_container_ids=[experiment_container_id],session_types=['three_session_B'])                         # movies 1
        session_C = boc.get_ophys_experiments(experiment_container_ids=[experiment_container_id],session_types=['three_session_C', 'three_session_C2'])     # movies 1 & 3

        if (len(session_A) + len(session_B) + len(session_C)) < 3:
            print(f"no data for exp {experiment_container_id}, skipping..." )
            continue

        # get session ids
        id_A = session_A[0]['id']
        id_B = session_B[0]['id']
        id_C = session_C[0]['id']

        # get response data
        data_set_A = boc.get_ophys_experiment_data(id_A)
        data_set_B = boc.get_ophys_experiment_data(id_B)
        data_set_C = boc.get_ophys_experiment_data(id_C)

        # download natural movies on first pass
        if 'natural_movies' not in os.listdir(output_dir):
            movies_path = os.path.join(output_dir, 'natural_movies')
            os.mkdir(movies_path)

            movie1 = data_set_A.get_stimulus_template('natural_movie_one')
            movie2 = data_set_C.get_stimulus_template('natural_movie_two')
            movie3 = data_set_A.get_stimulus_template('natural_movie_three')

            np.save(os.path.join(movies_path, f'natural_movie_1.npy'), movie1)
            np.save(os.path.join(movies_path, f'natural_movie_2.npy'), movie2)
            np.save(os.path.join(movies_path, f'natural_movie_3.npy'), movie3)

        # same id for all sessions, but different index
        # only care about neurons identified across all three sessions
        cell_ids_A = data_set_A.get_cell_specimen_ids()
        cell_ids_B = data_set_B.get_cell_specimen_ids()
        cell_ids_C = data_set_C.get_cell_specimen_ids()
        cell_ids = np.intersect1d(np.intersect1d(cell_ids_A, cell_ids_B), cell_ids_C)
        N_neurons += len(cell_ids)

        # get the indices of these cells in each session (ids stay the same, not indices)
        cell_idces_A = data_set_A.get_cell_specimen_indices(cell_ids)
        cell_idces_B = data_set_B.get_cell_specimen_indices(cell_ids)
        cell_idces_C = data_set_C.get_cell_specimen_indices(cell_ids)

        # traces
        ts_A, dff_A = data_set_A.get_dff_traces()   # (Timesteps,) & (N_neurons, Timesteps)
        ts_B, dff_B = data_set_B.get_dff_traces()
        ts_C, dff_C = data_set_C.get_dff_traces()

        # events
        events_A = boc.get_ophys_experiment_events(ophys_experiment_id=id_A)  # (N_neurons, Timesteps)
        events_B = boc.get_ophys_experiment_events(ophys_experiment_id=id_B)
        events_C = boc.get_ophys_experiment_events(ophys_experiment_id=id_C)

        # ------ MOVIE 3 -------

        movie3_table = data_set_A.get_stimulus_table('natural_movie_three')  # movie 3 is 2 minutes long and is presented a total of 10 times, but in two epochs
        movie3_resps = events_A[:, movie3_table['start']]  # (T=36000, N)
        movie3_resps = np.stack(np.split(movie3_resps, 10, axis=1), axis=1)  # (N, R=10, T=36000)
        movie3_resps = movie3_resps[cell_idces_A]

        # ------ MOVIE 2 -------

        movie2_table = data_set_C.get_stimulus_table('natural_movie_two')   # movie 2
        movie2_resps = events_C[:, movie2_table['start']]     # (T=9000, N)
        movie2_resps = np.stack(np.split(movie2_resps, 10, axis=1), axis=1)      # (N, R=10, T=900)
        movie2_resps = movie2_resps[cell_idces_C]

        # ------ MOVIE 1 -------

        movie1A_table = data_set_A.get_stimulus_table('natural_movie_one') # movie 1 is in all three sessions --> 30 repeats
        movie1A_resps = events_A[:, movie1A_table['start']]
        movie1A_resps = np.stack(np.split(movie1A_resps, 10, axis=1), axis=1)   # (N1, R=10, T=900)

        movie1B_table = data_set_B.get_stimulus_table('natural_movie_one')
        movie1B_resps = events_B[:, movie1B_table['start']]
        movie1B_resps = np.stack(np.split(movie1B_resps, 10, axis=1), axis=1)   # (N2, R=10, T=900)

        movie1C_table = data_set_C.get_stimulus_table('natural_movie_one')
        movie1C_resps = events_C[:, movie1C_table['start']]
        movie1C_resps = np.stack(np.split(movie1C_resps, 10, axis=1), axis=1)   # (N3, R=10, T=900)

        # Because N1, N2 and N3 can be different, let's take the units present across all three sessions !
        movie1A_resps = movie1A_resps[cell_idces_A]
        movie1B_resps = movie1B_resps[cell_idces_B]
        movie1C_resps = movie1C_resps[cell_idces_C]

        # concatenate repeats gotten out of the three sessions
        movie1_resps = np.concatenate([movie1A_resps, movie1B_resps, movie1C_resps], axis=1)

        # save on the go
        torch.save(movie1_resps, os.path.join(resps_path, f'exp{experiment_container_id}_movie1.pt'))
        torch.save(movie2_resps, os.path.join(resps_path, f'exp{experiment_container_id}_movie2.pt'))
        torch.save(movie3_resps, os.path.join(resps_path, f'exp{experiment_container_id}_movie3.pt'))

        # for later
        movie1_resps_pop.append(movie1_resps)
        movie2_resps_pop.append(movie2_resps)
        movie3_resps_pop.append(movie3_resps)

    # concatenate traces along nrn dim
    movie1_resps_pop = np.stack(movie1_resps_pop, dim=0)    # (N, R=30, T=900)
    movie2_resps_pop = np.stack(movie2_resps_pop, dim=0)    # (N, R=10, T=900)
    movie3_resps_pop = np.stack(movie3_resps_pop, dim=0)    # (N, R=10, T=3600)

    # save files
    torch.save(movie1_resps_pop, os.path.join(resps_path, f'allen_ophys_resps_{structure}_movie1.pt'))
    torch.save(movie2_resps_pop, os.path.join(resps_path, f'allen_ophys_resps_{structure}_movie2.pt'))
    torch.save(movie3_resps_pop, os.path.join(resps_path, f'allen_ophys_resps_{structure}_movie3.pt'))

print("finished job!")
