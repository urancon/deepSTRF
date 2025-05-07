import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
pd.set_option('display.max_columns', None)

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


# TODO: make executable with argparse, install for command line ?
#  spike sorting hyperparams and session as optional arguments ? session = why not, spike sorting = no, filtered in the dataset class

# ==========================================
# Helper function
# ==========================================

def get_responses(sess_data, unit_ids, movie_id):
    """
    Helper function to fetch responses for given session, structure, movie, and unit_ids
    """
    if movie_id == 1:
        name = "natural_movie_one"
        n_consecutive_bins = 900    # movie_one is 900 frame long (@ 30Hz)
    elif movie_id == 3:
        name = "natural_movie_three"
        n_consecutive_bins = 3600   # movie_three is 3600 long
    else:
        raise ValueError

    pres = sess_data.stimulus_presentations
    pres_ids = pres[pres.stimulus_name == name].index.values
    first_frame_pres_ids = pres_ids[::n_consecutive_bins]

    dt = 0.033361 # s, or 1/30 Hz
    bin_edges = np.linspace(0.0, n_consecutive_bins * dt, n_consecutive_bins + 1) # relative to stimulus onset / frame presentation

    da = sess_data.presentationwise_spike_counts(
        bin_edges=bin_edges,
        stimulus_presentation_ids=first_frame_pres_ids,
        unit_ids=unit_ids
    )

    data = da.values    # da.values: (repeats, neurons, timebins)
    out = np.transpose(data, (2, 0, 1))     # shape (N, R, T)
    out = torch.from_numpy(out).float()
    return out

# ==========================================
# User configuration
# ==========================================
output_dir = Path("../data").expanduser().resolve()
manifest_path = output_dir / "manifest.json"
structures = ['VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl']
movie_ids = [1, 3]

# Initialize cache
cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

# Load natural movie templates as torch.Tensor (T, C=1, H, W)
# download natural movies on first pass
if 'natural_movies' not in os.listdir(output_dir):
    movies_path = os.path.join(output_dir, 'natural_movies')
    os.mkdir(movies_path)

    for mid in movie_ids:
        da = cache.get_natural_movie_template(mid)
        # da: xarray DataArray shape (T, H, W)
        movie_arr = da.astype(np.float32)
        movie_arr = movie_arr[:, None, :, :]  # add channel dim --> (T, C, H, W)
        print(f"Movie {mid}: tensor shape {movie_arr.shape}")
        np.save(os.path.join(movies_path, f"natural_movie_{mid}"), movie_arr)

# Loop through sessions and structures
sessions = cache.get_session_table()
obs = sessions[sessions.session_type == 'brain_observatory_1.1']

for sid, row in obs.iterrows():
    session_dir = output_dir / f"session_{sid}"
    if session_dir.exists():
        print(f"Session {sid} already processed, skipping.")
        continue
    session_dir.mkdir(parents=True, exist_ok=True)

    # session-level session data
    sess_data = cache.get_session_data(sid, filter_by_validity=True)
    acr_list = row.ecephys_structure_acronyms

    for struct in structures:
        if struct not in acr_list:
            continue
        # select high-quality units in this structure
        units = sess_data.units
        mask = (
            (units.ecephys_structure_acronym == struct) &
            (units.snr >= 1.5) &
            (units.isi_violations <= 0.5) &
            (units.presence_ratio >= 0.9)
        )
        unit_ids = units[mask].index.values
        if len(unit_ids) == 0:
            print(f"No valid units for {struct} in session {sid}")
            continue

        for mid in movie_ids:
            try:
                # fetch responses
                resp = get_responses(sess_data, unit_ids, mid)
                print(f"Session {sid}, {struct}, movie {mid}: {resp.shape}")

                # save responses to session_dir
                out_file = session_dir / f"allen_ecephys_resps_{struct}_movie{mid}.pt"
                torch.save(resp, out_file)
                print(f"Saved responses to {out_file}")

            except Exception as e:
                print(f"Failed fetching movie {mid} for session {sid}, struct {struct}: {e}")
