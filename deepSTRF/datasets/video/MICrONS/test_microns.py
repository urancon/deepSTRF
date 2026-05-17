import os
import numpy as np
import torch
import datajoint as dj

# -----------------------------------------------------------------------------
#  CONFIGURATION  ——— adjust these to your local DataJoint setup
# -----------------------------------------------------------------------------
dj.config['database.host']     = '127.0.0.1'
dj.config['database.user']     = 'microns_user'
dj.config['database.password'] = 'my_password'
dj.config['database.port']     = 3306

# import the cajal microns_phase3_nda schema
import microns_phase3_nda.ingest as ingest  # or wherever Stimulus, Trial, etc live
from microns_phase3_nda import Stimulus, Trial, RidgeTrace, Unit, Scan

# -----------------------------------------------------------------------------
#  1) Find all “oracle” stimulus conditions
# -----------------------------------------------------------------------------
# first, list the unique condition_type values so you can confirm what to select
all_types = np.unique((Stimulus & {}).fetch('condition_type'))
print("Available condition_types:", all_types)

# pick the ones containing “oracle” (should be exactly six entries)
oracle_types = [t for t in all_types if 'oracle' in t.lower()]
print("Will fetch oracle types:", oracle_types)

# -----------------------------------------------------------------------------
#  2) Pull each oracle movie (H×W×T) and pack into a torch tensor
# -----------------------------------------------------------------------------
videos = []
for cond in oracle_types:
    # fetch all distinct Stimulus keys of that type
    keys = (Stimulus & dict(condition_type=cond)).fetch('KEY')
    # for each key, grab its movie
    for key in keys:
        mov = (Stimulus & key).fetch1('movie')  
        # mov is H×W×T; add channel dim => 1×H×W×T, then we’ll stack B×C×H×W×T
        mov = mov.astype(np.float32) / 255.0  # normalize if you like
        videos.append(torch.from_numpy(mov[None, ...]))

# now stack into (B, C, H, W, T)
video_tensor = torch.stack(videos, dim=0)
print("video_tensor shape:", video_tensor.shape)
# should be: torch.Size([6, 1, H, W, T])

# -----------------------------------------------------------------------------
#  3) Fetch the VISp neural responses under those same stimulus conditions
# -----------------------------------------------------------------------------
# First find all VISp units
visp_units = (Unit & dict(brain_area='VISp')).fetch('KEY')
print(f"Found {len(visp_units)} VISp units")

# For each unit, for each repeat of each oracle condition, grab calcium traces
# stored in the table RidgeTrace (or whichever your repo uses).
responses = []
for unit_key in visp_units:
    # RidgeTrace holds traces for each trial, indexed by condition and repeat
    traces = (RidgeTrace & unit_key & dict(condition_type=cond)).fetch(
        'trace', order_by='repeat_id'
    )  # shape: (R,) per cond
    if traces.size > 0:
        # traces is R×T; stack conditions along first axis, then repeats
        responses.append(traces)  

# concatenate over units => a list of arrays [ (R×T), ... ] of length N_units
# we want shape (N, R, T)
response_tensor = torch.stack([torch.from_numpy(r.astype(np.float32)) for r in responses], dim=0)
print("response_tensor shape:", response_tensor.shape)
# should be: torch.Size([N, R, T])

# -----------------------------------------------------------------------------
#  DONE!  You now have:
#    • video_tensor: (B=6, C=1, H, W, T)
#    • response_tensor: (N_visp, R=10, T)
# -----------------------------------------------------------------------------

