# Allen Visual Coding (Ecephys)

**Dataset source**: [Visual Coding](https://observatory.brain-map.org/visualcoding/)

**Citation**: 
```text
@misc{AllenEcephysObservatory,
  author = {Allen Institute for Brain Science},
  title = {Allen Brain Observatory - Neuropixels Visual Coding},
  year = {2019},
  url = {https://observatory.brain-map.org/visualcoding}
}
```

**Associated papers**:
- ["Survey of spiking in the mouse visual system reveals functional hierarchy"](https://doi.org/10.1038/s41593-019-0550-9)
  (2021), Siegle et al.
- ["Sharing neurophysiology data from the Allen Brain Observatory"](https://doi.org/10.7554/eLife.85550)
  (2023), Siegle et al.


## Description

**Population fitting:** ✅

**Description of Stimuli:**

Three natural grayscale videos taken from Orson Welles' 1958 movie _Touch of Evil_.
  - movie #1: 30 secs, 20 trials (valid/test)
  - movie #3: 120 secs, 10 trials (train)

**Description of neural recordings:**
- _**putative**_ neurons (spike-sorted units)
- 6 visual cortex areas
  - VISal
  - VISam
  - VISl
  - VISp
  - VISpm
  - VISrl
- sample rate @ 30 Hz aligned to stimulus frames
- 58 sessions total


## Setup Instructions

**Requirements:** `allensdk` installed (install with pip)

0. Specify the data folder (default: `deepSTRF/datasets/Allen_Ecephys/data/`) in the script `download_allen_ecephys.py` and run it. 
This will download the stimulus and response data files, do a bit of preprocessing and save the corresponding tensors. 
Be patient as the Allen Institute's servers can be very slow. At the end of this process, you will have in your destination folder
two subfolders `natural_movies/` and `responses/`. The file organization will be the following:
```text
data/
 |____ natural_movies/
 |            |_______ natural_movie_1.npy
 |            |_______ natural_movie_3.npy
 |
 |____ responses/
            |_______ session_715093703/
            |           |______ allen_ecephys_resps_VISal_movie1.pt
            |           |______ allen_ecephys_resps_VISal_movie3.pt
            |           |______ allen_ecephys_resps_VISl_movie1.pt
            |           |______ allen_ecephys_resps_VISl_movie3.pt
            |           | ...
            |           |______ allen_ecephys_resps_VISrl_movie1.pt
            |           |______ allen_ecephys_resps_VISrl_movie3.pt
            |           |______ session_715093703.nwb    
            |                                                                   
            |_______ session_719161530/
            |           |______ ...
            |
            |_______ ...
```
1. You can use the `Allen_Ecephys_Dataset` class right way by specifying the above data folder as the input path. E.g.,

```python
from deepSTRF.datasets.video import Allen_Ecephys_Dataset

train_set = Allen_Ecephys_Dataset("deepSTRF/datasets/Allen_OPhys/data/", areas=('VISp',), spat_res=(152, 304), seq_len=75,
                                optim_set='train')
```