# Allen Visual Coding (Ophys)

**Dataset source**: [Visual Coding](https://observatory.brain-map.org/visualcoding/)

**Citation**: 
```text
Allen Institute MindScope Program. (2016). Allen Brain Observatory – 2-photon Visual Coding [Dataset]. 
Available from https://brain-map.org/explore/circuits.
```

**Associated papers**:
- ["A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex"](https://doi.org/10.1038/s41593-019-0550-9)
  (2020), de Vries et al.
- ["Sharing neurophysiology data from the Allen Brain Observatory"](https://doi.org/10.7554/eLife.85550)
  (2023), Siegle et al.



## Description

**Population fitting:** ✅

**Description of Stimuli:**

Three natural grayscale videos taken from Orson Welles' 1958 movie _Touch of Evil_.
  - movie #1: 30 secs, 30 trials (test)
  - movie #2: 30 secs, 10 trials (validation)
  - movie #3: 120 secs, 10 trials (train)

**Description of neural recordings:**
- _**actual**_ neurons, not putative clusters (2-Photon Calcium Imaging ==> no spike sorting)
- 6 visual cortex areas
  - VISal
  - VISam
  - VISl
  - VISp
  - VISpm
  - VISrl
- sample rate @ 30 Hz aligned to stimulus frames


## Setup Instructions

**Requirements:** `allensdk` installed (install with pip)

0. Specify the data folder (default: `deepSTRF/datasets/Allen_OPhys/data/`) in the script `download_allen_ophys.py` and run it. 
This will download the stimulus and response data files, do a bit of preprocessing and save the corresponding tensors. 
Be patient as the Allen Institute's servers can be very slow. At the end of this process, you will have in your destination folder
two subfolders `natural_movies/` and `responses/`. The file organization will be the following:
```text
data/
 |____ natural_movies/
 |            |_______ natural_movie_1.npy
 |            |_______ natural_movie_2.npy
 |            |_______ natural_movie_3.npy
 |
 |____ responses/
            |_______ VISp/
            |           |______ allen_ophys_resps_movie1.pt
            |           |______ allen_ophys_resps_movie2.pt
            |           |______ allen_ophys_resps_movie3.pt
            |           |______ exp511507650_movie1.pt
            |           |______ exp511507650_movie2.pt
            |           |______ exp511507650_movie3.pt
            |           | ...
            |           |______ exp6658599_movie3.pt     
            |                                                                   
            |_______ VISl/
            |           |______ ...
            |
            |_______ ...
```
1. You can use the `Allen_OPhys_Dataset` class right way by specifying the above data folder as the input path. E.g.,

```python
from deepSTRF.datasets.video import Allen_Ophys_Dataset

train_set = Allen_Ophys_Dataset("deepSTRF/datasets/Allen_OPhys/data/", areas=('VISp',), spat_res=(152, 304), seq_len=75,
                                optim_set='train')
```