# CRCNS AA2 Dataset

**Dataset Source:** [AA2 Dataset](https://crcns.org/data-sets/aa/aa-2/about)

**Citation:**
```text
Theunissen, Frederic E.; Gill, Patrick; Noopur, Amin; Zhang, Junli; Woolley, Sarah M. N.; Fremouw, Thane (2011): Single-unit recordings from multiple auditory areas in male zebra finches. CRCNS.org.
http://dx.doi.org/10.6080/10.6080/K0JW8BSC
```

**Papers Using the Dataset:**
- ["Sound representation methods for spectro-temporal receptive field estimation"](https://doi.org/10.1007/s10827-006-7059-4) (2006) by Patrick Gill, Junli Zhang, Sarah M. N. Woolley, Thane Fremouw and Frédéric E. Theunissen
- ["Role of the Zebra Finch Auditory Thalamus in Generating Complex Representations for Natural Sounds"](https://doi.org/10.1152/jn.00128.2010) (2010) by Noopur Amin, Patrick Gill, Frederic E Theunissen

## Dataset Details

**Population fitting:** ✅

**Description of Stimuli:**
- 72 clips of conspecific vocalizations, 20 clips of flat ripples, and 25 clips of song ripples up to 5 s duration.
- Sample rate @ 32 kHz and 16 bit precision
- Up to 10-20 response trials for a given stimulus

**Description of Neurons:**
- Extracellular single-unit recordings from 57 male zebra finches
- Total Number of Neurons: 494 
  - 143 mld 
  - 59 OV 
  - 189 L
    - 17 L1 
    - 53 L2a
    - 42 L2b
    - 43 L3
    - 31 others ("L")
  - 37 CM
  - 66 others ("None")

**Available data:**
- Full Python preprocessing.
- One very simple .txt file for each cell (unit) response:
  - Spike timestamp relative to stimulus onset
  - Each line corresponds to one response trial

**Processing needed (Dataset class init() method):**
- Transforming the sound waveform (.wav file) into a 32-band spectrogram.
- Choosing neurons based on their recording site, stimulus type, and animal.
- Transforming the spike times of each repeat of each stimulus into PSTHs
  - Remove pre-onset spikes
  - Align trials temporally
  - Pad/cut to the right (present/future time steps) so that trials have the same duration


## Benchmark results

TODO

|  **Area**   | **Model backbone** | **Rank** | **Remarks** | **Params / nrn** | **Perfs <br/>(CCraw / CCnorm) [%]** | **Paper (backbone)** | 
|:-----------:|:------------------:|:--------:|:-----------:|:----------------:|:-----------------------------------:|:--------------------:|
|   **mld**   |                    |    🥇    |             |                  |                                     |                      |
|             |                    |    🥈    |             |                  |                                     |                      |
|             |                    |    🥉    |             |                  |                                     |                      |
|   **OV**    |                    |    🥇    |             |                  |                                     |                      |
|             |                    |    🥈    |             |                  |                                     |                      |
|             |                    |    🥉    |             |                  |                                     |                      |
| **L (all)** |                    |    🥇    |             |                  |                                     |                      |
|             |                    |    🥈    |             |                  |                                     |                      |
|             |                    |    🥉    |             |                  |                                     |                      |
|   **CM**    |                    |    🥇    |             |                  |                                     |                      |
|             |                    |    🥈    |             |                  |                                     |                      |
|             |                    |    🥉    |             |                  |                                     |                      |
| **others**  |                    |    🥇    |             |                  |                                     |                      |
|             |                    |    🥈    |             |                  |                                     |                      |
|             |                    |    🥉    |             |                  |                                     |                      |


## Setup

**Requirements**: a [CRCNS account](https://crcns.org/register).

Easiest path — auto-download via the CRCNS NERSC mirror:

```python
from deepSTRF.datasets.audio import CRCNS_AA2_Dataset

ds = CRCNS_AA2_Dataset(
    download=True, dt_ms=5,
    crcns_username="your_username",
    crcns_password="your_password",
)
```

Alternatively set `$CRCNS_USERNAME` / `$CRCNS_PASSWORD`. Default cache dir
is `platformdirs.user_cache_dir('deepSTRF')/CRCNS_AA2`,
overridable via `$DEEPSTRF_DATA_DIR`. `download=True` is idempotent.

If you already have the data laid out manually:
1. Download from [the dataset page](https://crcns.org/data-sets/aa/aa-2/about).
2. Extract `all_stims/` and `all_cells/` into a `data/` folder.
3. `ds = CRCNS_AA2_Dataset('/path/to/data', dt_ms=5)`.

## Filtering

The full selection API from [the data paradigm doc](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional)
is available on AA2: filter neurons by metadata (`select_pop_by_nrn_attr`)
or by stim coverage (`select_pop_by_stim_attr`), filter stims by metadata
(`select_stims_by_attr`), and rely on the bidirectional rule so that
narrowing the stim space automatically hides cells with no responses
left in it.
