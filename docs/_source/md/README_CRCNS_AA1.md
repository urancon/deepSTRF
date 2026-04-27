# CRCNS AA1 Dataset

**Dataset Source:** [AA1 Dataset](https://crcns.org/data-sets/aa/aa-1/about)

**Citation:**
```text
Theunissen, FE; Hauber, ME; Woolley, SMN; Gill, P; Shaevitz, SS; Amin, Noopur; Hsu, A; Singh, NC; Grace, GA; Fremouw, T; Zhang, Junli; Cassey, P; Doupe, AJ; David, SV (2009): Single-unit recordings from two auditory areas in male zebra finches. CRCNS.org.
http://dx.doi.org/10.6080/K0F769GP
```

**Papers Using the Dataset:**
- ["Tuning for Spectro-temporal Modulations: a Mechanism for Auditory Discrimination of Natural Sound"](https://doi.org/10.1038/nn1536) by Woolley S., Fremouw T., Hsu A. and Theunissen F.E.
- ["Modulation and phase spectrum of natural sounds enhance neural discrimination performed by single auditory neurons"](https://doi.org/10.1523/JNEUROSCI.2449-04.2004) by Hsu A., Woolley S., Fremouw T. and Theunissen F.E.
- ["Modulation spectra of natural sounds and ethological theories of auditory processing"](https://doi.org/10.1121/1.1624067) by Singh N. and Theunissen F.E.


## Dataset Details

**Population fitting:** ✅

**Description of Stimuli:**
- 10 clips of conspecific vocalizations and 20 clips of flat ripples, up to 5 s duration.
- Sample rate @ 32 kHz and 16 bit precision
- Up to 10 response trials for a given sound

**Description of Neurons:**
- Total Number of Neurons: 100 (50 MLd + 50 Field L)
- Extracellular single-unit recordings from anesthetized male zebra finches

**Available data:**
- Full Python preprocessing.
- One very simple .txt file for each cell (unit) response:
  - Spike timestamp relative to stimulus onset
  - Each line corresponds to one response trial

**Processing needed (Dataset class init() method):**
- Transforming the sound waveform (.wav file) into a 32-band spectrogram.
- Choosing neurons based on their recording site and stimulus type.
- Transforming the spike times of each repeat of each stimulus into PSTHs
  - Remove pre-onset spikes
  - Align trials temporally
  - Pad/cut to the right (present/future time steps) so that trials hvae the same duration


## Benchmar results

|   **Area**    | **Model backbone** | **Rank** | **Remarks** |     **Params / nrn**      | **Perfs <br/>(CCraw / CCnorm) [%]** |                                        **Paper (backbone)**                                         | 
|:-------------:|:------------------:|:--------:|:-----------:|:-------------------------:|:-----------------------------------:|:---------------------------------------------------------------------------------------------------:|
| **Field L**   |      StateNet      |    🥇    |  GRU, pop   |          24,900           |               / 71.0                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|               |    Transformer     |    🥈    |     pop     |          29,109           |               / 65.5                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|               |       2D-CNN       |    🥉    |     pop     |          26,915           |               / 65.0                | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|    **MLd**    |      StateNet      |    🥇    | Mamba, pop  |          32,334           |               / 73.4                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|               |       2D-CNN       |    🥈    |     pop     |          29,109           |               / 68.9                |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|               |    Transformer     |    🥉    |     pop     |          34,475           |               / 68.3                | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |



## Setup

**Requirements**: a [CRCNS account](https://crcns.org/register) (the data
host requires login).

Easiest path — auto-download via the CRCNS NERSC mirror:

```python
from deepSTRF.datasets.audio import CRCNS_AA1_Dataset

ds = CRCNS_AA1_Dataset(
    download=True, dt_ms=5,
    crcns_username="your_username",
    crcns_password="your_password",
)
```

Alternatively, set `$CRCNS_USERNAME` / `$CRCNS_PASSWORD` in the env and
omit the credential kwargs. Default cache dir is
`platformdirs.user_cache_dir('deepSTRF')/CRCNS_AA1`, overridable via
`$DEEPSTRF_DATA_DIR`. `download=True` is idempotent.

If you already have the data laid out manually:
1. Download `crcns-aa1.zip` at [the original dataset repository](https://crcns.org/data-sets/aa/aa-1/about).
2. Extract `all_stims/`, `Field_L_cells/`, `MLd_cells/` into a `data/`
   folder.
3. `ds = CRCNS_AA1_Dataset('/path/to/data', dt_ms=5)`.

## Filtering

Each `stim_meta` dict carries `name`, `type` (`"conspecific"` or
`"flatrip"`), `sample_rate`, `n_samples`, `duration_s`. Each
`neuron_metadata` dict carries `cell_id`, `area` (`"Field_L"` or
`"MLd"`), `animal_id`, `cell_seq`, `rig`. Combined with the
[base-class selection API](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional):

```python
ds.select_pop_by_nrn_attr("area", "MLd")           # only MLd cells
ds.select_stims_by_attr("type", "conspecific")     # only conspecific stims
                                                   # (auto-hides 2 cells with
                                                   # no conspecific data)
```
