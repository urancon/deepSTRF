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


```{warning}
**AA1 is a subset of AA2.** Every AA1 cell, stimulus and response is also in
[CRCNS AA2](README_CRCNS_AA2.md): the 100 AA1 cells are AA2 cells with the same
names, the 30 AA1 wav files are byte-identical to AA2 wavs, and all 5920 AA1
spike and stimulus-pointer files are byte-identical to the corresponding AA2
files. Neither CRCNS page nor either README says so explicitly. The AA2 README
says AA2 holds "every neuron with a compatible stimulus presentation protocol in
regions MLd, L, and CM which the Theunissen lab has collected as of September
1st, 2007", and AA1 was released in 2008 from the same experimenters and
protocol.

Consequences:

- **Never concatenate AA1 with AA2** (or train one model on both): every AA1
  cell would appear twice, and, unless both use the same split, an AA1 copy of
  an AA2 test response can land in the training set.
- To reproduce AA1 results with AA2's richer labels, load AA2 and select the
  AA1 cells by `cell_id`. AA2 gives AA1's 50 "Field L" cells their sub-area
  (L1: 5, L2a: 16, L2b: 7, L3: 14, L: 8); the 50 MLd cells are `mld` in AA2.
- AA2 has no extra stimuli for these 100 cells: each one has exactly the
  same files in both datasets.
```

## Dataset Details

**Population fitting:** ✅

**Description of Stimuli:**
- 20 clips of conspecific vocalizations and 10 clips of flat ripples (ml-noise), up to 5 s duration.
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


## Benchmark results

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

Easiest path — auto-download via the CRCNS download server:

```python
from deepSTRF.datasets.audio import CRCNSAA1Dataset

ds = CRCNSAA1Dataset(
    download=True, dt_ms=5,
    username="your_username",
    password="your_password",
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
3. `ds = CRCNSAA1Dataset('/path/to/data', dt_ms=5)`.

## Filtering

Each `stim_meta` dict carries `name`, `type` (`"conspecific"` or
`"flatrip"`), `sample_rate`, `n_samples`, `duration_s`. Each
`nrn_meta` dict carries `cell_id`, `area` (`"Field_L"` or
`"MLd"`), `animal_id`, `cell_seq`, `rig`. Combined with the
[base-class selection API](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional):

```python
ds.select_pop_by_nrn_attr("area", "MLd")           # only MLd cells
ds.select_stims_by_attr("type", "conspecific")     # only conspecific stims
                                                   # (auto-hides 2 cells with
                                                   # no conspecific data)
```
