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

**Requirements**: A CRCNS account to download the dataset.

0. Download the data (**crcns-aa1.zip**) at [the original dataset repository](https://crcns.org/data-sets/aa/aa-1/about).
1. Extract the files and place all three **all_stims/**, **Field_L_cells/** and **MLd_cells/** inside the **data/** folder.
2. You can use it right away by creating a `CRCNS_AA1_Dataset(...)` object with the path to the **data/** folder
