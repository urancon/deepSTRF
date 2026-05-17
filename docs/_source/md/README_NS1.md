# Natural Sound Dataset #1 (NS1)

**Dataset Source:** [NS1 Dataset](https://osf.io/ayw2p/)

**Original Paper:** ["Network receptive field modeling reveals extensive integration and multi-feature selectivity in auditory cortical neurons"](https://doi.org/10.1371/journal.pcbi.1005113) by Nicol S. Harper, Oliver Schoppe, Ben D. B. Willmore, Zhanfeng F. Cui, Jan W. H. Schnupp, Andrew J. King.

**Papers Using the Dataset:**
- ["A dynamic network model of temporal receptive fields in primary auditory cortex"](https://doi.org/10.1371/journal.pcbi.1006618) by M. Rahman, B. D. B. Willmore, A. J. King, N. S. Harper.
- ["Measuring the performance of neural models"](https://doi.org/10.3389/fncom.2016.00010) by O. Schoppe, N. S. Harper, B. D. B. Willmore, A. J. King, J. W. H. Schnupp.

## Dataset Details:

**Population fitting:** ✅

**Description of Stimuli:**
- 20 clips of natural sounds (speech, ferret vocalizations, other animal vocalizations, and environmental sounds), and 12 Dynamic Random Chords (DRCs) each 5 seconds in duration.
- Clips were played in random order
- Natural sound clips were repeated 20 times, DRCs 5 times

**Description of Neurons:**
- Total Number of Neurons: 119
- Valid Neurons: 73
- Valid Neurons Criteria: Noise ratio < 40 (The noise ratio is given)
- Neuron Types: 549 single and multi-unit recordings from six adult pigmented ferrets (five female and one male), among which 284 were single units.

**Available data:**
- *Needs MATLAB Data Processing.* More info in the readme file in the dataset.
- Structured neuron array with 119 items. In each element:
  - Sound waveform of each stimulus (y, fs).
  - Spike times of each repeat of each stimulus.
  - Information on the recording, on the noise ratio, etc.

**Processing needed:**
- Transforming the sound waveform into a 34-band spectrogram.
- Choosing 73 neurons out of 119 based on their noise ratio.
- Transforming the spike times of each repeat of each stimulus into PSTHs.
- What's been done is explained in the source paper.


## Benchmark results

|  **Model backbone**  | **Rank** |                 **Remarks**                  |     **Params / nrn**      |  **Perfs <br/>(CCraw / CCnorm) [%]**  |                                        **Paper (backbone)**                                         | 
|:--------------------:|:--------:|:--------------------------------------------:|:-------------------------:|:-------------------------------------:|:---------------------------------------------------------------------------------------------------:|
|       StateNet       |    🥇    |                   GRU, pop                   |          30,465           |              55.6 / 75.1              |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |          
|     Transformer      |    🥈    |                     pop                      |          29,205           |              53.9 / 73.0              |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |          
|        2D-CNN        |    🥉    |                     pop                      |          36,275           |              51.8 / 70.1              | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |



## Setup

Easiest path — auto-download from OSF (no account required) plus the
pre-computed spectrogram from the DNet GitHub repo:

```python
from deepSTRF.datasets.audio import NS1Dataset

ds = NS1Dataset(download=True, dt_ms=5)
```

Default cache dir is `platformdirs.user_cache_dir('deepSTRF')/NS1`,
overridable via `$DEEPSTRF_DATA_DIR`. `download=True` is idempotent.

If you already have the data laid out manually:

1. Download the response data from [the original OSF repository](https://osf.io/ayw2p/).
2. Download the pre-computed stimulus spectrogram (`test_data_5ms.mat`)
   from the [DNet code repository](https://github.com/monzilur/DNet) of
   [Rahman et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006618).
3. Place `MetadataSHEnCneurons.mat`, the extracted `spikesandwav/`
   folder, and `test_data_5ms.mat` inside a `data/` folder.
4. `ds = NS1Dataset('/path/to/data', dt_ms=5)`.

## Filtering

`stim_meta` carries a `"type"` field with values
`{"water_sounds", "ferret_vocalization", "insects_buzzing", "human_speech", "unknown"}`.
`nrn_meta` carries `cell_id`, `area` (`"A1"`), `depth_um`,
`noise_ratio`, `single_n` / `single_t` (single-unit flags), `n_electrodes`
(total electrodes on the rig at recording time), and `electrode_number`
(1-indexed electrode this cell was recorded on). Combined with the
[base-class selection API](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional):

```python
ds.select_stims_by_attr("type", "human_speech")     # only the 4 speech stims
ds.select_pop_by_nrn_attr("single_t", "Yes")        # only single units
```

NS1's `nrn_masks` is full coverage (every cell saw every stim), so the
bidirectional rule does not prune any cell. It will, however, prune cells
in concatenated datasets that mix NS1 with sparser sources.
