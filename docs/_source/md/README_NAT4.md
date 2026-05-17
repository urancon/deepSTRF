# A Natural Sound Dataset: A1 & PEG (NAT4)

**Dataset Source:** [NAT4 Dataset](https://doi.org/10.5281/zenodo.8044773)

**Original Papers:**
- ["Can deep learning provide a generalizable model for dynamic sound encoding in auditory cortex?"](https://doi.org/10.1101/2022.06.10.495698) by Jacob R. Pennington, Stephen V. David.
- ["A convolutional neural network provides a generalizable model of natural sound coding by neural populations in auditory cortex"](https://doi.org/10.1371/journal.pcbi.1011110) by Pennington JR, David SV.

## Dataset Details:

**Description of Stimuli:**
- 20 repetitions of 18 sounds + 1 repetition of 577 sounds.
- Each stimulus is 1.5 seconds in duration.

**Description of Neurons:**
- Total Number of Neurons: 849 for A1, 398 for PEG
- Valid Neurons: 777 for A1, 339 for PEG
- Valid Neurons Criteria: Auditory neurons (see the papers for further details)

**Available Data:**
- One population recording per area (`<area>_NAT4_ozgf.fs100.ch18.tgz`)
  packaging the full population time series with the 18 val stimuli
  pre-averaged over 20 reps in the first 27 s, then 575 est stimuli at
  R=1 each. CSV + JSON inside the tarball.
- One per-site recording per recording session
  (`<area>_single_sites/<site>.tgz`) with raw 1 ms-resolution spike
  times stored as HDF5 — used for trial-resolved val responses (R=20).
- Per-area `<area>_pred_correlation.csv` flagging the 777 A1 and 339
  PEG neurons that the dataset's authors classified as auditory.

**deepSTRF parses the NAT4 archive directly — NEMS0 is no longer required.**


## Benchmark results

| **Area** | **Model backbone** | **Rank** |                  **Remarks**                  | **Params / nrn** | **Perfs <br/>(CCraw / CCnorm) [%]** |                                        **Paper (backbone)**                                         | 
|:--------:|:------------------:|:--------:|:---------------------------------------------:|:----------------:|:-----------------------------------:|:---------------------------------------------------------------------------------------------------:|
|  **A1**  |      StateNet      |    🥇    |                   LSTM, pop                   |      40,271      |             46.6 / 65.1             |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
|          |       2D-CNN       |    🥈    | [AdapTrans](docs_md/README_AdapTrans.md), pop |      XX,XXX      |             46.4 / 64.5             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|          |    Transformer     |    🥉    |                      pop                      |      28,437      |             46.6 / 64.4             |                     [Rançon et al.](https://doi.org/10.1101/2025.01.08.631909)                      |
| **PEG**  |    Transformer     |    🥇    |                      pop                      |      28,437      |             39.7 / 55.5             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|          |       2D-CNN       |    🥈    | [AdapTrans](docs_md/README_AdapTrans.md), pop |      XX,XXX      |             39.2 / 55.2             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |
|          |      StateNet      |    🥉    |                   LSTM, pop                   |      40,271      |             38.9 / 54.7             | [Pennington et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011110) |


## Setup

Easiest path — auto-download from Zenodo into the platformdirs cache:

```python
from deepSTRF.datasets.audio import NAT4Dataset

ds_a1  = NAT4Dataset(area='A1',  download=True)   # ~108 MB
ds_peg = NAT4Dataset(area='PEG', download=True)   #  ~50 MB
```

Default cache dir is `platformdirs.user_cache_dir('deepSTRF')/NAT4`,
overridable via `$DEEPSTRF_DATA_DIR`. To use a custom path explicitly:

```python
ds = NAT4Dataset('/path/to/your/data/', area='A1', download=True)
```

`download=True` is idempotent — it skips files / dirs that already
exist, so re-instantiating the dataset is cheap.

If you already have the data laid out manually, just pass the path:

```python
ds = NAT4Dataset('/path/to/your/data/', area='A1')
```

Expected files in the data dir:
* `A1_NAT4_ozgf.fs100.ch18.tgz` (or already-extracted directory)
* `A1_single_sites/` (extracted from `A1_single_sites.zip`)
* `A1_pred_correlation.csv`
* (likewise for PEG)


## Estimation vs validation subsets

NAT4 has two stim subsets: **est** (575 stims, R=1) and **val** (18 stims,
R=20). Each `stim_meta` entry carries a `"subset"` field equal to
`"est"` or `"val"` so you can filter at either load time or iteration time.

```python
# load only one subset (skips the per-site spike-time pass under subset='est')
ds_est = NAT4Dataset(area='A1', subset='est')   # 575 stims
ds_val = NAT4Dataset(area='A1', subset='val')   # 18 stims

# or load everything and filter later
ds = NAT4Dataset(area='A1')                     # 593 stims
ds.select_stims_by_attr('subset', 'val')         # __len__ -> 18
                                                 # 33 val-less A1 cells auto-hidden
                                                 # via the bidirectional rule
ds.reset_stim_selection()                        # back to 593
```

Note: 33 of the 849 A1 cells have no val data. Under `subset='val'` (or
`select_stims_by_attr('subset', 'val')`), the bidirectional rule in the
base class hides them from `__getitem__` automatically, so training
loops only see val-having cells. See
[the data paradigm doc](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional)
for the full contract.

## Per-cell metadata

`neuron_metadata[n]` carries the raw NEMS `cell_id` plus parsed components:

| Field                | Example       | Notes                                                       |
|----------------------|---------------|-------------------------------------------------------------|
| `cell_id`            | `'ARM029a-04-1'` | Raw NEMS id.                                              |
| `area`               | `'A1'`        | Cortical area (A1 or PEG).                                  |
| `auditory`           | `True`        | Per-cell flag from `<area>_pred_correlation.csv`.           |
| `site`               | `'ARM029a'`   | Recording site (animal + recording number + session).       |
| `animal`             | `'ARM'`       | 3-letter animal code (e.g. ARM, CRD, DRX, TNC).             |
| `electrode`          | `4`           | Electrode index, parsed from cell_id.                       |
| `unit_in_electrode`  | `1`           | Unit index on that electrode.                               |

Best-effort: any field whose source is missing or unparseable is `None`
for that neuron. Combine with `select_pop_by_nrn_attr`, e.g.:

```python
ds.select_pop_by_nrn_attr('animal', 'ARM')       # all cells from animal ARM
ds.select_pop_by_nrn_attr('auditory', True)      # the 777 / 339 auditory cells
```
