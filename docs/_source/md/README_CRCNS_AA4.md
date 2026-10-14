# CRCNS AA4 Dataset

**Dataset Source:** [AA4 Dataset](https://crcns.org/data-sets/aa/aa-4/about-aa-4)

**Citation:**
```text
 Elie J E and Theunissen F E (2019), Simultaneous extracellular recordings of avian auditory neurons in zebra finches presented with all the repertoire of vocalizations used by this species for vocal communication. CRCNS.org.
http://dx.doi.org/10.6080/K00C4T06
```

**Papers Using the Dataset:**
- ["Meaning in the avian auditory cortex: Neural representation of communication calls"](https://onlinelibrary.wiley.com/doi/10.1111/ejn.12812) (2015) by Julie Elie and Frédéric Theunissen
- ["Invariant neural responses for sensory categories revealed by the time-varying information for communication calls"](https://dx.plos.org/10.1371/journal.pcbi.1006698) (2019) by Julie Elie and Frédéric Theunissen

## Dataset Details

**Population fitting:** ✅

**Batching:** ✅

**Description of Stimuli:**
A total of 170 different clips of conspecific vocalizations (songs and calls) and clips of artificial ripple noise, up to 3 s duration.
- Sample rate @ 24.4 kHz
- Transformed into 32-band mel spectrograms followed by a compression function
- Around 10 response trials on average for a given stimulus

**Description of Neurons:**
- Extracellular single-unit recordings from 4 male and 2 female zebra finches
- anesthetized subjects
- Total Number of units: 1401 (including 914 single units) 
- Targeted avian auditory cortical regions included: 
  - **Field L** (including the thalamo recipient L2, the primary auditory regions L1 and L3), 
  - **caudolateral and caudomedial mesopallium** (CLM and CMM), 
  - **caudomedial nidopallium** (NCM)
- However, neurons were _not_ individually assigned one of these specific regions.

|        **Animal**        | **Sex** | **#units** | **#stims** |
|:------------------------:|:-------:|:----------:|:----------:|
|     **BlaBro09xxF**      |    F    |    151     |    130     |
|     **GreBlu9508M**      |    M    |    355     |    130     |
|     **LblBlu2028M**      |    M    |     53     |    137     |
|     **WhiBlu5396M**      |    M    |    198     |     73     |
|     **WhiWhi4522M**      |    M    |    304     |    131     |
|     **YelBlu6903F**      |    F    |    282     |    129     |

**Available data:**
- Full Python preprocessing.
- One folder for each animal subject, containing several .h5 files of neural recordings (one for each unit)

**Processing needed (Dataset constructor):**
- Transforming the sound waveform (.wav file) into a 32-band spectrogram.
- Choosing neurons based on stimulus type and animal.
- Transforming the spike times of each repeat of each stimulus into PSTHs
  - Remove pre-onset spikes
  - Align trials temporally
  - Pad/cut to the right (present/future time steps) so that trials have the same duration


## Setup

**Requirements**: a [CRCNS account](https://crcns.org/register).

Easiest path — auto-download via the CRCNS download server:

```python
from deepSTRF.datasets.audio import CRCNSAA4Dataset, AA4_ANIMAL_IDS

ds = CRCNSAA4Dataset(
    download=True, dt_ms=5,
    username="your_username",
    password="your_password",
)
```

Alternatively set `$CRCNS_USERNAME` / `$CRCNS_PASSWORD`. Default cache dir
is `platformdirs.user_cache_dir('deepSTRF')/CRCNS_AA4`, overridable via
`$DEEPSTRF_DATA_DIR`. `download=True` is idempotent.

If you already have the data laid out manually, the `data/` folder should
look like this:

```
data/
 |____ BlaBro09xxF/
 |____ GreBlu9508M/
 |____ LblBlu2028M/
 |____ WhiBlu5396M/
 |____ YelBlu6903F/
        |______ ...
```

```python
ds = CRCNSAA4Dataset('/path/to/data', stimuli=('song', 'call'),
                       animals=(AA4_ANIMAL_IDS[0],))
```

## Filtering

Each `stim_meta` dict carries `name` (the stimulus md5 — the canonical
identifier; the wav filename is per-animal and not unique across the
corpus), `type` (e.g. `"song"`, `"call"`), `class` (broader category),
`duration_s`, `md5s` (the md5s of every wav merged into this stimulus, see
below) and `overlap_group` (stimuli that share any audio carry the same id). Each `nrn_meta` dict carries `cell_id` (the basename
of the source h5 file), `animal_id` (one of `AA4_ANIMAL_IDS`), `sex`
(`"M"` or `"F"` — last char of `animal_id`), `site` (recording site
label, e.g. `"Site1"`), `electrode` (int 1-32 across both hemisphere
arrays, 16 channels each in 5/6 birds; 1 array in the 6th),
`ldepth` / `rdepth` (left- and right-array depth in µm at this site),
`sort_type` (`"single"` or `"multi"`; `"noise"` / `"tdt"` are filtered out
at load), `sort_id` (online-sort id, int), and `subsort_id` (offline
spike-sorting id parsed from the trailing `_ss<N>` of the filename;
`None` if absent).

The dataset paper does **not** publish a per-cell brain-area assignment,
so neurons cannot be filtered by area — the natural axis to slice by is
`animal_id`. Otherwise the full selection API from
[the data paradigm doc](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional)
is available: select neurons by metadata (`select_pop_by_nrn_attr`),
select stims by metadata (`select_stims_by_attr`), and the bidirectional
rule auto-hides cells that have no responses to the current stim
selection.

## Duplicate stimuli and the train / val / test split

The md5 of a wav file is not a reliable identity for a sound in AA4, for two
reasons.

- **The same sound saved several times.** The 20 songs and 10 ripple noises
  played to every bird exist in up to three copies: one shared by BlaBro09xxF,
  GreBlu9508M, WhiWhi4522M and YelBlu6903F, LblBlu2028M's copy (about 420
  samples changed at the onset) and WhiBlu5396M's re-quantised copy. Three calls
  exist in two copies. The copies differ about 50 dB below the signal but have
  different md5s, and the original file names differ too.
- **Call bouts that share calls.** A call stimulus is a 2.5-s bout of three
  renditions from one caller and one call type, drawn from a pool of about ten
  renditions; each bird heard different draws. Two bouts that contain the same
  rendition share 0.2–0.6 s of identical audio. This also happens between bouts
  played to the same bird.

The loader handles both with a table computed once from the audio of the whole
corpus:

- **Merging (default).** Copies of one sound become one stimulus
  (`merge_duplicates=True`), so the neurons of all birds share it: 594 wav files
  give 539 stimuli (463 calls, 66 songs, 10 ripple noises). Each bird heard only
  one copy, so no unit gets two trial sets for one stimulus. `name` is the md5 of
  the copy heard by most birds, and `md5s` lists the merged ones. Pass
  `merge_duplicates=False` to keep one stimulus per md5 (the previous
  behaviour).
- **Overlap groups.** `overlap_group` links every stimulus that shares any audio
  with another (a 300-ms log-mel patch matching at r > 0.99). The 539 stimuli
  form 369 groups, of at most 18 stimuli; no group mixes stimulus types or
  callers.

A split that ignores the groups leaks: with a random stratified 70/10/20 split
of the md5s, 78 of 172 test stimuli shared audio with a training stimulus.
`group_split` draws whole groups instead, 70/10/20 within each stimulus type:

```python
from torch.utils.data import DataLoader, Subset
from deepSTRF.datasets.audio import CRCNSAA4Dataset
from deepSTRF.utils import neural_collate

ds = CRCNSAA4Dataset(download=True, dt_ms=5)
split = ds.group_split(seed=0)        # {'train': [...], 'val': [...], 'test': [...]}
train = Subset(ds, ds.stim_to_iter_indices(split["train"]))
val = Subset(ds, ds.stim_to_iter_indices(split["val"]))
test = Subset(ds, ds.stim_to_iter_indices(split["test"]))
loader = DataLoader(train, batch_size=8, shuffle=True, collate_fn=neural_collate)
```

The draw is over the whole corpus, so a given seed puts a sound in the same set
whichever birds or stimulus types are loaded. With seed 0 and all birds, the
split is 377 / 54 / 108 stimuli, and each bird gets 68–73 % / 8–11 % / 17–23 % of
its own stimuli.

