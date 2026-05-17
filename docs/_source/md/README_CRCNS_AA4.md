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

Easiest path — auto-download via the CRCNS NERSC mirror:

```python
from deepSTRF.datasets.audio import CRCNSAA4Dataset, AA4_ANIMAL_IDS

ds = CRCNSAA4Dataset(
    download=True, dt_ms=5,
    crcns_username="your_username",
    crcns_password="your_password",
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
corpus), `type` (e.g. `"song"`, `"call"`), `class` (broader category), and
`duration_s`. Each `neuron_metadata` dict carries `cell_id` (the basename
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
