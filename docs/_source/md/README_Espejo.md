# Ferret A1 with vocalization-modulated noise & natural sounds (Espejo)

**Dataset Source:** [Zenodo deposit 3445557](https://doi.org/10.5281/zenodo.3445557)

**Original Paper:**
- ["Spectral tuning of adaptation supports coding of sensory context in auditory cortex"](https://doi.org/10.1371/journal.pcbi.1007430) — Lopez Espejo M., Schwartz Z.P., David S.V. (2019), *PLoS Computational Biology* 15(10): e1007430.

## Dataset Details

The deposit packages **two disjoint releases** from awake passively-listening adult ferret primary auditory cortex (A1). Cells and stimuli do not overlap between the two — the dataset class loads one release at a time, and they cannot be concatenated (different `F`).

### NAT — natural sounds

- 540 single-units across 35 recording sites in 6 ferrets.
- 93 different 3-second natural sounds in the canonical bank (vocalizations, speech, environmental, music). Different sites present different subsets — the union across sites in this deposit is ~686 unique stimulus names.
- Each stimulus was presented with 0.5 s pre-stim and 0.5 s post-stim silence — total epoch is 4 s in the file (T=400 bins at fs=100 Hz) or up to 5.5 s for sites with longer windows.
- Stimuli stored as **18-band gammatone log-spectrograms** ("ozgf" NEMS format), F=18.
- A subset (~3 sounds per site) was presented 10–30 times for high-quality PSTH estimation; the rest at 1–5 reps.

### VMN — vocalization-modulated noise

- 200 single-units across 103 recording sites in 5 ferrets.
- 30 different 3-second VMN stimuli per site (28 single/low-rep + 2 high-rep), totalling 75 unique stimulus names across the population.
- Each VMN stim consists of two narrowband noise bands modulated by independent envelopes drawn from natural vocalizations or human-speech recordings.
- Stimuli stored as **2-band envelopes** ("envelope" format), F=2.
- 2 stims per site at 10–30 reps (test); 28 at 2–5 reps (estimation).

Both releases sample at **fs = 100 Hz** → `dt = 10 ms` native.

**deepSTRF parses these archives directly — NEMS0 is no longer required.** Three VMN sites (`chn002h`, `eno023c`, `eno028f`) store responses as `RasterizedSignal` CSVs rather than `PointProcess` HDF5; the native loader handles both transparently.


## Setup

Easiest path — auto-download from Zenodo into the platformdirs cache:

```python
from deepSTRF.datasets.audio import Espejo_Dataset

ds_nat = Espejo_Dataset(stimuli='nat', download=True)   # ~638 MB
ds_vmn = Espejo_Dataset(stimuli='vmn', download=True)   #  ~25 MB
```

Default cache dir is `platformdirs.user_cache_dir('deepSTRF')/Espejo`, overridable via `$DEEPSTRF_DATA_DIR`. To use a custom path explicitly:

```python
ds = Espejo_Dataset('/path/to/your/data/', stimuli='nat', download=True)
```

`download=True` is idempotent — skips both the .tgz download and the untar step if either is already in place.

If you have the data laid out manually:

```
<path>/A1_natural_sounds/NAT/<exptid>_<hash>.tgz
<path>/A1_voc_mod_noise/VMN/<exptid>_<hash>.tgz
```

just pass the path:

```python
ds = Espejo_Dataset('/path/to/your/data/', stimuli='nat')
```

The per-site `.tgz` archives are loaded lazily (untarred to a tempdir per site at construction time); loading VMN takes ~10 s, NAT takes ~2 min on a typical laptop.


## Estimation vs test subsets

Following the paper's `split_by_occurrence_counts` convention, each stim is classified per-site: stims at a site's maximum repetition count are **test**, the rest are **estimation**. Globally a stim is labelled test if any site classified it that way. `stim_meta` surfaces both fields:

| Field        | Example                         | Notes                                                                |
|--------------|---------------------------------|----------------------------------------------------------------------|
| `name`       | `'STIM_00cat172_rec1_geese...'` | Raw NEMS epoch name.                                                  |
| `type`       | `'nat'` or `'vmn'`              | Stimuli release.                                                      |
| `n_repeats`  | `20`                            | Max occurrences across sites.                                         |
| `split`      | `'test'` or `'estimation'`      | Paper-faithful split (test = max-rep within a site).                  |
| `duration_s` | `5.0`                           | Full epoch including pre/post silence.                                |
| `n_samples`  | `500`                           | Number of bins at the dataset's `dt`.                                 |

Filter at load time or iteration time:

```python
ds_test = Espejo_Dataset(stimuli='vmn', subset='test')          # 35 stims
ds_est  = Espejo_Dataset(stimuli='vmn', subset='estimation')    # 40 stims

# or load everything and filter later
ds = Espejo_Dataset(stimuli='vmn')                              # 75 stims
ds.select_stims_by_attr('split', 'test')                        # 35
ds.reset_stim_selection()
```

Because different sites present different stim subsets, the per-stim coverage (the `(S, N)` `nrn_masks` matrix) is sparse — at most ~30 % of `(stim, neuron)` pairs are filled. Cells without responses for a given stim are populated with the canonical `(1, 1)` NaN sentinel; the `subset='test'` filter combined with the [bidirectional rule](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional) on `select_stims_by_attr` automatically hides any cell with no test responses at all.


## Per-cell metadata

`neuron_metadata[n]` carries the raw NEMS `cell_id` plus parsed components:

| Field            | Example          | Notes                                                          |
|------------------|------------------|----------------------------------------------------------------|
| `cell_id`        | `'AMT003c-11-1'` | Raw NEMS id.                                                   |
| `experiment_set` | `'nat'`          | `'nat'` or `'vmn'`.                                            |
| `site`           | `'AMT003c'`      | Recording site (first dash-segment of the id).                 |
| `animal_id`      | `'AMT'`          | Animal code (alphabetic prefix of the site).                   |
| `channel`        | `'11'`           | NAT: digits. VMN: letter + digit (`'c1'`, `'a3'`).             |
| `unit`           | `'1'`            | NAT only (last dash-segment). `None` for VMN's 2-segment ids.  |

Combine with `select_pop_by_nrn_attr`:

```python
ds.select_pop_by_nrn_attr('animal_id', 'AMT')   # cells from animal AMT
ds.select_pop_by_nrn_attr('site', 'AMT003c')    # cells from one experiment
```


## Remarks

- **Multi-session cells.** A handful of cells (13 in VMN, all under `por*` site ids) appear in multiple `.tgz` archives — same `site_id`, different session hashes. The loader concatenates rasters across sessions along the repeat axis, so the user-visible `(R, T)` tensor pools all recordings of that cell.
- **Raw waveforms.** The Zenodo deposit ships pre-computed cochleagrams only; the raw `.wav` stimuli are mirrored at LBHB's [BAPHY repository](https://bitbucket.org/lbhb/baphy/src/master/Config/lbhb/SoundObjects/%40NaturalSounds/). A future revision could expose `stimfmt='waveform'` for finer time resolution — the current loader is fixed at `dt_ms = 10`.
- **No SSA / behavior data.** The Zenodo deposit's two archives cover the passive listening sessions only. Stimulus-specific-adaptation (SSA) and behavioral (tone-detection task) data shown in the paper are not in the public release.
