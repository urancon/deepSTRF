# Le 2025 Dataset

**Dataset Source:** [figshare 29203457](https://doi.org/10.6084/m9.figshare.29203457)

**Companion code (paper analyses):** [melizalab/auditory-restoration](https://github.com/melizalab/auditory-restoration)
(also archived at [Zenodo 15864044](https://doi.org/10.5281/zenodo.15864044)).

**Citation:**
```text
Le, B.; Bjoring, M. C.; Meliza, C. D. (2025). The zebra finch auditory cortex
reconstructs occluded syllables in conspecific song. Nature Communications,
16:8452. https://doi.org/10.1038/s41467-025-63182-y
```


## Dataset Details

**Population fitting:** ✅

**Description of Stimuli:**
- 8 conspecific song motifs ("natural-syntax", `nat8a` / `nat8b` stim sets)
  plus 8 scrambled-syntax pseudo-motifs (`synth8b`); each motif 837–1200 ms,
  high-pass filtered at 500 Hz, RMS-normalized to −27 dBFS (cohorts 1 & 2) or
  −30 dBFS (cohort 3).
- 2 critical intervals (CI1, CI2) per motif, 58–100 ms duration each.
- Up to **7 variants per CI**, probing the auditory restoration illusion:

  | code | name                  | content                                          | cohorts |
  | ---- | --------------------- | ------------------------------------------------ | ------- |
  | C    | Continuous            | Unmodified motif (shared across both CIs)        | all     |
  | G    | Gap                   | CI replaced by silence                           | all     |
  | N    | Noise                 | CI-duration noise burst in isolation             | all     |
  | GB   | Gap + Burst           | CI replaced by noise within the motif (illusion) | all     |
  | CB   | Continuous + Burst    | Motif unchanged, noise added on top of CI        | all     |
  | GM   | Gap-Masked            | Whole motif masked by noise, CI deleted          | nat8b / synth8b |
  | CM   | Continuous-Masked     | Whole motif masked, CI intact                    | nat8b / synth8b |

- Wav files at native sample rate (40 kHz `nat8a`, 48 kHz `nat8b`,
  44.1 kHz `synth8b`).
- 10 presentation repeats per (motif, variant) per neuron; some trials
  curated out → effective `R` ranges 4–10.

**Description of Neurons:**
- Extracellular single-unit recordings from anesthetized adult zebra finch
  auditory pallium (NCM/L3, L2a/b, CM/L1; some sites unlabelled).
- Three cohorts × two stim sets = four (cohort, stim-set) pairings:

  | sub-experiment | response dirs            | cohort(s) | birds | units |
  | -------------- | ------------------------ | --------- | ----- | ----- |
  | `nat8a`        | `nat8a-{alpha,beta}-responses/` | 1 + 2 | 14 + 10 | 407 + 387 |
  | `nat8b`        | `nat8b-responses/`              | 3     | ~8      | 445   |
  | `synth8b`      | `synth8b-responses/`            | 3     | ~5      | 476   |

  *Cohort 1 (`nat8a-alpha`) had a familiarity manipulation*: half the birds
  were socially housed with half of the 8 stimulus singers before recording.
  Cohorts 2 and 3 are entirely unfamiliar (singers had died by then).

**Spike sorting:**
- Cohort 1: MountainSort 4 + `phy` curation.
- Cohorts 2 & 3: Kilosort 2.5 + `phy` curation.
- Spike times stored per-unit as JSON
  [pprox](https://meliza.org/spec:2/pprox/) files.
  *Two pprox schemas coexist*: cohort 1 uses the legacy `spec:2/pprox` (spike
  times in ms, `condition` field encodes the variant); everything else uses
  `spec:2/stimtrial` (spike times in seconds, `stimulus.name` carries the
  full filename stem). The dataset class dispatches on `$schema` and
  presents a uniform interface to callers.

**Processing needed (Dataset class `__init__`):**
- Wav → gammatone spectrogram (50 log-spaced bands, 1–8 kHz,
  2.5 ms window, hop = `dt_ms`, `log(P+1)` compression on the int16 sample
  values, as the authors' code reads the wavs). Matches the paper's `gammatone`
  package recipe (Methods p. 10).
- pprox → `(R, T)` PSTH per (stim, neuron), aligned to stim onset,
  binned at `dt_ms`.
- Optional 21 ms Hanning smoothing
  (Hsu / Borst / Theunissen 2004 convention).
- Optional per-neuron Sahani-Linden signal_power / noise_power / snr,
  length-weighted across stims (∼1 min for `nat8b`; skip with
  `compute_reliability=False`).


## Benchmark results

Quick baseline established from `examples/le_2025_baseline.ipynb`:
`nat8b`, `dt_ms=5`, 50-band gammatone, train on 6 motifs, val/test 1 each,
**Continuous variant only** (8 unoccluded motifs, no occlusion).

|   **Area**    | **Model backbone** | **Rank** | **Remarks** | **Params / nrn** | **Perfs <br/>(CCraw / CCnorm) [%]** |                **Paper (backbone)**                |
|:-------------:|:------------------:|:--------:|:-----------:|:----------------:|:-----------------------------------:|:--------------------------------------------------:|
| **nat8b (all areas)** |  StateNet (GRU) | 🥇 | pop, 6-train | 1,030 | 18.3 / 46.5 (median) | this notebook |

This is a no-tuning baseline — proper LOOCV across all 8 motifs and a
hyperparameter sweep are expected to push CCnorm appreciably higher.
This baseline fits only the 8 `C` stimuli of one experiment, so no audio is
shared across its sets; for fits on all variants and experiments, use
`song_split` (see *Train / val / test split* below).


## Setup

The figshare archive (zipped 105 MB; unpacks to ~720 MB) is auto-downloaded
by the dataset class:

```python
from deepSTRF.datasets.audio import Le2025Dataset

ds = Le2025Dataset(experiment='nat8b', download=True)
```

Default cache dir is
`platformdirs.user_cache_dir('deepSTRF') / 'Le_2025'`, overridable via
`$DEEPSTRF_DATA_DIR`. `download=True` is idempotent — already-unpacked
trees are reused.

Manual install:
1. Download `zebf-auditory-restoration-1.zip` from
   [figshare 29203457](https://doi.org/10.6084/m9.figshare.29203457).
2. Unzip — the archive expands to a `zebf-auditory-restoration-1/`
   directory containing `metadata/`, `*-responses/`, `*-stimuli/`.
3. `ds = Le2025Dataset(path='/path/to/zebf-auditory-restoration-1',
   experiment='nat8b')`.

`gammatone>=1.0` is required for the paper-faithful spectrogram and is
shipped as an optional extra:

```bash
pip install 'deepSTRF[le]'
```


## Filtering

Each `stim_meta` dict carries `name`, `motif`, `song_id`,
`familiarity_group`, `critical_interval`, `variant`, `syntax`
(`"natural"` / `"scrambled"`), `experiment`, `sample_rate_hz`,
`duration_s`, `ci_onset_s`, `ci_offset_s`.

- `song_id` names the song the same way in every experiment: the singer
  for natural songs (`"R56"` for both nat8a `R56_*` and nat8b
  `ep_nat8mk1_*`), the motif name for synth8b. Split on `song_id`, never on
  `motif` (see *Train / val / test split* below).
- `familiarity_group` is `"178B"` or `"180B"`: the group of singers the
  song comes from (the cohort-1 familiarity manipulation housed birds with
  one of the two groups). `None` for synth8b.

Each `nrn_meta` dict carries `cell_id`, `animal_id`, `animal_uuid`,
`cohort`, `experiment`, `area`, `hemisphere`, `familiar_motifs`, `sex`,
`age_days`, `pprox_file`, plus `signal_power` / `noise_power` / `snr` when
`compute_reliability=True` (default).

Combined with the
[base-class selection API](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional)
and the variant-aware helpers:

```python
ds.select_variant('C')                       # only unoccluded motifs (8 stims)
ds.select_motif('nat8mk0')                   # all 12 variants of one motif
ds.select_critical_interval(1)               # CI=1 + the CI-independent C / CM
ds.select_pop_by_nrn_attr('cohort', 1)       # familiarity sub-experiment
ds.select_pop_by_nrn_attr('area', 'NCM/L3')  # one auditory subdivision

# The paper's core restoration analysis runs on a (motif, CI) × {C, CB, GB, GM} quartet:
ds.select_restoration_quartet('nat8mk0', 1)
```

Filter on the precomputed reliability stats to drop weakly responsive cells:

```python
good = [n for n, meta in enumerate(ds.nrn_meta)
        if meta.get('snr', 0) > 0.1]
ds.select_population(good)
```


## Train / val / test split

**Split by song, across experiments.** Most stimuli in this dataset share
audio with other stimuli, so a split by stimulus file puts test audio in
the training set:

- the variants of one song (C, G, GB, CB, GM, CM) are the same recording
  except inside one critical interval (58–100 ms of a ~1 s motif) or under
  an added noise masker;
- nat8a and nat8b contain the same 8 songs under different names (`R56`
  is `nat8mk1`, and so on; `metadata/song-birds.csv`), so a split drawn on
  `motif` separately in each experiment can test on a song that the other
  experiment trains on;
- nat8b uses one frozen white-noise token for every noise burst and masker,
  and synth8b uses another, so the isolated noise bursts (`N`) of different
  songs are the same sound.

`Le2025Dataset.song_split(seed)` handles all three:

- natural songs: **6 train / 1 val / 1 test**; the val and test songs come
  from different familiarity groups;
- synth8b pseudo-motifs: 6 train / 1 val / 1 test (their syllables come
  from other birds, so they share no audio with the natural songs);
- every variant goes to its song's set, except the isolated noise bursts
  (`N`), which always go to train: they contain no song;
- the draw is over song names, so the same `seed` gives the same songs in
  every experiment and in any concatenation of them. Every unit has train,
  val and test stimuli.

```python
from torch.utils.data import Subset
from deepSTRF.utils import concat_neural_datasets

ds = concat_neural_datasets([Le2025Dataset(experiment=e, download=True)
                             for e in ('nat8a', 'nat8b', 'synth8b')])
split = ds.song_split(seed=0)                  # absolute stim indices
train = Subset(ds, ds.stim_to_iter_indices(split['train']))
val   = Subset(ds, ds.stim_to_iter_indices(split['val']))
test  = Subset(ds, ds.stim_to_iter_indices(split['test']))
```

`seed=0` tests on `O129` (nat8a; `nat8mk6` in nat8b) and `synth8mk0`, and
validates on `B2` (`nat8mk5`) and `synth8mk6`. The original paper has no
held-out test set (it decodes with leave-one-motif-out cross-validation),
so this split is deepSTRF's own.

**What the leak was worth.** One GRU population model (844 k parameters)
fit to all 1,715 units of nat8a + nat8b + synth8b, MSE loss, early stopping
on validation cc, best-on-validation weights. Test scores are medians over
units, as mean over 3 runs (min–max in brackets); the leaky row varies the
model seed on one split, the other rows vary the split seed:

| Split | Train stims | raw cc | cc_norm |
| ----- | ----------- | ------ | ------- |
| 6/1/1 by `motif`, per experiment (leaky; model seeds 0–2) | 198 | 0.221 [0.215–0.223] | 0.549 [0.549–0.550] |
| `song_split`, 6/1/1 (split seeds 0, 1, 3) | 210 | 0.186 [0.182–0.191] | **0.470** [0.459–0.486] |
| song-level 4/2/2 (split seeds 0–2) | 156 | 0.174 [0.168–0.186] | 0.405 [0.392–0.422] |

At matched training size, removing the leak lowers test cc_norm by about
0.08. Holding out two songs per set instead of one costs another 0.065,
because the model is short of training songs; that is why `song_split`
holds out one. The cost is a noisier per-experiment score: across the
three 6/1/1 seeds, cc_norm ranges 0.38–0.41 on nat8a, 0.52–0.53 on nat8b
and 0.48–0.58 on synth8b. Report pooled scores, or average several seeds.

Because the occluded variants (G, GB, CB, GM) go with their song, restoration
analyses on a held-out song (e.g. comparing predicted GB and CB responses)
remain possible on the test set.


## Cohort / sub-experiment layout

The handoff between `experiment` and on-disk directories:

```
zebf-auditory-restoration-1/
├── metadata/
│   ├── recordings.csv           area / hemisphere by site (cohorts 2 & 3)
│   ├── song-birds.csv           motif-name mapping (nat8a ↔ nat8b) + CI timings in ms
│   └── ephys-birds.csv          cohort / sex / age / familiarity-group by bird
├── nat8a-alpha-responses/       cohort 1 (familiarity, 407 units, legacy pprox)
├── nat8a-beta-responses/        cohort 2 (unfamiliar, 387 units, stimtrial pprox)
├── nat8a-stimuli/               shared by alpha + beta; bird-named motifs
├── nat8b-responses/             cohort 3 (445 units, natural syntax)
├── nat8b-stimuli/               same motifs as nat8a, renamed nat8mk0..7
├── synth8b-responses/           cohort 3 (476 units, scrambled syntax)
└── synth8b-stimuli/             synth8mk0..7
```

`experiment='nat8a'` unifies the alpha and beta response dirs (they share
the same stimulus set); the per-unit `cohort` field distinguishes them.
Cross-experiment concatenation works via the standard
[`concat_neural_datasets`](data_paradigm.md#concatenation):

```python
nat = Le2025Dataset('...', experiment='nat8a')
syn = Le2025Dataset('...', experiment='synth8b')
both = nat + syn   # block-diagonal coverage matrix; bidirectional rule applies
```
