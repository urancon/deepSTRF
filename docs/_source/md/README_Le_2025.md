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
  2.5 ms window, hop = `dt_ms`, `log(P+1)` compression). Matches the
  paper's `gammatone` package recipe (Methods p. 10).
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

Each `stim_meta` dict carries `name`, `motif`, `critical_interval`,
`variant`, `syntax` (`"natural"` / `"scrambled"`), `experiment`,
`sample_rate_hz`, `duration_s`, `ci_onset_s`, `ci_offset_s`.

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
