# Alice EEG Dataset

**Dataset Source:** [The Alice Datasets — Deep Blue Data (UMich)](https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t),
re-released by Brodbeck et al. as a preprocessed MNE-Python deposit at the
University of Maryland DRUM repository
([DOI 10.13016/pulf-lndn](https://doi.org/10.13016/pulf-lndn)).

**Citation:**
```text
Bhattasali, S., Brennan, J. R., Luh, W.-M., Franzluebbers, B., & Hale, J. T.
(2020). The Alice Datasets: fMRI & EEG Observations of Natural Language
Comprehension. Proceedings of the 12th Conference on Language Resources and
Evaluation (LREC), 120–125.
```

**Papers using the dataset:**
- ["Hierarchical structure guides rapid linguistic predictions during naturalistic listening"](https://doi.org/10.1371/journal.pone.0207741)
  by Brennan, Hale & Bolhuis (2019), PLOS ONE.
- ["Eelbrain, a Python toolkit for time-continuous analysis with temporal response functions"](https://doi.org/10.7554/eLife.85012)
  by Brodbeck et al. (2023), eLife. (Uses the same Alice EEG release that
  deepSTRF consumes; the benchmark numbers below are taken from this paper.)


## Dataset Details

**Population fitting:** ✅ (across subjects, across channels, or both)

**Description of Stimuli:**
- First chapter of *Alice in Wonderland* read by a single narrator, split
  into 12 audio segments, totalling ~12.4 minutes (2129 words).
- Mono 44.1 kHz `.wav`, plus a word-onset / n-gram-surprisal CSV table.

**Description of Responses:**
- 33 human participants listened passively to the chapter while EEG was
  recorded.
- 61 EEG channels per subject (10–20-like extended montage).
- `R = 1` per (subject, channel) — each segment was played once per
  subject; no within-subject repeats.

**Available data:**
- Per-subject MNE-Python `.fif` files containing the continuous recording,
  channel montage, segment-onset annotations, bad-channel list, and
  artifact-rejection annotations.

**Processing performed by the dataset class:**
- Audio segments are loaded and converted to log-power ERB-band
  spectrograms (a frequency-domain Gaussian approximation of the gammatone
  spectrogram used in the eelbrain analysis; same band structure as
  Brodbeck Fig 4 panels).
- Per-subject EEG is downsampled to `1000 / dt_ms` Hz (default 100 Hz),
  segmented at the 12 audio-onset annotations, and aligned to the
  spectrogram time grid.
- Bad channels (`raw.info['bads']`) and bad-window annotations (`BAD_*`)
  are converted to NaN at the response level, following the deepSTRF
  [data paradigm](data_paradigm.md) — single source of truth, no separate
  mask.

## Two modes: subjects-as-neurons vs subjects-as-repeats

Alice EEG sits at the intersection of two natural ways to organise the
data. `Alice_EEG_Dataset` exposes both via the `treat_subjects_as` kwarg.

### `treat_subjects_as="neurons"` (default)

Every `(subject, channel)` pair becomes one entry in the neuron axis.
`N = sum_s(channels_s)`, `R = 1`. This is the standard deepSTRF view —
each "neuron" has one trial, and bad-channel-on-subject combos carry the
structural-NaN sentinel. The metrics to report are
[`corrcoef`](metrics_paradigm.md#63-corrcoefpred-gt-masknone-reductionmean)
and
[`fve`](metrics_paradigm.md#scope-a-functional-nan-aware-single-axis-api).

### `treat_subjects_as="repeats"`

Channels-as-neurons, subjects-as-repeats. `N = n_montage_channels` (61),
`R = n_subjects`. Bad `(channel, subject)` cells become NaN at the repeat
slot, and the deepSTRF metrics handle them transparently.

This mode enables **inter-subject reliability** analysis via
[`normalized_corrcoef(method='schoppe')`](metrics_paradigm.md#methodschoppe)
— predictions are scored against an inter-subject signal-power ceiling
analogous to (but **not** the same as) the single-unit trial-reliability
ceiling.

**Interpretive caveat:** the noise model underpinning the Schoppe
correction assumes iid trial noise around a shared deterministic signal.
Between-subject variability is structured (anatomy, source orientation,
attention) and only approximately iid. The math runs and gives a useful
group-level ceiling, but the resulting `CCnorm` is interpreted as
"how well does the model predict the shared, across-subject EEG
component" — not as a trial-reliability bound on a single recording. Use
it as a model-comparison axis, not as an absolute predictive-power
ceiling.

## Benchmark targets (Brodbeck et al. 2023, eLife)

Brodbeck reports **% variability explained** (the deepSTRF
[`fve`](metrics_paradigm.md) metric) per channel, averaged across 33
subjects. Headline numbers from Figure 4:

| Predictor model | Average % variability explained |
|---|---|
| Envelope alone | ~14 % |
| Envelope + acoustic-onset | ~17 % |
| Gammatone spectrogram + onset spectrogram | ~20 % |

The accompanying [example notebook](../../examples/alice_eeg_tutorial.ipynb)
reproduces these numbers with a linear baseline and compares against
DNN cores plugged into the same data pipeline. The deepSTRF reframing of
the onset-spectrogram condition is `AdapTrans + Linear` — a learnable
peripheral adaptation front-end in place of the hand-engineered
Fishbach-2001 onset detector.

## Setup

**Requirements:** the `[eeg]` optional extra (pulls in MNE-Python):

```bash
pip install "deepSTRF[eeg]"
```

Easiest path — auto-download from the UMd DRUM mirror (~2.5 GiB total,
anonymous HTTPS, idempotent):

```python
from deepSTRF.datasets.audio import Alice_EEG_Dataset

ds = Alice_EEG_Dataset(download=True, dt_ms=10, n_frequency_bands=8)
```

Default cache dir is
`platformdirs.user_cache_dir('deepSTRF')/Alice_EEG`, overridable via
`$DEEPSTRF_DATA_DIR`.

If the data is already laid out manually:

```python
ds = Alice_EEG_Dataset(path="/path/to/brodbeck_eelbrain_elife", dt_ms=10)
```

Expected layout under `path`:
```
brodbeck_eelbrain_elife/
├── eeg.0/eeg/Sxx/Sxx_alice-raw.fif
├── eeg.1/eeg/Sxx/Sxx_alice-raw.fif
├── eeg.2/eeg/Sxx/Sxx_alice-raw.fif
└── stimuli/{1..12}.wav  +  AliceChapterOne-EEG.csv
```

## Filtering

Each `stim_meta` dict carries `name`, `type` (`"alice_chapter1"`),
`sample_rate`, `n_samples`, `duration_s`. Each `neuron_metadata` dict
carries `channel_id`, `subject` (or `None` in repeats mode), `area`
(`"EEG"`), and `xyz` (channel position from the standard 10–20 montage,
or `None` if not in the montage). Combined with the
[base-class selection API](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional):

```python
# default — all subjects, both modes
ds = Alice_EEG_Dataset(download=True)

# only one subject
ds = Alice_EEG_Dataset(download=True, subjects=["S20"])

# inter-subject reliability mode
ds = Alice_EEG_Dataset(download=True, treat_subjects_as="repeats")

# post-construction: select a frontal cluster of channels
ds.select_pop_by_nrn_attr("channel_id", "1")    # one channel by id
```

## Status and gap to Brodbeck

The shipped `Alice_EEG_Dataset` + canonical preprocessing (0.5–20 Hz
bandpass, base-class `standardize_stims` + `normalize_responses`)
correctly loads the data and feeds the deepSTRF model API.

On a single subject, single held-out segment, with `StateNet GRU C=14`
(~6.8k params), one obtains test cc ≈ 0.05 mean (~0.12 best channel),
test fve ≈ 0.003. Brodbeck reports test fve ≈ 0.14–0.20 averaged across
33 subjects with 12-fold CV. The **~8× gap** is regularization-bound,
not pipeline-bound:

- Brodbeck fits TRFs with **boosting** — coordinate descent + strict L1
  sparsity + 50 ms Hamming-basis smoothing on the temporal axis. The
  effective parameter count is far below deepSTRF's dense STRF kernel.
- Adam + weight decay alone cannot replicate that prior. Training on
  the full 12 segments shows train cc rising to 0.29 while held-out
  val cc stalls at 0.07 — pure overfitting, not capacity-limited.

### Concrete improvements to close the gap

Listed in increasing order of implementation effort. Each is a
standalone follow-up branch from this one.

1. **Hamming-basis STRF kernel.** Add a `BasisKernel` to
   `deepSTRF.models.layers` that constrains the temporal axis of the
   STRF to a sparse basis of overlapping Hamming windows. Direct port
   of eelbrain's `basis_window=50ms`. Plug into `Linear`/`NRF` via the
   existing `kernel` kwarg. Highest expected impact.
2. **Subject embeddings + shared StateNet backbone.** A learned
   per-subject context vector concatenated to the GRU input. Different
   from naive pooling (which already has per-subject readouts via the
   `N` axis but doesn't condition the shared backbone). Enables true
   multi-subject pretraining.
3. **`eelbrain.boosting` wrapper as an alternative `Fitter`.** Direct
   apples-to-apples comparison with the reference paper. Useful as a
   regression test for any future deepSTRF method on EEG/MEG data.
4. **Word-onset / surprisal predictors** from
   `stimuli/AliceChapterOne-EEG.csv`. Reproduces Brodbeck Fig 5+
   (TRF-of-discrete-events comparisons).
5. **Topomap helper** using `mne.viz.plot_topomap` from
   `neuron_metadata['xyz']` — for the eLife figure.
6. **Per-subject `download=True`** instead of all 2.5 GiB at once.

The accompanying [tutorial notebook](../../examples/alice_eeg_tutorial.ipynb)
exercises the dataset end-to-end and documents the gap explicitly. It
is a **library-on-EEG demonstration**, not a numerical reproduction of
the eelbrain paper; closing the gap is the work of the follow-ups
above.
