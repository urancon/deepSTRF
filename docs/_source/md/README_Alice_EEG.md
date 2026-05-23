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
data. `AliceEEGDataset` exposes both via the `treat_subjects_as` kwarg.

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

Brodbeck reports **% variability explained** per channel, averaged across
33 subjects. The key thing to internalise about these numbers — and the
[envelope-tracking literature](#what-good-looks-like-on-scalp-eeg)
generally — is the **unit**:

> **% variability explained = 100 · r²**, *not* Pearson `r`.

Figure 4B's "envelope predictive power" topography has a colorbar that
**maxes at 1 %**. That is `r² = 0.01`, i.e. a per-channel Pearson `r ≈
0.10`. The incremental panels (4C/4D — the predictive-power *gain* from
adding onsets / spectrogram over the envelope-only model) span **±0.1 %**
Δ-variance, i.e. another `r ≈ ±0.03`. So the real headline ceiling is:

| Predictor model | % variability explained | equiv. Pearson `r` |
|---|---|---|
| Envelope alone | up to ~1 % | up to ~0.10 |
| + acoustic-onset | + ~0.1 % | + ~0.03 |
| + spectrogram | + ~0.1 % | + ~0.03 |

These are tiny absolute prediction accuracies **by design** — scalp EEG
single-trial envelope tracking is a low-SNR problem (see below). The
science is in the *significance of the increment* and the *shape of the
recovered TRF*, not the absolute predictive power.

### What this library reproduces

We verified empirically (subject S01, single 9/1/2 stim split, Heeris
gammatone-8 spectrogram, 0.5–20 Hz bandpass) that **four independent
estimators converge on the same per-channel test accuracy**:

| Estimator | Mean test `r` | Max | Mean % var |
|---|---|---|---|
| deepSTRF `Linear` STRF (AdamW + MSE) | 0.084 | 0.16 | 0.86 % |
| `sklearn` Ridge (α-grid) | 0.084 | 0.16 | 0.97 % |
| deepSTRF `StateNet` GRU (C=14, 6.8k params) | 0.086 | 0.17 | 0.74 % |
| **eelbrain `boosting`** (Brodbeck's own method) | **0.090** | 0.17 | **1.05 %** |

The `Δ` between the deepSTRF Linear STRF and eelbrain's L1-boosting +
50 ms-basis pipeline is **+0.005 `r` — within run-to-run noise.** There
is **no algorithmic gap and no spec-pipeline gap**: deepSTRF reaches the
published single-subject envelope-TRF ceiling on this dataset. The
recurrent `StateNet` matches the linear STRF with **7× fewer
parameters** — the sample-efficient choice for the small per-subject
data, and the natural backbone for multi-subject pooling.

The accompanying [example notebook](../../examples/alice_eeg_tutorial.ipynb)
walks through a linear baseline and a `StateNet` core on the same data
pipeline. The deepSTRF reframing of the onset-spectrogram condition is
`AdapTrans + Linear` — a learnable peripheral adaptation front-end in
place of the hand-engineered Fishbach-2001 onset detector.

## What "good" looks like on scalp EEG

If you come to this dataset from single-unit or ECoG work, the
per-channel `r ≈ 0.1` ceiling will look like a broken fit. It is not.
Single-trial scalp-EEG envelope-TRF prediction `r` is **0.05–0.15**
across the foundational literature (Lalor & Foxe 2010; Ding & Simon
2012; Di Liberto et al. 2015; Crosse et al. 2016; Broderick et al.
2018; Brodbeck et al. 2018). `r ≥ 0.3` only happens with **intracranial
recordings** (ECoG/sEEG) or **unit-level** data with many trial
repeats — the cortical envelope-tracking response is ~1 µV against
30–50 µV of background, so even `r² = 1 %` is a real, replicable signal.

Because absolute prediction accuracy is low, the EEG/MEG-TRF field
reports its results differently from the spike-prediction `cc_norm` that
deepSTRF's auditory datasets use:

1. **The TRF/STRF kernel shape itself** is the primary deliverable. The
   recovered response function has interpretable, replicable peaks
   (P1/M50 ≈ 50 ms, N1/M100 ≈ 100 ms, P2 ≈ 200 ms); their latencies and
   topographies *are* the science. The kernel is estimated precisely
   even when single-trial prediction is poor. deepSTRF surfaces this via
   [`AudioEncodingModel.STRF_gradmap`](README_gradmap_strf.md).
2. **Predictive power as a significance test, not a score.** With n=33
   subjects, even `r² = 0.5 %` is `p ≪ 0.001` at the group level. The
   question is "does adding predictor X *significantly increase*
   predictive power over the model without it?" — the Δ%-variance you
   see in Brodbeck Fig 4C/4D, tested across subjects.
3. **Nested-model comparison / variance partitioning.** "Does a
   phoneme-surprisal predictor explain variance *beyond* the acoustic
   envelope?" Fit nested models, compare predictive power. This is the
   main thing TRFs are *for* — disentangling overlapping, correlated
   acoustic / lexical / semantic predictors.
4. **Backward (decoding) models + attention decoding.** Reconstructing
   the stimulus envelope *from* EEG pools all channels and reaches
   higher `r` (≈ 0.1–0.3); auditory-attention decoding then reports
   *classification accuracy* (often 80–90 % in 60 s windows), not `r`.
5. **Group-level cluster statistics** over the (time-lag × sensor) TRF
   (TFCE / cluster-permutation corrected) — the result is a significant
   spatiotemporal cluster, reported as a topography + time-course.

For deepSTRF's purposes, the per-channel `r` we measure is the right
*sanity* number; the scientifically useful outputs on EEG are (1) the
recovered TRF kernels and (3) nested-model predictive-power comparisons.

## Setup

**Requirements:** the `[eeg]` optional extra (pulls in MNE-Python):

```bash
pip install "deepSTRF[eeg]"
```

Easiest path — auto-download from the UMd DRUM mirror (~2.5 GiB total,
anonymous HTTPS, idempotent):

```python
from deepSTRF.datasets.audio import AliceEEGDataset

ds = AliceEEGDataset(download=True, dt_ms=10, n_frequency_bands=8)
```

Default cache dir is
`platformdirs.user_cache_dir('deepSTRF')/Alice_EEG`, overridable via
`$DEEPSTRF_DATA_DIR`.

If the data is already laid out manually:

```python
ds = AliceEEGDataset(path="/path/to/brodbeck_eelbrain_elife", dt_ms=10)
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
`sample_rate`, `n_samples`, `duration_s`. Each `nrn_meta` dict
carries `channel_id`, `subject` (or `None` in repeats mode), `area`
(`"EEG"`), and `xyz` (channel position from the standard 10–20 montage,
or `None` if not in the montage). Combined with the
[base-class selection API](data_paradigm.md#8-iteration-honours-the-current-selection-bidirectional):

```python
# default — all subjects, both modes
ds = AliceEEGDataset(download=True)

# only one subject
ds = AliceEEGDataset(download=True, subjects=["S20"])

# inter-subject reliability mode
ds = AliceEEGDataset(download=True, treat_subjects_as="repeats")

# post-construction: select a frontal cluster of channels
ds.select_pop_by_nrn_attr("channel_id", "1")    # one channel by id
```

## Status

The shipped `AliceEEGDataset` + canonical preprocessing (0.5–20 Hz
bandpass, base-class `standardize_stims` + `normalize_responses`)
correctly loads the data and feeds the deepSTRF model API, and **reaches
the published single-subject envelope-TRF ceiling** (see the
[four-estimator comparison above](#what-this-library-reproduces)).
There is no accuracy gap to close against Brodbeck on the acoustic-only
models — `r ≈ 0.09` mean per channel is what the data supports, and
deepSTRF's `Linear` and `StateNet` both get there.

What *would* extend the analysis (in the directions the EEG-TRF field
actually cares about — kernel recovery and nested-model comparison
rather than raw predictive power), in increasing order of effort:

1. **Word-onset / surprisal predictors** from
   `stimuli/AliceChapterOne-EEG.csv`. Reproduces Brodbeck Fig 5–6
   (TRF-of-discrete-events; function vs content words). This is the
   nested-model-comparison story — "does a lexical predictor explain
   variance *beyond* acoustics?" — and is where TRFs earn their keep.
2. **Hamming-basis STRF kernel.** Add a `BasisKernel` to
   `deepSTRF.models.layers` that constrains the temporal axis of the
   STRF to a sparse basis of overlapping Hamming windows (eelbrain's
   `basis=0.050`). It won't move the per-channel `r` much — we're at the
   ceiling — but it produces **smoother, more interpretable TRF
   kernels**, which is the actual deliverable on EEG.
3. **Subject embeddings + shared StateNet backbone.** A learned
   per-subject context vector concatenated to the GRU input — true
   multi-subject pretraining, beyond the per-subject readouts the `N`
   axis already provides. The most promising route to a meaningfully
   *higher* number, by pooling the across-subject shared response.
4. **`eelbrain.boosting` wrapper as an alternative `Fitter`.** The
   apples-to-apples cross-check used to validate this dataset (see
   `untracked/alice_eeg_eelbrain_compare.py`); worth promoting to a
   reusable utility + regression test for future EEG/MEG work.
5. **Topomap helper** using `mne.viz.plot_topomap` from
   `nrn_meta['xyz']` — for the eLife-style scalp figures.
6. **Per-subject `download=True`** instead of all 2.5 GiB at once.

The accompanying [tutorial notebook](../../examples/alice_eeg_tutorial.ipynb)
exercises the dataset end-to-end. It is a **library-on-EEG
demonstration** — showing that the same model API that fits ferret A1
spikes also fits human scalp EEG, lands at the field-standard ceiling,
and recovers interpretable TRF kernels.
