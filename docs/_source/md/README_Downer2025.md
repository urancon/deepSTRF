# Squirrel-monkey auditory cortex MUA with TIMIT & monkey vocalizations (Downer 2025)

**Dataset Source:** [Zenodo deposit 16175377](https://doi.org/10.5281/zenodo.16175377) — Multi-channel auditory cortex electrophysiology in squirrel monkey (CC BY 4.0, ~29 GB).

**Original Papers:**
- ["Deep neural networks explain spiking activity in auditory cortex"](https://doi.org/10.1371/journal.pcbi.1013334) — Ahmed B., Downer J.D., Malone B.J., Makin J.G. (2025), *PLoS Computational Biology* 21(8): e1013334.
- ["Temporally precise population coding of dynamic sounds by auditory cortex"](https://doi.org/10.1152/jn.00709.2020) — Downer J.D., Bigelow J., Runfeldt M., Malone B.J. (2021), *Journal of Neurophysiology*.

## Dataset Details

Threshold-crossing **multi-unit activity (MUA)** from 1718 recording channels across 41 sessions in three squirrel monkeys (B, C, F), recorded passively while each animal listened to a battery of TIMIT speech sentences and species-specific vocalizations. Spike-sorting was not feasible at acquisition — Ahmed 2025 explicitly treats each channel as a single "multi-unit" (§Data collection, p4).

**Two stimulus classes** are loaded as separate `Downer2025Dataset` instances:

### TIMIT — English speech (`stimuli='timit'`)
- **499 unique sentences** from the TIMIT corpus, 1.99–3.59 s each (median 3.11 s), 16 kHz mono. Each sentence is presented with 0.5 s of pre/post silence padding (`befaft_s = (0.5, 0.5)`); total ~25 min of stimulus per session.
- **Canonical per-session design (Ahmed 2025 p4):** 489 sentences at 1 rep + 10 sentences at 11 reps = 599 presentations per session. Some sessions doubled the 11-rep block (so per-(stim, neuron) `R` can also be 2 or 22).
- **Test split** = the 10 11-rep IDs `[12, 13, 32, 43, 56, 163, 212, 218, 287, 308]` (programmatically discovered from the per-session stim-code histograms, stable across all sessions).
- All sessions presented all 499 sentences → the resulting `(S=499, N=1718)` `nrn_masks` matrix is fully dense (zero NaN sentinels).

### mVocs — monkey vocalizations (`stimuli='mvocs'`)
- **303 unique vocalizations** (grunts, screams, coos) packaged in the 28 min concat WAV `MonkVocs_15Blocks.wav` (41 kHz stereo, mono-averaged and resampled to 16 kHz by the loader). The play-order `mVocsStimCodes` defines the per-voc canonical rep count.
- **Canonical WAV design:** 228 vocs at 1 rep, 75 vocs at variable reps (2–30 reps), and **11 vocs at exactly 15 reps**. The 11 at 15 reps are the paper's test set: IDs `[7, 9, 12, 15, 24, 29, 30, 33, 44, 45, 48]`.
- Per-voc canonical duration = minimum inter-onset interval across that voc's occurrences (excludes variable inter-stim silence). Voc lengths range 0.95–4.69 s, median ~1.8 s.
- Different sessions presented different voc subsets → **~7 % sparse** (37 572 / 520 554 cells are `(1, 1)` NaN sentinels).
- `mvoc` `stim_meta` carries an extra `n_reps_in_wav` key so users can sub-filter beyond the binary test/estimation split.

By default both stim classes go through the same mel-spectrogram pipeline (`audio_fs=16 000`, `n_mels=80`, `fmax=8 000`, `window_ms=25`, `compression='log1p'`, `spec_zscore=True`) so two instances are concat-compatible. These defaults match Ahmed 2025's Kaldi-fbank convention as closely as the deepSTRF mel pipeline allows: 80 mel bands, log compression, per-band per-stim z-score, ~25 ms FFT window. The 8 kHz cap matches Ahmed 2025's STRF baseline cochleagram (`50–8000 Hz`, p8).

## Setup

Easiest path — auto-download from Zenodo into the platformdirs cache (heads up: **the archive is ~29 GB**):

```python
from deepSTRF.datasets.audio import Downer2025Dataset

ds_timit = Downer2025Dataset(stimuli='timit', download=True)
ds_mvocs = Downer2025Dataset(stimuli='mvocs', download=True)
```

Default cache dir is `platformdirs.user_cache_dir('deepSTRF')/Downer2025`, overridable via `$DEEPSTRF_DATA_DIR`.

If you already have the data unpacked:

```
<path>/sessions/<session_id>/<animal>_<session_id>_Ch<N>[suffix]_MUspk.mat
<path>/stimuli/out_sentence_details_timit_all_loudness.mat
<path>/stimuli/SqMoPhys_MVOCStimcodes.mat
<path>/stimuli/MonkVocs_15Blocks.wav
<path>/sessions_metadata.yml
```

just pass the path:

```python
ds = Downer2025Dataset('/path/to/auditory_cortex_data', stimuli='timit')
```

End-to-end load times on a typical workstation: ~8 min for TIMIT (1718×499), ~9 min for mVocs (1718×303).

## Estimation vs test subsets

```python
ds_test = Downer2025Dataset(stimuli='timit', subset='test')          # 10 stims
ds_est  = Downer2025Dataset(stimuli='timit', subset='estimation')    # 489 stims

# or load everything and filter later
ds = Downer2025Dataset(stimuli='timit')                              # 499 stims
ds.select_stims_by_attr('split', 'test')                              # 10
```

`stim_meta[s]` surfaces both fields:

| Field              | TIMIT example                | mVoc example | Notes                                                       |
|--------------------|------------------------------|--------------|-------------------------------------------------------------|
| `name`             | `'fadg0_si1279'`             | `'mvoc_007'` | TIMIT sentence name, or `mvoc_NNN`.                          |
| `type`             | `'timit'`                    | `'mvoc'`     | Stimulus class.                                              |
| `stim_id`          | `1..499` (TIMIT) / `1..303`  | (same)       | MATLAB-indexed; matches the upstream `*Stimcode` arrays.     |
| `duration_s`       | `2.82`                       | `1.95`       | Full waveform duration (including the TIMIT 0.5 s pre/post). |
| `split`            | `'test'` or `'estimation'`   | (same)       | Paper-faithful.                                              |
| `n_reps_canonical` | `1` or `11` (TIMIT)          | `1` or `15`  | The paper's per-stim count.                                  |
| `befaft_s`         | `(0.5, 0.5)` (TIMIT)         | `(0.0, 0.0)` | Silence flanking the stim, only annotated for TIMIT.         |
| `n_reps_in_wav`    | *(absent)*                   | `1..30`      | mVoc only — the per-voc count in the canonical WAV.          |

## Per-cell metadata

`nrn_meta[n]` keys:

| Field                    | Example          | Notes                                                                |
|--------------------------|------------------|----------------------------------------------------------------------|
| `cell_id`                | `'b_180413_Ch49'`| Filename minus `_MUspk.mat`. Unique across the 1718 channels.        |
| `session_id`             | `'180413'`       | Recording date.                                                       |
| `animal_id`              | `'b'`            | One of `'b'`, `'c'`, `'f'`.                                           |
| `hemisphere`             | `'RH'` / `'LH'`  | Monkey C is the only one recorded in both hemispheres.               |
| `area_group`             | `'core'` / `'non-primary'` | High-level area assignment from the YAML.                  |
| `area`                   | `'A1' / 'R' / 'ML' / 'AL' / 'CL' / 'CPB' / 'RPB'` | Fine area (transcribed from YAML comments).      |
| `channel`                | `49`             | Probe channel number.                                                 |
| `channel_suffix`         | `None` / `'p'` / `'s2'` / `'ps2'` | Annotates re-mounted recordings on the same channel #. |
| `n_channels_in_session`  | `16`             | Probe layout for that session (1, 16, 32, 48 or 64 in the paper).    |
| `coord_x`, `coord_y`     | `0.4`, `-0.4`    | 2D craniotomy coordinates (mm).                                       |
| `recording_type`         | `'multi-unit'`   | Constant; all channels are MUA.                                       |

Filter examples:

```python
# Just primary (core) auditory cortex, all animals
ds.select_pop_by_nrn_attr('area_group', 'core')                # 909

# Same, using the Ahmed-2025-style 'primary' alias at construction time:
ds = Downer2025Dataset(stimuli='timit', areas=('primary',))    # 909

# Just A1 (most populous core sub-area)
ds.select_pop_by_nrn_attr('area', 'A1')                         # 813

# Only monkey B's right hemisphere
ds.select_pop_by_nrn_predicate(
    lambda n: n['animal_id'] == 'b' and n['hemisphere'] == 'RH')
```

After 1718 channels are loaded, both **filename-derived metadata** and **YAML-derived metadata** are surfaced; `area_group` is straight from the YAML data, while the fine `area` label is transcribed once from the YAML comments (the comments are not machine-readable from the YAML data block, so the mapping is hard-coded in `downer2025.py`).

## Reproducing Ahmed 2025's analysis cohort

The paper's main figures use a 404-multi-unit "well-tuned to TIMIT" subset and a 489-multi-unit "well-tuned to mVocs" subset, selected by a two-stage criterion: Wilcoxon rank-sum test of trial-to-trial response correlations against a circularly-shifted null, then a δ ≥ 0.5 effect-size filter on the same distributions (§"Trial-to-trial neural variability", pp. 4–5). `Downer2025Dataset` provides an opt-in replication:

```python
ds = Downer2025Dataset(stimuli='timit', smooth=False)
ds.compute_paper_tuning(n_resamples=10_000)
ds.select_pop_by_nrn_predicate(lambda n: n.get('ahmed2025_timit_well_tuned', False))
# → ~413 neurons (paper target: 404)
```

The method writes four keys per neuron — `ahmed2025_<stim>_tuned`, `ahmed2025_<stim>_well_tuned`, `ahmed2025_<stim>_p_wilcoxon`, `ahmed2025_<stim>_delta_normalized`. With `n_resamples=10_000` (~11 min on TIMIT, ~4 min on mVocs) the well-tuned counts come within ~2–3 % of the paper (413 / 475 vs 404 / 489). For an exact match on the looser **tuned** counts (1195 / 1231) bump to `n_resamples=100_000` (~80 min per mode). For most analyses, the library-canonical SNR / CCmax filter via `compute_neuron_quality()` is a faster paper-agnostic alternative.

## Remarks

- **MUA only.** Every `*_MUspk.mat` file holds threshold-crossing spike times for one channel — no spike-sorting, no per-spike cluster ID. Ahmed 2025: *"we refer to the source of spikes on a single channel as a 'multi-unit'"* (p4).
- **Suffix variants** (`p`, `s2`, `ps2`) annotate re-mounted recordings on the same channel number and are kept as distinct cells (with distinct `cell_id`s). Their `trial` struct is identical to the plain-channel files in the same session; only the spike-time history differs.
- **Bandwidth cap.** Both stim classes are resampled to 16 kHz and mel-filtered with `fmax=8 000` to match the paper's 50–8000 Hz cochleagram baseline. The mVocs source actually contains content up to ~20 kHz (squirrel monkey audible range is well above 20 kHz), so users wanting the full band can override `audio_fs=` and `fmax=` — but cross-mode concatenation will then refuse (different `F`/bandwidth between TIMIT and mVocs).
- **8 sessions** lack a top-level `*_TRIALINFO.mat` file. We always read the `trial` struct embedded in each channel's MUspk file instead — they're identical across channels within a session, so this is a no-op for the well-formed sessions and a fallback for the rest.
- **Bad sessions** flagged in `sessions_metadata.yml` are absent from the Zenodo upload (verified at load time). The `Downer2025Dataset` constructor honours the flag defensively anyway.
