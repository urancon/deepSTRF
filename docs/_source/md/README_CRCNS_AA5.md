# CRCNS AA5 Dataset

**Dataset Source:** [AA5 Dataset](https://crcns.org/data-sets/aa/aa-5)

**Citation:**
```text
Robotka H, Gahr M and Theunissen F E (2022), Simultaneous extracellular recordings of avian auditory neurons in freely behaving zebra finches presented with all the repertoire of vocalizations used by this species for vocal communication. CRCNS.org.
http://dx.doi.org/10.6080/K0TT4P5Q
```

**Paper Using the Dataset:**
- ["Sparse ensemble neural code for a complete vocal repertoire"](https://doi.org/10.1016/j.celrep.2023.112034) (2023) by Robotka H, Thomas L, Yu K, Wood W, Elie J E, Gahr M and Theunissen F E, *Cell Reports* 42(2): 112034.

The call stimuli come from the zebra finch repertoire library described in
["The vocal repertoire of the domesticated zebra finch: a data-driven approach to decipher the information-bearing acoustic features of communication signals"](https://doi.org/10.1007/s10071-015-0933-6)
(Elie & Theunissen, *Animal Cognition* 19(2): 285–315, 2016).

## Dataset Details

**Population fitting:** ✅

**Batching:** ✅

AA5 is the **awake, freely behaving** counterpart of [CRCNS AA4](README_CRCNS_AA4.md):
same lab, same repertoire-wide stimulus design, but the birds were
unrestrained in their home cage (housed in pairs, able to call back) while
chronic 16-channel arrays recorded from their auditory pallium for ~4 weeks.

**Description of Stimuli:**
- **110 calls** covering the 10 ethogram-based call types of the zebra finch
  repertoire, from several vocalizers each: begging (Be, 12), long tonal
  (LT, 13), thuk (Th, 5), distress (Di, 5), aggressive / Wsst (Ag, 9),
  whine (Wh, 13), nest (Ne, 15), tet (Te, 15), distance call (DC, 14) and
  song motifs (So, 9). Recorded at 25 kHz.
- **20 songs**, plus spectrally (`_sfilt`) and temporally (`_tfilt`)
  filtered versions of them (20 each), at 32 kHz.
- **10 random ripples** (`randripple*`, 32 kHz), played at some sites only.
- Each stimulus was played ~10 times per site, in a pseudo-random order,
  with 3–6 s gaps. Playback **levels differ per stimulus on purpose**, to
  mimic natural loudness (Robotka et al. 2023). The level is stored in the
  wav amplitudes, so deepSTRF **never normalises stimuli individually**.
- Every trial is released in a fixed **5 s window, from −0.5 s to +4.5 s
  around stimulus onset**. deepSTRF keeps that whole window: pre- and
  post-stimulus silence included.

**Description of Neurons:**
- Chronic extracellular recordings (16-electrode arrays, spike-sorted) in
  **2 male and 2 female** awake, freely behaving zebra finches.
- Targeted areas: primary (Field L, CLM) and secondary (NCM, CMM) auditory
  pallium. A per-unit area assignment was made for the paper's analysis
  set but is **not** part of the release.
- The release includes all sorted units, not only the paper's single-unit
  candidates (spike SNR > 5). Per-unit `spike_snr` lets you apply that cut.
- Trials recorded during motion artifacts were removed by the authors
  **unit by unit**, so the number of repeats varies per (stim, unit): the
  median is 3, and 5–95% range from 1 to 8.

| **Bird** | **Sex** | **#sites** | **#units with data** |
|:--------:|:-------:|:----------:|:--------------------:|
| **ZF4F** |    F    |     14     |         224          |
| **ZF5M** |    M    |     17     |  *not yet counted*   |
| **ZF6M** |    M    |      9     |         161          |
| **ZF7F** |    F    |     10     |  38 (in 4 of 10 sites) |

A "site" is one recording location (microdrive depth + date), and all its
units were recorded simultaneously. The release has 50 sites in total
(~163 GB).

## Setup

**Requirements**: a [CRCNS account](https://crcns.org/register), and disk
space for at least one archive at a time (up to 7 GB).

### 1. One-time slimming of the raw release

The raw release is one `<site>.tar.gz` per recording site. Each archive
holds one pickle per unit, and every pickle repeats the stimulus waveforms
and the cage-microphone recording of each trial, which is why the release
is so large. deepSTRF reads a compact cache instead (**~90 MB for 27
sites**). `prepare_aa5` builds it by streaming each archive one unit at a
time, with no extraction to disk (about 10 s per GB):

```python
from deepSTRF.datasets.audio import prepare_aa5

# src: folder holding the archives (any depth, any subset of sites) and/or
#      already-extracted <site>/PlaybackPkl/ folders.
prepare_aa5("/path/to/CRCNS_AA5", "/path/to/aa5_cache")
```

`prepare_aa5`:
- works with **any subset** of the 50 sites;
- skips sites that are already cached, so you can re-run it after
  downloading more archives;
- skips truncated downloads;
- can delete each archive once its site is cached (`delete_sources=True`).

The raw pickles were written with pandas 1.x. deepSTRF reads them under
pandas 1.5 as well as recent pandas, so no special environment is needed.

Alternatively, let the dataset do it: `CRCNSAA5Dataset(cache, raw_path=...)`
slims any new site it finds under `raw_path`. `download=True` downloads the
archives from the CRCNS NERSC mirror one at a time, slims each and deletes
it (`$CRCNS_USERNAME` / `$CRCNS_PASSWORD`). The default cache directory is
`platformdirs.user_cache_dir('deepSTRF')/AA5`, and `$DEEPSTRF_DATA_DIR`
overrides it.

### 2. Loading

```python
from deepSTRF.datasets.audio import CRCNSAA5Dataset

ds = CRCNSAA5Dataset("/path/to/aa5_cache")              # everything cached (~4 s)
ds = CRCNSAA5Dataset("/path/to/aa5_cache", animals=["ZF6M"],
                     stimuli=("call",), dt_ms=5)
ds.select_call_type("DC")                               # distance calls only
```

Stimuli and responses share `T = 5000 / dt_ms` bins, and the sound starts
at bin `ds.onset_bin` (0.5 s). The default `dt_ms=5` keeps the full
release at ~0.8 GB of responses (~3.7 GB at 1 ms).

- **Spectrograms**: 32-band mel (`n_mels`), cubic-root compression by
  default. All stimuli are resampled to a common `audio_fs` (25 kHz), so
  calls (25 kHz) and songs / ripples (32 kHz) share one filterbank.
  Frames are **causal**: frame `t` covers the `window_ms` of audio ending
  at the end of bin `t`.
- **Waveforms**: `return_waveform=True` returns `(1, T * hop)` audio at
  `audio_fs`, grid-locked to the response bins.
- **Smoothing**: each trial is smoothed with the usual 21 ms Hanning window
  (`smooth=True`).

## Trial quality: what the loader drops, and why

When slimming the data, deepSTRF compares each playback's cage-microphone
recording with the stimulus file to find where the sound actually was.
The microphone is recorded on the same clock as the spikes. Two kinds of
trials are then **dropped by default**:

- **Misaligned playbacks** (`drop_misaligned=True`): in ~0.7% of all
  playbacks (0.9% of those the microphone can locate confidently), the
  sound was more than 25 ms (`misalign_tol_ms`) away from the logged onset,
  from 0.3 s early to 2.5 s late. The spikes of these trials are aligned to
  the wrong moment. In the one flagged playback recorded on enough
  simultaneous units to test it, the spikes followed the microphone, not
  the logged onset.
- **Overlap with the next playback** (`drop_next_playback_overlap=True`):
  in ~2% of playbacks, the next stimulus starts before +4.5 s, so the end
  of the window contains responses to another sound. The whole trial is
  dropped rather than NaN-ing its tail, because in deepSTRF any NaN inside
  `responses[s][n]` marks the whole (stim, unit) pair as missing.

Everything else is kept. `ds.playback_meta[site]` lists every playback with
its onset-check values (`mic_offset_ms`, `mic_peak`, `mic_second`), the
loudest 20 ms of microphone signal before and after the sound
(`mic_pre_peak_db`, `mic_post_peak_db`, in dB relative to the sound
itself, hence unreliable at quiet-microphone sites), `next_onset_s` and
the resulting `misaligned` / `next_playback_overlap` / `dropped` flags.
`ds.trial_ids[s][n]` gives the site-wide playback index of each response
row, which lets you re-align units recorded simultaneously trial by trial.

## Filtering

Each `stim_meta` dict carries:

| key | meaning |
|---|---|
| `name` | wav file name, the canonical stimulus id |
| `stim_class` | `call`, `song`, `song_sfilt`, `song_tfilt` or `ripple` |
| `call_type` | 2-letter call type (`Ag`, `Be`, `DC`, `Di`, `LT`, `Ne`, `So`, `Te`, `Th`, `Wh`); `So` for songs, `None` for ripples |
| `vocalizer` | ID of the bird that produced the call (e.g. `BlaBla0506`) |
| `vocalizer_code` | raw 3-letter code from the file name (e.g. `MAF`) |
| `vocalizer_sex`, `vocalizer_age` | **inferred** from that code (1st letter M/F, 2nd letter A = adult / C = chick); the code is not documented in the release |
| `rendition` | rendition id of the call |
| `song_id`, `ripple_id` | song / ripple number |
| `native_fs` | original sample rate (25 or 32 kHz) |
| `sound_onset_s`, `sound_offset_s`, `duration_s` | extent of the sound within the window (s, relative to onset) |
| `level_db` | RMS level over the sound (dB re. int16 full scale) |

Each `nrn_meta` dict carries:

| key | meaning |
|---|---|
| `cell_id` | `<site>_e<electrode>-c<cluster>` |
| `animal_id`, `sex` | bird (`ZF4F`, `ZF5M`, `ZF6M`, `ZF7F`) and its sex |
| `site`, `recording_date` | recording site and date (yymmdd) |
| `depth_turns`, `depth_um` | microdrive depth (1 turn = 250 µm) |
| `electrode`, `cluster`, `sort_id` | electrode number and spike-sorting ids, as in the release |
| `spike_snr` | the authors' spike-waveform SNR; the paper uses > 5 for candidate single units |
| `rate_threshold`, `kde_bw` | the authors' motion-artifact exclusion parameters |
| `auditory_z` | mean / SD, over all the unit's trials, of the firing-rate change 0–500 ms after vs. 500–0 ms before onset: the effect size behind the paper's "auditory" test, recomputed here (negative = inhibited by sound) |
| `n_trials` | number of trials loaded for the unit |
| `spike_shape`, `spike_std` | mean spike waveform and its SD (40 samples) |

`spike_snr` measures **spike-sorting quality** and is unrelated to the
response `snr` / `ccmax` that `ds.compute_neuron_quality()` adds to
`nrn_meta`. For example:

```python
ds.compute_neuron_quality()                                   # ~3 min on everything
ds.select_pop_by_nrn_predicate(lambda n: n["spike_snr"] > 5 and n["snr"] > 0.1)
```

Response SNRs are low overall in this dataset (median ≈ 0.02 over the
5 s window, which is mostly silence). `auditory_z` is a fast proxy that
tracks them well.

See the **[aa5_inspection notebook](../ipynb/aa5_inspection.ipynb)** for a
visual tour of the dataset.
