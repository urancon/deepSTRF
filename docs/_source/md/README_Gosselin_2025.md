# Gosselin 2025 Dataset

**Dataset Source:** [Zenodo 17271813](https://doi.org/10.5281/zenodo.17271813) (CC-BY-4.0)

**Citation:**
```text
Gosselin, E.; Bagur, S.; Jamali, S.; Puel, J.-L.; Bourien, J.; Bathellier, B. (2025).
Sound feature representations decorrelate across the mouse auditory pathway.
PLOS Biology, 23(10): e3003452. https://doi.org/10.1371/journal.pbio.3003452
```

Three stages of the mouse auditory pathway — cochlear nucleus (CN), inferior
colliculus (IC) and auditory cortex (AC) — recorded in awake, head-fixed,
passively listening mice with **the same 307 sounds, 12 repeats each**. The
release also contains a biophysical model of the auditory nerve driven by the
same sounds, which deepSTRF exposes as an alternative **input representation**.


## Dataset Details

**Population fitting:** ✅ (every neuron heard every sound)

**Description of Stimuli:** 307 sounds, played in random order, one second
apart. Most last 500 ms (range 8–500 ms). The wav files (192 kHz, 16-bit)
carry the absolute presentation level: the same int16 scale holds for every
file, so 50 dB and 70 dB versions of a sound differ by exactly 20 dB.

| Category (`stim_meta['category']`) | Sounds | Content |
|---|---|---|
| `pure_tones` | 28 | 14 frequencies, 2–60 kHz, at 50 and 70 dB SPL |
| `ramps` | 26 | tones ramped 50→70 dB or 70→50 dB, 2–50 kHz |
| `low_chords`, `medium_chords`, `high_chords`, `broad_chords`, `harmonic_chords` | 48 | sums of 2–4 tones |
| `am` | 48 | sinusoidal AM at 4–160 Hz on tones, chords or white noise |
| `chirps_velocity`, `chirps_frequency` | 20 | up and down frequency sweeps of different speeds and ranges |
| `white_noises`, `narrow_noises`, `broad_noises` | 30 | white noise, level-ramped noise, band-limited noise |
| `complex_sounds` | 60 | 15 natural-like clips (sped-up music, animal calls, environments), forward and time-reversed, at 50 and 70 dB |
| `decompositions` | 47 | synthetic re-creations of 4 of the clips (`dauphin2`, `India`, `bird`, `bird3`) and the components they are built from |

**Description of Neurons:**

| Area | Method | Neurons | Sessions | Native bins | Trial window |
|---|---|---|---|---|---|
| CN | Neuropixels 1.0, Kilosort 2.5 + manual curation | 1421 | 20 | 1 ms spike counts | −200 to +700 ms |
| IC | Neuropixels 1.0, Kilosort 2.5 + manual curation | 551 | 15 | 1 ms spike counts | −200 to +700 ms |
| AC | 2-photon, GCaMP6s, deconvolved | 4217 | 15 | 43.5 ms | −200 to +844 ms |

- The authors kept only units with reliable sound responses (inter-trial
  correlation above 0.05 for spikes, 0.3 for imaging).
- AC traces are linearly deconvolved fluorescence (an estimate of firing-rate
  changes), Gaussian-smoothed (σ = 31 ms) and baseline-subtracted per trial.
  They are **signed**, and they lag firing by about 100 ms (GCaMP6s rise time).
- About 10 repeats are available per AC neuron; missing repeats are dropped,
  so `responses[s][n]` has 9–12 rows.
- CN and IC are raw spike counts (not baseline-subtracted).
- AC was imaged in different mice from CN / IC.


## Setup

```python
from deepSTRF.datasets.audio import Gosselin2025Dataset

# CN + IC spikes at the native 1 ms, log-spectrogram stimuli (the defaults; ~26 GB in RAM)
ds = Gosselin2025Dataset(path="~/data/gosselin2025", download=True)

# IC only, at 5 ms, smoothed with a 21 ms Hanning window
ds = Gosselin2025Dataset(path="~/data/gosselin2025", areas=("IC",), dt_ms=5, smooth=True)

# Auditory cortex (2-photon): its own instance, at the native 43.5 ms bins
ds_ac = Gosselin2025Dataset(path="~/data/gosselin2025", areas=("AC",))
```

`download=True` fetches only the files the call needs (md5-checked): the sound
set and stimulus table (31 MB), then 9.4 GB for CN, 3.7 GB for IC, 3.0 GB for AC
and 1.5 GB for the auditory-nerve model. The first load builds a slim cache in
`<path>/_cache` (~1 GB for everything, ~40 s); after that the original
`Dataset_*.npy` files can be deleted, except `Dataset_CC.npy` if you use
`cochlear_channels="fibres"`.

Time bins: CN / IC default to their native 1 ms and accept `dt_ms` in 1, 2, 4,
5, 10, 20, 25, 50, 100 (so the 200 ms onset and the 900 ms window fall on bin
edges); AC only its native 43.5 ms. CN and IC can share an instance; AC cannot
be mixed with them.

Smoothing: off by default, as for every deepSTRF dataset. `smooth=True` applies
a 21 ms Hanning window to CN / IC; it is refused for AC, which the authors
already smoothed.

Memory: responses are float32. At the default 1 ms, CN takes ~19 GB and IC
~7.3 GB (CN + IC ~26 GB); at `dt_ms=5`, ~3.8 GB and ~1.5 GB. AC takes ~1.2 GB.
On a small machine (or Colab), pass `dt_ms=5` or load IC alone.


## Stimulus representations

Every stimulus covers the same window as the responses, with the sound placed
at its true onset (+200 ms) and silence elsewhere.

**Log spectrogram (default).** Log-spaced Hamming spectrogram computed directly
at `dt_ms` (`fmin=2000`, `fmax=64000`, `bins_per_octave=6` → 31 bands;
`window_ms` defaults to `2 * dt_ms`). Values are dB relative to one int16 step,
floored at the level of 16-bit quantisation noise (18 dB at a 10 ms window).
The floor is the same for every sound, so level differences are kept.

**Waveform.** `return_waveform=True` gives the `(1, T_audio)` trial waveform at
192 kHz (960 samples per 5 ms bin, 8352 per 43.5 ms bin), as int16 / 32768
with no per-sound normalisation. The 50-dB sounds therefore sit around −50 dBFS:
check that your `wav2spec` front end's log floor is well below that, or scale
the input.

**Auditory-nerve model.** `spec_backend="cochlear_model"` replaces the
spectrogram with the firing rates of the release's biophysical auditory-nerve
model (an adaptation of the Meddis model to the mouse cochlea), shaped like a
spectrogram `(1, F, T)` so every model works unchanged.

- `cochlear_channels="ihc"` (default): 51 channels, the mean rate of the fibres
  of each inner hair cell, ordered by characteristic frequency.
- `cochlear_channels="fibres"`: all 691 model fibres.
- `cochlear_transform=None` (default) keeps the linear rates in spikes/s, as
  the authors use them; `"log1p"` compresses them.
- The release has no metadata for the model. The ordering by characteristic
  frequency and the 51 hair-cell groups were recovered from the data (within a
  hair cell, fibres are ordered by increasing spontaneous rate; the paper's
  methods mention 40 hair cells). `ds.cochlear_cf_khz` is an **approximate**
  label: log-spaced over the model's 5–50 kHz range.
- The model output ends at +700 ms. For AC's longer window the remaining bins
  take each channel's spontaneous rate, which is what the model outputs once
  the sound is over.


## Metadata

`stim_meta[s]` (in the release's CSV order, the same for every area):

| Key | Meaning |
|---|---|
| `name` | wav name (one typo of the release CSV corrected) |
| `category` | one of the 15 categories above |
| `intensity_db` | list of levels in dB SPL, as in the release CSV (two for ramps; `[0, 70]` for AM) |
| `frequency_khz` | list of frequencies (tone, chord components, sweep start/end; empty for broadband) |
| `duration_ms` | from the wav file (`duration_ms_csv` keeps the release value, wrong for 6 sounds) |
| `level_db` | the single level, when there is one |
| `reversed` | time-reversed complex clip |
| `direction` | `"up"` / `"down"` for sweeps and level ramps |
| `source_clip` | the clip a complex or decomposition sound belongs to |
| `decomposition_role` | `"original"`, `"reconstructed"` or `"component"` for the 4 decomposed clips |
| `am_rate_hz`, `carrier` | AM rate and carrier (`"tone"`, `"chord"`, `"white_noise"`) |
| `split_group` | the unit `group_split` keeps together |
| `index_cc`, `index_cn_ic`, `index_ac` | the sound's index in each release array |

`nrn_meta[n]`:

| Key | Meaning |
|---|---|
| `cell_id` | `"CN_0042"`, stable across selections |
| `area` | `"CN"`, `"IC"` or `"AC"` |
| `session`, `session_id` | recording session as released, and `"CN-s3"` (unique across areas) |
| `index_in_release` | row in the original `Dataset_*.npy` |
| `best_freq_khz` | *derived by the loader*: the 70-dB pure tone with the largest evoked response (`None` if no tone evokes a response above baseline) |
| `spont_rate` | *derived by the loader*: mean pre-stimulus activity (spikes/s for CN / IC) |

Dataset-level attributes: `species`, `hearing_range_hz`, `recording_type`
(`"extracellular-neuropixels"` or `"2p-calcium"`), `onset_ms`, and for AC
`indicator` and `response_units`.

The release has no animal, location (VCN / DCN, cortical field), ROI position
or imaging-plane information.


## Train / val / test split

```python
split = ds.group_split(seed=0)             # {'train', 'val', 'test'}: stimulus indices
from torch.utils.data import Subset
train = Subset(ds, ds.stim_to_iter_indices(split["train"]))
```

Kept together:
- sounds that are the same waveform up to a gain (the 50 / 70 dB copies of the
  pure tones and of the complex clips, 44 pairs, found from the audio);
- each decomposed clip's reconstruction and all its components, since the
  reconstruction is assembled from their audio.

Everything else is a separate sound: tones, sweeps and ramps at other
frequencies or directions, a clip and its time reversal (a reversed clip can be
in test while its forward version is in training), and an original clip and
its reconstruction (similar, but no shared audio).

Sounds are drawn category by category, so every split holds tones, chords, AM,
sweeps, noises and complex sounds. The draw covers all 307 sounds whatever
`areas` you load, so a seed gives the same split for CN, IC and AC. At
`seed=0`: 230 / 25 / 52 sounds. The 4 decomposed clips form one stratum with a
25-sound unit (the dolphin call), so they are rarely in val or test.


## Caveats

- **Units fade across repeats in CN and IC.** The repeats appear to be in
  recording order and some units drift off the probe: in CN the fraction of
  silent trials rises from 0.43 at the first repeat to 0.62 at the twelfth.
  These zeros are shipped as released; `compute_neuron_quality()` and an SNR
  threshold are the recommended filter.
- **IC units are sparse** (median ~0.1 spikes/s over the trial window).
- **AC is signed and slow**: pair it with `mse_loss`, and expect a ~100 ms lag.
- The release README states a −200 to +800 ms window; the arrays actually hold
  −200 to +700 ms (CN, IC, model) and −200 to +844 ms (AC).
