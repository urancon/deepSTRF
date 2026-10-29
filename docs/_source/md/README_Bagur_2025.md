# Bagur 2025 Dataset

**Dataset Source:** [Zenodo 13941450](https://doi.org/10.5281/zenodo.13941450)

**Citation:**
```text
Bagur, S.; Bourg, J.; Kempf, A.; Tarpin, T.; Bergaoui, K.; Guo, Y.; Ceballo, S.;
Schwenkgrub, J.; Verdier, A.; Puel, J. L.; Bourien, J.; Bathellier, B. (2025).
A spatial code for temporal information is necessary for efficient sensory learning.
Science Advances, 11(2): eadr6214. https://doi.org/10.1126/sciadv.adr6214
```

Three stages of the mouse auditory pathway — inferior colliculus (IC), auditory
thalamus (TH) and auditory cortex (AC) — recorded in awake, head-fixed mice with
**the same 140 sounds, 13–15 repeats each**, with two-photon calcium imaging and
silicon-probe electrophysiology. The release also holds a cochlear model driven
by the same sounds, which deepSTRF exposes as an alternative **input
representation**. AC units come with their subfield and position on the
cortical surface, and IC single units with their atlas coordinates.


## Dataset Details

**Population fitting:** ✅ (every unit heard every sound)

**Description of Stimuli:** 140 sounds played in random order, mostly 500 ms
long (100–1000 ms). The wav files (192 kHz, 16-bit) carry the level: the 60, 70
and 80 dB copies of a tone differ by 10 dB in the files.

| Category (`stim_meta['category']`) | Sounds | Content |
|---|---|---|
| `blank` | 1 | 1 s of silence |
| `pure_tones` | 33 | 11 tones, 4–37.4 kHz (log-spaced), at 60, 70 and 80 dB |
| `up_ramps`, `down_ramps` | 21 + 21 | 1–5 tones (4–24 kHz) with rising or falling intensity, 70 dB |
| `am` | 12 | sinusoidal AM at 1, 3, 7 and 20 Hz on a 9.8 kHz tone, a 24 kHz tone or white noise, 70 dB, 1 s |
| `chirp_durations` | 12 | up and down sweeps of 100, 250 and 500 ms, 60 dB |
| `chirps` | 40 | 500 ms up and down sweeps over different ranges (4–24 kHz), at 60 and 80 dB |

**Description of Neurons:** the release holds only the units the authors kept
as sound-responsive.

| `areas`, `method` | Recording | Units | Bins | Repeats | Trial window |
|---|---|---|---|---|---|
| `"AC"`, `"2p"` | 2-photon, cell bodies, layers 2–5 | 19,414 | 47 frames, ~31.7 ms | 14 | −0.5 to +1 s |
| `"TH"`, `"2p"` | 2-photon, **thalamic axon boutons** imaged in AC | 3,969 | 47 frames | 15 | −0.5 to +1 s |
| `"IC"`, `"2p"` | 2-photon, dorsal (secondary) IC | 5,936 | 47 frames | 13 | −0.5 to +1 s |
| `"IC"`, `"ephys"` | silicon probe, central IC, single units | 442 | 300 × 5 ms | 14 | −0.5 to +1 s |
| `"TH"`, `"ephys"` | silicon probe, single units | 484 | 300 × 5 ms | 15 | −0.5 to +1 s |

- **Electrophysiology = raw spike counts.** The release subtracts each trial's
  pre-stimulus mean; the loader recovers the spike counts exactly (each trial's
  minimum is minus its baseline) and checks they are integer.
- **2-photon** values are linearly deconvolved fluorescence (an estimate of
  firing-rate changes), Gaussian-smoothed (σ = 31 ms) and baseline-subtracted
  per trial: **signed**, and lagging firing by about 100 ms (GCaMP6s).
- No trial is missing.


## Setup

```python
from deepSTRF.datasets.audio import Bagur2025Dataset

# IC single units at 5 ms, log-spectrogram stimuli (~1 GB in RAM)
ds = Bagur2025Dataset(path="~/data/bagur2025", areas="IC", method="ephys", download=True)

# Thalamus and cortex imaging in one instance (~8.7 GB in RAM)
ds_2p = Bagur2025Dataset(path="~/data/bagur2025", areas=("AC", "TH"), method="2p")
```

`areas` and `method` are required; one instance loads one method (one time
base). AC exists only in imaging.

`download=True` fetches only the files the call needs (md5-checked): the code
and sound archives (250 MB), then 6.7 GB for AC, 1.5 GB for TH and 1.9 GB for IC
imaging, 154 MB / 148 MB for IC / TH electrophysiology, 1.5 GB for the cochlear
model. The first load builds a slim cache in `<path>/_cache` (5.2 GB for
everything, mostly the float16 imaging data; AC takes about a minute); after
that the `Data_*.mat` files and the zip archives can be deleted.

Time bins: electrophysiology defaults to the native 5 ms and accepts `dt_ms` in
5, 10, 20, 25, 50, 100 (so the onset and the 1.5 s window fall on bin edges);
imaging only its frame period. The 31.70 ms frame period is not a whole number
of 192 kHz samples, so `ds.dt` is 6086 / 192 = 31.698 ms (a 0.1 ms drift over a
trial), and frame k is taken as centred on the released frame time.

Smoothing: off by default. `smooth=True` applies a 21 ms Hanning window to
spike counts; it is refused for imaging, which the authors already smoothed.

Memory (float32 responses): AC 7.2 GB, TH imaging 1.6 GB, IC imaging 2.0 GB, IC
spikes 1.0 GB, TH spikes 1.2 GB.


## Stimulus representations

Every stimulus covers the same window as the responses, with the sound at its
true onset (t = 0, checked: the cochlear model leaves its baseline in the first
5 ms bin after onset) and silence elsewhere.

**Log spectrogram (default).** Log-spaced Hamming spectrogram computed directly
at `dt` (`fmin=2000`, `fmax=64000`, `bins_per_octave=6` → 31 bands; `window_ms`
defaults to `2 * dt`), in dB relative to one int16 step, floored at the level of
16-bit quantisation noise. The floor is the same for every sound, so level
differences are kept.

**Waveform.** `return_waveform=True` gives the `(1, T_audio)` trial waveform at
192 kHz (960 samples per 5 ms bin, 6086 per imaging frame), as int16 / 32768.

**Cochlear model.** `spec_backend="cochlear_model"` replaces the spectrogram
with the release's cochlear model (an adaptation of the Meddis model to the
mouse cochlea), shaped `(1, 657, T)`: the mean of its 15 stochastic repeats,
in spikes/s, averaged onto the instance's bins.

- The release's model values are baseline-subtracted per trial (signed), like
  the neural data: spontaneous rates are lost.
- No channel metadata is released. `ds.cochlear_bf_khz` is each fibre's best
  frequency on the 70 dB tones; `ds.cochlear_cf_khz` an **approximate**
  characteristic frequency (5.1–51 kHz): a smooth fit of best frequency on
  fibre index, since fibres 0–~580 are ordered tonotopically while the last
  ~75 lie above the highest tone (37.4 kHz) and respond weakly.
- The file has 657 fibres where the paper describes 590 on 40 hair cells; the
  hair-cell grouping cannot be recovered.


## Metadata

`stim_meta[s]` (in release order):

| Key | Meaning |
|---|---|
| `name` | wav name (`blank` for the silence) |
| `category`, `group` | category above; finer group (`pure_60dB`, `up_chirps_80dB`, ...) |
| `level_db` | 60, 70 or 80 (None for the blank) |
| `frequency_khz` | tone frequencies **measured from the audio** (tones, ramp tone sets, AM carriers); start and end for sweeps |
| `frequency_label_khz` | the rounded frequencies of the names (the "9 kHz" tone is 9.798 kHz) |
| `start_freq_khz`, `stop_freq_khz` | from the release's `SoundInfo.mat` |
| `duration_ms` | from the wav file |
| `direction` | `"up"` / `"down"` for ramps (intensity) and sweeps (frequency) |
| `am_rate_hz`, `carrier` | AM rate and carrier (tone in kHz, or `"white_noise"`) |
| `chirp_duration_ms` | for `chirp_durations` |
| `split_group` | the unit `group_split` keeps together |
| `index_in_release` | the sound's index in the release arrays |

`nrn_meta[n]`:

| Key | Meaning |
|---|---|
| `cell_id` | `"AC-2p_00042"`, stable across selections |
| `area`, `method` | `"AC"` / `"TH"` / `"IC"`, `"2p"` / `"ephys"` |
| `recording_type` | `"2p-calcium"` or `"extracellular-silicon-probe"` |
| `unit_type` | `"soma ROI"`, `"axon bouton"` (TH imaging) or `"single unit"` |
| `index_in_release` | row in the original `Data_*.mat` |
| `subfield` | AC imaging: AI, DP, VPAF, SRAF, AAF (None for 968 unassigned ROIs) |
| `x_um`, `y_um`, `depth_um` | AC imaging: position on the brain surface after aligning the mice on the tonotopic map (intrinsic imaging), and imaging depth (100–600 µm) |
| `subdivision` | IC electrophysiology: IC-C, IC-D, IC-L, `"outside IC"` (22 units), None (91 units without histology) |
| `ap`, `ml`, `dv` | IC electrophysiology: Allen-atlas coordinates as released (AP apparently in mm, ML / DV in µm; None without histology) |
| `best_freq_khz` | *derived by the loader*: the 70 dB tone with the largest evoked response (None if no tone evokes a response above baseline) |
| `spont_rate` | *derived by the loader*: mean pre-stimulus rate in spikes/s (electrophysiology); None for imaging (baseline-subtracted) |

Dataset-level attributes: `areas`, `method`, `species`, `hearing_range_hz`,
`recording_type`, `onset_ms`, `zenodo_record`, `twop_frame_times_s` (imaging),
`indicator` and `response_units` (imaging), `cochlear_bf_khz` and
`cochlear_cf_khz` (cochlear input).

The release has no animal or session identifiers, and no positions for TH (both
methods) or IC imaging.


## Train / val / test split

```python
split = ds.group_split(seed=0)             # {'train', 'val', 'test'}: stimulus indices
from torch.utils.data import Subset
train = Subset(ds, ds.stim_to_iter_indices(split["train"]))
```

Kept together: sounds that are the same waveform up to a gain — the 60 / 70 /
80 dB copies of each tone, the 60 / 80 dB copies of each sweep, and four 500 ms
sweeps that the release ships twice, under a `chirp_durations` name and a
`chirps` name (31 groups, found from the audio).

Everything else is a separate sound, including a ramp or sweep and its time
reversal (the down version can be tested while the up version is trained on).
Sounds are drawn category by category, so every split holds every category;
the blank always goes to train. The draw covers all 140 sounds whatever you
load, so a seed gives the same split for every population. At `seed=0`:
98 / 13 / 29 sounds.


## Caveats

- **The authors' unit selection cannot be reproduced from the released
  traces.** The paper keeps units whose mean inter-trial correlation is at
  least 0.12, and the release ships only those, yet no reading we tried brings
  more than a fraction of them above 0.12 (spikes: up to 54–67% with per-sound
  mean responses; imaging: 7–31%). The selection was probably computed at
  another stage of processing. For model evaluation, gate units by
  `compute_neuron_quality()` and an SNR threshold.
- **Tone frequencies differ from their names.** The names are rounded: the
  11 tones are log-spaced 4–37.4 kHz, the 5 ramp tones 4–24 kHz (6.26, 9.80,
  15.34 kHz for "6, 9, 15"). The authors' tables also give the ramp
  `4+6+9+24kHz` all five tones; its audio has the four of its name.
  `frequency_khz` holds the measured values.
- `tono60dB_4kHz` is 11 dB quieter in its wav than the other 60 dB tones (and
  sign-inverted).
- **The wav level scale differs from Gosselin 2025's** (same lab): a 70 dB tone
  is about 14 dB lower in int16 here, so absolute levels cannot be compared
  across the two datasets from the wav files.
- The authors' own spectrograms (`Code/Utilities/Spectrograms.mat`) start the
  sounds about 10 ms late; the neural data and the cochlear model do not.
