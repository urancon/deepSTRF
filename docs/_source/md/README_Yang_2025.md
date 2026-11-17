# Yang 2025 Dataset (budgerigar, decoding)

**Dataset Source:** [Zenodo 14057061](https://doi.org/10.5281/zenodo.14057061)

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Citation:**
```text
Yang, Z.; Long, M. A. (2025). Convergent vocal representations in parrot and
human forebrain motor networks. Nature, 640(8058): 427–434.
https://doi.org/10.1038/s41586-025-08695-8
```

Single units in the left anterior arcopallium (AAC) of four adult male
budgerigars (*Melopsittacus undulatus*), recorded with a 128-channel silicon
probe while the freely moving birds produced warble song and calls. The bird's
own vocalisations come from a **piezo microphone on the skull**, sampled on the
same 30 kHz clock as the spikes; through bone conduction it mostly picks up the
recorded bird's own voice.

AAC is a vocal-**motor** area: its activity leads the sound. This is therefore
a **decoding** dataset —
predict the spectrogram of the bird's own voice from the population activity
(see [Decoding](decoding.md)). It is not suitable for encoding (stimulus →
response) models.


## Dataset Details

**Population fitting:** ✅ within a bird (all units of a bird are simultaneous);
each bird is a separate population.

| `bird` | Single units | Warble bouts | Calls | Vocal time (episodes) |
|---|---|---|---|---|
| `Ti81` | 49 | 45 | 40 | 315 s |
| `Li145` | 81 | 53 | 60 | 81 s |
| `Bl122` | 43 | 22 | 27 | 307 s |
| `Or61` | 47 | 47 | 43 | 272 s |

Units are curated single units (Kilosort 2.5 + manual curation by the authors),
recorded in one session per bird (for Or61, two cut segments of it, joined);
median firing rates are 39–68 Hz across birds.
Only the authors' annotated vocalisations are used (the first part of each
session); later vocal periods are not annotated and are not included.

**Items.** One item per vocal *episode*: a warble bout or a call. Following the
authors, a warble bout is a run of at least 3 vocal elements separated by gaps
under 1 s, and every other element is a call — so calls can come in close pairs.
Episodes range from 15 ms (calls) to 126 s (one Bl122 bout); episodes longer than `max_episode_s`
(default 5 s) are cut into equal pieces. Every item starts `pre_roll_ms`
(default 300 ms) before its vocal onset, so that a causal decoder has neural
history to read; later pieces of a bout take that pre-roll from the end of the
previous piece. The pre-roll frames are part of the item and are scored by the
metrics; to score the vocal span only, mask the first
`round((vocal_onset_s - t_start_s) * 1000 / dt_ms)` frames of each item. This
matters most for calls, where the pre-roll is most of the item.

- `stims[s]` — the decoding **target**: `(1, F, T)` causal log-mel spectrogram of
  the piezo audio (default 32 bands, 500 Hz – 10 kHz, 10 ms window; natural log
  of mel power). On the piezo, the vocal energy sits at 1–8 kHz and is more than
  20 dB above the quiet floor over 0.5–10 kHz. `return_waveform=True` returns the
  raw 30 kHz audio instead.
- `responses[s][n]` — the decoder **input**: `(1, T)` spike counts (one trial,
  R = 1). Neurons of the other birds are missing (NaN) for that item. No motor
  lead is applied to the data: the authors shift activity by 12 ms in their
  analyses; deepSTRF decoders learn their own lags.


## Setup

```python
from deepSTRF.datasets.audio import Yang2025Dataset

# downloads ~3.3 GB of the 15.7 GB Zenodo zip (only the needed files), then
# builds a ~100 MB cache of the vocal episodes
ds = Yang2025Dataset(download=True)

ds.select_pop_by_nrn_attr("bird", "Ti81")     # one bird = one population
ds.select_stims_by_attr("split", "train")
```

If you already have the raw release, pass `raw_path=` (the folder holding
`bgI.mat` and the four `<bird>_ChronicLeftAAC/` folders) instead of
`download=True`. After the cache is built, the raw files (kept in
`default_cache_dir('Yang2025_raw')` by `download=True`) are no longer needed and
can be deleted.


## Metadata

`stim_meta` keys:

| Key | Meaning |
|---|---|
| `bird` | `Ti81`, `Li145`, `Bl122` or `Or61` |
| `episode_type` | `warble` or `call` |
| `bout_id` | episode id, shared by the pieces of one bout (e.g. `Ti81_w012`, `Ti81_c003`) |
| `piece_idx`, `n_pieces` | position of the piece within its episode |
| `t_start_s` | item start on the session clock (s) |
| `vocal_onset_s`, `vocal_offset_s` | vocal span of the piece on the session clock (s) |
| `duration_s` | item length (s), including the pre-roll |
| `split` | `train`, `val` or `test` (see below) |

`nrn_meta` keys: `bird`, `cluster_id` (Kilosort cluster), `shank`,
`max_channel` (channel of the largest template), `coordinates` (`[x, y]` of
that channel in the probe channel map, as released), `fr` (session firing rate, Hz).


## Train / val / test split

`stim_meta['split']` is a fixed split, computed per bird. The units dealt out
are whole bouts, merged with any episode of the same bird less than 1 s away,
so that no item's pre-roll reaches into an episode of another split. Per bird
and unit type (warble / call), units are dealt out in a seeded random order to
test then val, up to about 15 % of the vocal time each, skipping any unit that
would overshoot by more than 25 %; the rest goes to train. Outsized bouts
(Bl122's 126 s bout) therefore always land in train.


## Caveats

- **Bone-conducted audio.** The piezo spectrum is coloured by bone conduction;
  it is a clean target for decoding but is not the free-field sound.
- **Little data per bird.** 81–315 s of vocal time per bird; Li145 vocalised
  for under 2 minutes.
- **Or61 recording is spliced.** Its audio and spikes are two cut segments
  joined at 1703 s; no item crosses the splice.
- **Calls are not typed.** The release marks calls but does not tell contact
  from alarm calls.
