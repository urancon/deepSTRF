# Changelog

All notable changes to deepSTRF are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the public API may still change between minor releases.

## [Unreleased]

### Added
- **CRCNS-AA5 dataset** (`CRCNSAA5Dataset`) — awake, freely behaving zebra
  finches (Robotka, Gahr & Theunissen; Robotka et al. 2023, *Cell Reports*)
  hearing their full vocal repertoire (110 calls over 10 call types, songs,
  spectrally / temporally filtered songs, ripples). A one-time streaming
  step (`prepare_aa5`) turns the ~163 GB release into a compact cache and
  works on any subset of its 50 recording sites. Playbacks whose sound the
  cage microphone places away from the logged onset, or that overlap the
  next playback, are dropped by default.
- **Leak-free train / val / test splits** for the datasets whose stimuli share
  audio, so that no test sound is heard in training under another name:
  - `Le2025Dataset.song_split(seed)`: whole songs (all their variants) per
    set, identified across nat8a / nat8b / synth8b (new `stim_meta` keys
    `song_id`, `familiarity_group`).
  - `CRCNSAA4Dataset.group_split(seed, holdout=...)`: keeps shared audio, and
    by default one bird's call type, in one set (`holdout='caller_type'`;
    also `'caller'`, `'call_type'`, `'audio'`). New `stim_meta` keys
    `overlap_group`, `caller`, `call_type`, `identity`, `md5s`.
  - `CRCNSAC1Dataset.group_split(seed)`: Wehr files sharing audio
    (`WEHR_SHARED_AUDIO`, `stim_meta['overlap_group']`) and Asari source
    recordings in one set, consistently across the two subsets.
  - `CRCNSAC1Dataset(asari_level='segment')`: one stimulus per Asari
    fragment, pooling the trials of all its occurrences (`stim_meta` keys
    `segment_file`, `source`, `source_name`, `occurrences`). The default
    `'sequence'` level is unchanged; no leak-free split exists there.
  - `NeuralDataset.stim_to_iter_indices(stim_indices)`: maps the absolute
    stimulus indices these splits return to `Subset` indices.

### Changed
- **Responses are no longer smoothed by default.** NS1, CRCNS-AA1, AA2,
  AA4, AA5, Le 2025 and Downer 2025 now default to `smooth=False`. The
  21-ms Hanning window is centred, so a smoothed bin holds spikes from up to
  ~10 ms later and the target is no longer strictly causal. Pass
  `smooth=True` to smooth (the NS1 pretrained checkpoint was trained with
  smoothing; see the two kernel fixes below).
- **CRCNS-AC1 MGB (cell-attached) responses are spike counts per bin**, one
  spike per threshold crossing with a 1-ms refractory period, unsmoothed.
  They were a Hann-smoothed above-threshold trace in arbitrary units;
  `smooth=True` restores a (unit-area) Hann smoothing. One MGB session whose
  spike detection fails (`20071009-ha1-003`) is now left out by default
  (`ASARI_EXCLUDED_SESSIONS`, `exclude_sessions=`): 13 MGB cells instead
  of 14.
- **CRCNS-AA4 merges duplicate stimulus files** (the same song, ripple or
  call saved under several md5 names) into one stimulus: 594 files become
  539 sounds over the six birds. `merge_duplicates=False` keeps one stimulus
  per file.
- **Response smoothing is now sum-normalised** (`hanning_smooth`,
  `NeuralDataset.smooth_responses`, and every loader's `smooth=True`): the
  Hanning kernel is divided by its sum, so smoothed responses keep their
  units (spike counts per bin). The old kernel was not normalised and scaled
  responses by the kernel sum: x2 at `dt_ms=5`, x10 at `dt_ms=1` (21 ms
  window). Correlation-based metrics (cc, cc_norm) are unchanged; MSE /
  Poisson losses and absolute response values change. The published
  checkpoint `urancon/deepSTRF-statenet-gru-ns1` was trained on the old
  scale (NS1 loads at `dt_ms=5` with `smooth=True`): its predictions are now
  2x the targets; its cc and cc_norm are unchanged.
- **The smoothing window now has the width its name says.** The kernel was
  `np.hanning(K)`, whose two end points are zero, so a "21 ms" window had
  K - 2 non-zero taps: 15 ms at `dt_ms=5`, and `[0, 1, 0]` (no smoothing at
  all) at `dt_ms=10`. It is now MATLAB's `hanning(K)` (`np.hanning(K + 2)`
  without the zero end points), as in the original Theunissen-lab code:
  K non-zero taps, i.e. 21 taps at 1 ms, 25 ms at 5 ms, `[.25, .5, .25]` at
  10 ms. Loaders at `dt_ms=10` with `smooth=True` now smooth. Unlike the
  normalisation, the width changes correlations: the NS1 pretrained
  checkpoint now scores a mean test `cc` of 0.687 instead of the published
  0.659, on the same weights.

### Changed
- **CRCNS-AA2 drops 11 artifact cells by default** (`drop_artifact_cells=True`,
  list in `crcns_aa2.ARTIFACT_CELLS`). Their spike files contain bursts of
  electrical pickup from the playback signal (up to 80 threshold crossings in
  5 ms within one trial, at the stimulus's loudest transients). See the AA2
  page; `drop_artifact_cells=False` restores the archive as released.

### Fixed
- **`ResponseSmoothingTransform`** crashed on use (it read an attribute that
  `__init__` never set). It now delegates to `hanning_smooth` and accepts any
  shape.
- **Noise-corrected metrics on silent responses.** Hsu `CCmax` (used by
  `normalized_corrcoef(method='hsu')`, `compute_CCmax` and
  `compute_neuron_quality`) became NaN for a whole cell as soon as one
  stimulus got a constant (e.g. silent) response, because NaN split-half
  correlations entered the average. Such stimuli are now dropped, and a
  split with one silent half counts as `rho = 0`. Schoppe `cc_norm` no longer
  explodes on nearly silent cells whose signal power is a rounding-level
  positive number: new `sp_rel_threshold` argument (default `1e-6`, relative
  to the mean single-trial variance) returns NaN below it.
- **CRCNS-AA2 cell filters.** With `animals=[...]`, the area labels were
  not filtered along with the cells, so `nrn_meta['area']` was attached to
  the wrong cells (18 of 19 wrong for three birds). The stimulus-class filter
  removed cells while iterating over them and skipped the next one, so it
  kept cells that never heard the requested class (119 of 355 with
  `stimuli=('songrip',)`).
- **CRCNS auto-download** (`download=True` for CRCNS AA1, AA2, AA4, AC1, and
  the new AA5): CRCNS moved its data to AWS in June 2026 and the old NERSC
  portal no longer serves the files. `crcns_download` now logs in to
  crcns.org and downloads from `download.crcns.org`, following the official
  CRCNS client, and resumes interrupted downloads. New helper:
  `crcns_file_list(dataset)`.

## [0.1.0] - 2026-06-03

First public release, available on PyPI: `pip install deepSTRF`.

### Added
- **Datasets** — a zoo of auditory neural-recording datasets (NS1, CRCNS
  AA1/AA2/AA4, NAT4, CRCNS-AC1, Espejo, Downer 2025, Wingert 2026, Le 2025,
  Alice EEG) on a common `NeuralDataset` API with shape `(B, N, R, T)`,
  dict-based batches, `download=True` auto-download where data is publicly
  mirrored, and a filter API (`select_pop_by_*`, `select_stims_by_*`).
- **Waveform input** — optional raw-waveform branch on every audio dataset,
  with a `wav2spec` front-end zoo (CausalMel, SincNet, LEAF, gammatone/-gram).
- **Models** — a four-slot encoding template (wav2spec → prefiltering → core →
  readout): Linear/LN, ConvNet2D, Transformer, StateNet (GRU/Mamba/S4/LMU),
  DNet, NetworkReceptiveField. Strictly causal in eval mode; output rank
  `(B, N, R=1, T)`. Pluggable parametric STRF kernels and parametric
  activations.
- **Metrics** — NaN-aware functional metrics (corrcoef, normalized corrcoef,
  FVE, Sahani–Linden SNR, CCmax, coherence) and Poisson/MSE losses.
- **Training** — an opt-in `Fitter` (early stopping + best-checkpoint
  selection, optional per-cell restoration), multi-seed sweeps
  (`fit_multi_seed`), and optional Weights & Biases / TensorBoard loggers.
- **Pretrained weights** — load checkpoints from the Hugging Face Hub via
  `from_pretrained`; save/push via `save_pretrained` / `push_to_hub`.
- **Packaging** — published to PyPI via GitHub Actions Trusted Publishing
  (OIDC); ships inline type hints (PEP 561 `py.typed`).

[Unreleased]: https://github.com/urancon/deepSTRF/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/urancon/deepSTRF/releases/tag/v0.1.0
