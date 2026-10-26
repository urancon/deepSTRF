# Changelog

All notable changes to deepSTRF are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the public API may still change between minor releases.

## [Unreleased]

### Changed
- **DNet output unit is stateful again**, as in Rahman et al. (2019): the
  output activation (default `nn.Sigmoid`) is followed by a learnable
  exponential decay per output neuron (`output_decay=True`). The default STRF
  window is now 5 frames (`temporal_window_size=5`, 25 ms at 5-ms bins), since
  hidden and output units carry their own memory.
- **NRF and DNet normalise their hidden units with `nn.BatchNorm2d`** after the
  STRF layer, instead of a per-timestep `CausalLayerNorm`, consistent with the
  per-neuron BatchNorm of Linear / LinearNonlinear. In eval mode BatchNorm uses
  frozen statistics, so the models stay strictly causal. Checkpoints saved
  with the previous NRF / DNet layout do not load into the new classes.

### Changed
- **`ParametricDoubleExponential` threshold `s` now starts at 0** (was
  `U(0.5, 1.5)`), the mean of a normalised drive, so inputs start on the rising
  part of the curve instead of its flat floor. On sparse data the old init left
  many neurons stuck in the floor with no gradient (NAT4 StateNet: 282 of 1083
  cells; cc_norm 0.644 → 0.680 with the new init).

### Fixed
- **NAT4 est and val responses are now in the same units** (spike counts per
  10-ms bin). The est responses used to be divided by the area's maximum count
  (7 in A1, 8 in PEG) while the val responses stayed raw counts, so a model
  trained on est predicted the val responses about 7x too low. Per-cell cc and
  cc_norm were unaffected; MSE and any absolute response level were not.
- **`ParametricDoubleExponential` no longer produces NaN gradients** for very
  negative drive: the inner exponential overflowed (float32) and the backward
  pass computed inf * 0. The exponent is now clamped where the output is
  already exactly zero, so forward values are unchanged.
- **`mse_loss` / `poisson_loss` are NaN-loud**: a neuron that has data but a NaN
  prediction now makes the scalar loss NaN instead of being silently dropped by
  `nanmean`. Neurons with no valid data are still skipped.

### Added
- **CRCNS-AA5 dataset** (`CRCNSAA5Dataset`) — awake, freely behaving zebra
  finches (Robotka, Gahr & Theunissen; Robotka et al. 2023, *Cell Reports*)
  hearing their full vocal repertoire (110 calls over 10 call types, songs,
  spectrally / temporally filtered songs, ripples). A one-time streaming
  step (`prepare_aa5`) turns the ~163 GB release into a compact cache and
  works on any subset of its 50 recording sites. Playbacks whose sound the
  cage microphone places away from the logged onset, or that overlap the
  next playback, are dropped by default.
- **AA5 unit selection**: `nrn_meta['reliability']` (split-half reliability
  of the responses during the sounds) and `CRCNSAA5Dataset.select_good_units()`
  (`spike_snr > 5` and `reliability > 0.1` by default: 290 of the 838 units).
  `reliability` is computed on 21-ms smoothed responses whatever `smooth` is.
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
- **`SincNet` peak-normalises its band-pass filters** (`normalize=True`, new
  default), as the reference SincNet (Ravanelli & Bengio 2018) and ICNet
  (Drakopoulos et al. 2025) code do. Previously the passband gain was ~1 and
  the peak tap 2(f2 - f1)/fs, so narrow low-frequency bands came out hundreds
  of times weaker than in the references. `normalize=False` restores the old
  filters.
- **`ICNet` defaults: normalisation layers and kernels sized in ms.** Every
  encoder / bottleneck conv is now followed by a `CausalLayerNorm`
  (`norm='layer'`), and the kernel lengths follow from a target receptive
  field (`receptive_field_ms=250`, new helper `icnet_kernel_sizes`) so the
  encoder covers the same duration at any sample rate and stride
  factorisation. With fixed 64-tap kernels it spanned from tens of ms to over
  a second across datasets, and without normalisation training was unstable on
  datasets recorded at unusual waveform levels. The decoder bias now starts the
  prediction at `init_rate` (0.05; pass the mean training response) instead of
  softplus(0) = 0.69. `ICNet(..., norm=None, receptive_field_ms=None,
  init_rate=None)` gives the paper's layer structure and init.
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
- **CRCNS-AA4 silent trials.** The loader dropped every trial without a
  spike after stimulus onset (8.5% of trials; in the h5 files often a lone
  `-999` sentinel) and turned a unit that heard a sound but never fired into
  a NaN "not heard" slot. Silent trials and silent responses are now kept as
  zeros: repeats per unit and stimulus are back to the recorded 9-12 (down
  to 1 before), and PSTHs, SNR and CCmax lose their upward bias.
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
