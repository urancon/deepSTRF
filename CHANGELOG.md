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

### Fixed
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
