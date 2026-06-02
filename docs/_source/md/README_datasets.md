# Conventions

One of the main contributions of this library is a growing zoo of off-the-shelf
electrophysiology (and EEG) datasets, compiled from various public sources,
preprocessed, and exposed through a single PyTorch API.

A major obstacle in sensory-response fitting is the sheer variability of
preprocessing methods and formats — different signals (extracellular spikes,
multi-unit activity, intracellular potential, scalp EEG), different labs, and
different experimental setups. deepSTRF hides that behind one common contract so
the same model and training loop work across datasets.

## One base class, one shape

Every dataset subclasses
{class}`~deepSTRF.datasets.neural_dataset.NeuralDataset` (itself a
`torch.utils.data.Dataset`), so it composes with the usual PyTorch utilities
(splitting, shuffling, concatenation, …). Internally each dataset stores, per
stimulus:

* the **stimulus** presented (a spectrogram `(1, F, T)`, or a raw waveform
  `(1, T_audio)` when `return_waveform=True`);
* the **trial-resolved responses**, time-aligned to the stimulus;
* per-stimulus and per-neuron **metadata** dicts (`stim_meta`, `nrn_meta`).

Datasets are sparse and ragged — neurons may not all hear every stimulus, and
stimuli vary in duration and repeat count. deepSTRF handles this with **NaN
sentinels** for missing (stimulus, neuron) trials and zero-/NaN-padding at
collate time. The full contract — the canonical `(B, N, R, T)` response shape,
the NaN-sentinel rules, and the bidirectional neuron/stimulus selection — is
documented in {doc}`data_paradigm`. Read that before touching any
response-path code.

## What a batch looks like

Pair a dataset with `neural_collate` and a `DataLoader`. Each batch is a
**dict** (not a positional tuple), so future per-trial variables can be added as
new keys without breaking your unpacking:

```python
from torch.utils.data import DataLoader
from deepSTRF.utils.data import neural_collate

loader = DataLoader(ds, batch_size=8, collate_fn=neural_collate)

for batch in loader:
    stims      = batch['stims']        # (B, ..., T)        zero-padded, no NaN
    responses  = batch['responses']    # (B, N, R, T)       NaN-padded
    valid_mask = batch['valid_mask']   # (B, N, R, T) bool  ~responses.isnan()
    metas      = batch['stim_meta']    # length-B list of per-stim dicts
    ...
```

Indexing the dataset directly (`ds[i]`) returns the same keys for a single item.
`CCmax` / `TTRC`-style normalisation is **not** precomputed at the dataloader
boundary — it is derived on demand from `responses` by the metrics in
{doc}`metrics_paradigm` (e.g. `normalized_corrcoef`).

## Selecting neurons and stimuli

The set of units (and stimuli) to work with is chosen through the selection API
rather than a fixed constructor argument — by default all neurons are selected.
The selection drives both `len(ds)` and iteration. See {doc}`data_paradigm` for
the exact semantics; the common entry points are:

```python
ds.select_population([0, 1, 2])                          # by index (or a single int)
ds.select_pop_by_nrn_attr("area", "Field_L")            # by metadata label
ds.select_pop_by_nrn_predicate(lambda n: n.get("snr", 0) > 0.5)   # by threshold
ds.select_stims_by_attr("type", "human_speech")         # restrict the stimulus set
```
