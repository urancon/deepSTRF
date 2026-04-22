# The deepSTRF data paradigm

This note documents how deepSTRF represents stimulus/response data internally,
why it is done that way, and what invariants any user, contributor, or model
author must respect. It is the single reference for the "data contract" of the
library and should stay in sync with the `NeuralDataset` / `AudioNeuralDataset`
/ `VideoNeuralDataset` base classes.

## 1. Problem statement

Sensory neurophysiology datasets are **triply ragged**:

- **Stimulus duration** varies: audio clips of 1–5 s, video clips of different
  lengths. `T_s` is per-stimulus.
- **Repeat count** varies per `(stim, neuron)` pair: different neurons were
  recorded under different numbers of trials. `R_{s,n}` is per-pair.
- **Stim × neuron coverage is sparse**: not every neuron heard every stimulus
  (recording sessions differ, cohorts differ, some bonus datasets concatenate
  disjoint populations).

We need a single storage scheme that handles all three, supports both
single-neuron and population training, and degrades gracefully when users
forget an invariant.

## 2. Storage at the dataset level

A concrete `NeuralDataset` subclass populates six attributes:

| Attribute             | Type                               | Shape / structure                                                                                 |
|-----------------------|------------------------------------|---------------------------------------------------------------------------------------------------|
| `self.stims`          | `list` of length `S`               | Each element is a stimulus tensor of modality-specific shape. Audio: `(1, F, T_s)`. Video: `(1, H, W, T_s)`. `T_s` varies. |
| `self.responses`      | `list[list]` of length `S × N`     | `responses[s][n]` is a float tensor of shape `(R_{s,n}, T_s)` (spike counts per repeat × time). |
| `self.stim_meta`      | `list` of length `S`               | Per-stim metadata **dict**: e.g. `{"name": "...", "type": "...", ...}`. Fields vary per dataset.  |
| `self.neuron_metadata`| `list` of length `N`               | Per-neuron metadata **dict**: e.g. `{"cell_id": "...", "animal_id": "...", "area": "..."}`.       |
| `self.N_neurons`      | `int`                              | Total neurons; equals `len(self.neuron_metadata)`.                                                |
| `self.nrn_masks`      | `(S, N)` bool `torch.Tensor`       | **Derived `@property`.** Computed on the fly from the NaN sentinels in `self.responses` — single source of truth, cannot go out of sync. |

`self.dt` (time-bin width in ms) and `self.path` (data location) are set by
the base class constructor from the `dt_ms` and `path` arguments.

## 3. Encoding missingness

Missingness has three distinct sources. deepSTRF encodes them in a single
channel (NaN in the response tensor) to keep one source of truth, with a
derived boolean mask exposed for ergonomics.

### 3.1 Structural: neuron `n` never saw stim `s`

Convention: `responses[s][n]` is a `(R=1, T=1)` all-NaN tensor.

The `nrn_masks` property on the dataset derives this on the fly — it
sets `nrn_masks[s, n] = False` iff `responses[s][n].isnan().any()`. So
at any time:

```python
dataset.nrn_masks[s, n]    # True iff neuron n has real data for stim s
```

Subclasses must produce the `(1, 1)` NaN sentinel tensor for structurally
missing entries, **not** skip the entry or leave it unset — the mask is
derived from responses, so failing to record the sentinel loses the
missingness information entirely.

### 3.2 Temporal padding: `T_s` varies across stims within a batch

Introduced at **batch time** by the collate function. Stims are zero-padded
on the right; responses are NaN-padded on the right. The batched output
shapes become `(B, 1, F, T_max)` for stims and `(B, N, R_max, T_max)` for
responses.

### 3.3 Repeat padding: `R_{s,n}` varies across neurons within a stim

Also introduced at batch time by the collate function. Responses are NaN-padded
along the repeat dimension up to `R_max`.

## 4. Why NaN-as-sentinel

This is a deliberate design choice, weighed against the explicit-mask
alternative used by HuggingFace / fairseq.

**Chosen properties:**

- **Single source of truth.** The mask is a *derived* view of the data; it
  cannot go out of sync.
- **Loud failure mode.** If downstream code forgets to respect missingness,
  NaN propagates through loss and gradients and crashes training
  immediately. The alternative (multiply-by-mask silently accepts
  zero-contamination on misuse).
- **No dtype overhead.** Spike counts are already float, so NaN fits
  natively.

**Accepted costs:**

- Any **response-side** preprocessing (smoothing, normalization, etc.) must
  be NaN-aware. Use `nanmean` / `nanstd` / `nanmax`, or apply the mask
  before reducing. Specifically: `NeuralDataset.smooth_responses()` and
  `normalize_responses()` (currently unimplemented) must handle NaN when
  they are filled in.
- Float-only; `int` spike-count tensors cannot hold NaN. Responses are
  stored as float throughout.
- NaN vs numerical instability is ambiguous when debugging. When a loss is
  NaN, check the mask path first.

**Invariants this does *not* compromise:**

- Stim tensors are the model's input. They are zero-padded, never NaN-padded.
  This means `BatchNorm`, `LayerNorm`, `softmax`, and all standard model-side
  reductions operate on clean data. The only caveat is the small statistical
  bias from including zero-padded regions in normalization stats — see §8.

## 5. Batching: what the collate produces

Use `neural_collate` from `deepSTRF.utils.data` with a PyTorch `DataLoader`:

```python
from torch.utils.data import DataLoader
from deepSTRF.utils.data import neural_collate

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=neural_collate)
```

One yielded batch is a 4-tuple:

| Name          | Shape                         | Contents                                                                                  |
|---------------|-------------------------------|-------------------------------------------------------------------------------------------|
| `stims`       | `(B, 1, F, T_max)` (audio)    | Float tensor. Zero-padded on the right along `T`. **Never contains NaN.**                 |
| `responses`   | `(B, N, R_max, T_max)`        | Float tensor. NaN-padded on the right along `R` and `T`; full-NaN slab where neuron n didn't hear stim s. |
| `valid_mask`  | `(B, N, R_max, T_max)` `bool` | `~responses.isnan()`. Derived once per batch. Canonical "this position holds real data."  |
| `stim_metas`  | `list` length `B`             | Per-stim metadata dicts, same as stored in `dataset.stim_meta`.                           |

If you need the coarser "did this neuron hear this stim" per batch-item
mask, recover it as `valid_mask.any(dim=(-1, -2))` — a `(B, N)` bool
tensor.

## 6. Recommended training-loop loss pattern

```python
for stims, responses, valid_mask, stim_metas in loader:
    pred_psth = model(stims)                    # (B, N, T_max), trained target is PSTH
    gt_psth   = responses.nanmean(dim=2)        # (B, N, T_max), NaN where all repeats NaN
    valid     = ~gt_psth.isnan()                # (B, N, T_max)
    loss      = F.mse_loss(pred_psth[valid], gt_psth[valid])
    loss.backward()
```

**Why boolean indexing.** `pred_psth[valid]` and `gt_psth[valid]` return flat
1D tensors of the same length. The reduction in the loss is then
automatically over the count of valid positions. Cleaner than multiplicative
masking, and numerically safer (no risk of `0 * NaN = NaN` if NaN survives
anywhere).

**When to use multiplicative masking instead.** If you need per-neuron or
per-stim losses that must preserve the `(B, N, T)` axis structure for
further aggregation, boolean indexing flattens away that structure. In that
case:

```python
per_element = (pred_psth - gt_psth.nan_to_num(0.0)) ** 2 * valid.float()
per_neuron  = per_element.sum(dim=(0, 2)) / valid.float().sum(dim=(0, 2))
```

## 7. Invariants for developers

1. **Stim tensors never contain NaN.** Zero-padding only.
2. **Response tensors may contain NaN.** Always respect `valid_mask` (at the
   loss) or `nrn_masks` (at the dataset) before reducing.
3. **Response-side preprocessing must be NaN-aware.** `smooth_responses`,
   `normalize_responses`, any user-written response transform.
4. **Never feed responses to the model.** They are targets, not inputs.
5. **Models emit predictions for every batched position** — including
   zero-padded stim regions and uncorded neuron-stim pairs. The loss, not
   the model, handles masking.
6. **Subclasses of `NeuralDataset` must call `self.validate()` as the
   last line of `__init__`** (the old `self.compute_nrn_masks()` call is
   no longer needed — `nrn_masks` is a `@property` derived from responses).

## 8. Gotchas

- **Non-causal models on zero-padded stims.** A bidirectional RNN, a
  Transformer without an attention mask, or a CNN with center-weighted
  temporal kernels will see "future silence" where the stim has been
  right-padded. The loss will ignore these positions on the output side
  (since `valid_mask` is False there), but the model's internal
  representations at valid positions can still be affected by attending
  to / convolving over the padded zeros. For causal architectures, this is
  not an issue.
- **Normalization bias from zero-padded stim regions.** `BatchNorm` /
  `LayerNorm` over `T` include zero-padded steps in the stats. Acceptable
  when padding fraction is small (< ~20 %); otherwise use a masked
  normalization or bucket batches by length.
- **GPU synchronization from boolean indexing.** `pred[valid]` has a
  data-dependent output size, which triggers a small GPU sync. Imperceptible
  at deepSTRF scales (`B ≤ 32`, `N ≤ 100`, `T ≤ 5000`); worth knowing if
  scaling up.
- **Spike-count dtype must be float.** Storing responses as `int` would
  prevent NaN encoding.

## 9. When the paradigm might need to evolve

- Migration to `torch.nested` tensors once the ecosystem matures — would
  remove explicit padding, possibly with model-side support gaps.
- Length-bucketed batching for datasets with very heterogeneous `T_s` —
  optimization, not a redesign.
- Per-repeat weighting (some repeats noisier than others) — would require
  either extending the mask from bool to float, or adding a separate weight
  tensor.

These are explicitly *not* on the current roadmap; flagged here so the
discussion doesn't have to be rediscovered.
