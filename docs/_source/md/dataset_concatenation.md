# Dataset concatenation

> Pool recordings across species, labs, and preparations into a single
> deepSTRF dataset, and fit one model on the union.

deepSTRF supports concatenating two or more `NeuralDataset` instances along
**both** the stim and neuron axes. The result is a "chimeric" dataset that
keeps each source's data intact and fills the cross-blocks with the
canonical NaN sentinel — the same one used everywhere else in the library
to mark missing `(stim, neuron)` pairs.

This is, to our knowledge, a feature unique to deepSTRF among
neurophysiology DNN benchmark libraries. We hope it makes a class of
experiments easier than they have been: training a single model on data
from different species, different labs, or different recording sessions,
to ask which computational principles generalise.

## What concatenation actually means here

Concatenating datasets in deepSTRF is **not** appending stimuli into one
long playlist (that would be `torch.cat`-style), and **not** adding more
neurons that observed the same stimuli. It is the more general 2-D case:

| Example                | Stim axis  | Neuron axis | Cross-block        |
|------------------------|------------|-------------|--------------------|
| `aa1 + aa2`            | A's ⊕ B's  | A's ⊕ B's   | NaN sentinels      |
| `aa1 + aa2 + ns1`      | three-way  | three-way   | NaN sentinels      |

For `k` input datasets with `(S_i, N_i)` each, the result has
`S = Σ S_i` stimuli and `N = Σ N_i` neurons arranged block-diagonally:

```
              neurons of A │ neurons of B │ neurons of C
stims of A │   real data   │      NaN     │      NaN
stims of B │      NaN      │   real data  │      NaN
stims of C │      NaN      │      NaN     │   real data
```

`combined.nrn_masks` reflects exactly that pattern, since the property is
derived on the fly from `combined.responses`.

## Why this is useful

- **More data, same model.** Auditory neurons in zebra finch (CRCNS AA1)
  and in ferret (NS1) share computational principles even though the
  experimental preparations differ. Concatenating the datasets and fitting
  a single backbone lets the model exploit that overlap, while output
  heads remain neuron-specific.
- **Same species, more recordings.** The CRCNS AA series (AA1 / AA2 /
  AA4 / AA5) are all zebra finch but from different cohorts; concatenation
  produces a substantially larger pooled dataset than any of them alone.
- **Cross-lab benchmarks.** Train once, evaluate the same model on
  per-source held-out sets — a cleaner benchmark than fitting four models
  separately and averaging.
- **Lossless missing-data accounting.** The block-diagonal mask is
  built into the loss path automatically (via `valid_mask` from
  `neural_collate`), so the model never sees a fake "this neuron heard
  this stim" signal.

## Public API

```python
from deepSTRF.utils.data import concat_neural_datasets

# N-ary form — recommended for clarity
combined = concat_neural_datasets([aa1, aa2, ns1])

# pairwise sugar via __add__ — convenient for ad-hoc work
combined = aa1 + aa2
```

Both produce identical results. The N-ary form is preferred because it
reads more clearly when more than two datasets are involved and avoids
relying on the `__add__` operator chaining.

## Compatibility requirements

deepSTRF will refuse to concatenate datasets that disagree on properties
that would make the result silently wrong. These are **hard asserts**, not
auto-adapted:

| Requirement              | Where checked                                  | Caller must…                    |
|--------------------------|------------------------------------------------|----------------------------------|
| Same `dt_ms`             | `NeuralDataset._concat_check_compat`           | re-instantiate with matching bin |
| Same `F` (audio)         | `AudioNeuralDataset._concat_check_compat`      | re-instantiate with matching `n_mels` |
| Same `(H, W)` (video)    | `VideoNeuralDataset._concat_check_compat` (TBD)| re-instantiate / pre-resample    |
| All inputs are datasets  | `concat_neural_datasets`                       | only pass `NeuralDataset`s       |

Resampling responses to a common `dt_ms` or warping spectrograms to a
common `F` is intentionally **not** done by deepSTRF — different choices
have different scientific implications, and we would rather the user
makes them explicitly than have the library hide them.

## Return type

The result's concrete type is the **most-specific common ancestor** of
the inputs:

| Inputs                                       | Result type                  |
|----------------------------------------------|------------------------------|
| `aa1, aa1` (or two of any single subclass)   | `CRCNSAA1Dataset` (preserved) |
| `aa1, aa2` (different audio subclasses)      | `AudioNeuralDataset`         |
| `aa1, video_ds`                              | `NeuralDataset`              |

Subclass-specific methods that don't make sense on the merged object
(e.g. `aa1.areas`) are simply not present on the result. The core API —
`stims`, `responses`, `stim_meta`, `nrn_meta`, `nrn_masks`,
`select_*`, `__len__`, `__getitem__` — works identically.

## Iterating only one source's data

`__len__` and `__getitem__` filter by the current neuron selection — they
expose only stimuli for which at least one currently selected neuron has
valid response data (see [`data_paradigm.md`](data_paradigm.md) §8 for
the general rule). This makes selecting a sub-population on a chimeric
dataset Just Work: the cross-block stims, which are full-NaN against the
selected neurons, are hidden automatically.

```python
combined = aa1 + aa2     # 30 + 117 = 147 stims, 100 + 494 = 594 neurons
len(combined)            # 147 — full pool by default

# select only AA1's neurons -> AA2's stims disappear from iteration
combined.select_population(list(range(aa1.N_neurons)))
len(combined)            # 30
combined[0]              # an AA1 stim, with valid responses for the selection
combined[30]             # IndexError, not a fully-NaN AA2 stim

# select MLd neurons across both sources (AA1 has 'MLd', AA2 has 'mld')
mld = [n for n, m in enumerate(combined.nrn_meta)
       if m["area"].lower() == "mld"]
combined.select_population(mld)
len(combined)            # all stims that any MLd neuron heard, in either source
```

A `DataLoader` over the chimeric dataset under any of these selections
visits only the relevant stims — no manual filtering, no risk of training
on a fully-NaN batch item.

The same idea works in the other direction. Filtering on a stim attribute
(`select_stims_by_attr`, `select_stim`, `select_stims`) narrows iteration
to the matching stims, and the bidirectional rule auto-hides cells that
have no responses left in the selection:

```python
# train only on conspecific stims across the chimeric pool
combined = aa1 + aa2
combined.select_stims_by_attr("type", "conspecific")
# - cells that have responses to >=1 conspecific stim survive
# - cells that only have responses to flatrip / non-conspecific stims are hidden
```

## Caller invariants we don't enforce

- **Neuron and stim UIDs should be mutually exclusive across sources.**
  Pooling a dataset with its own subset is degenerate (use constructor
  arguments instead). Duplicate UIDs across sources are silently accepted
  but produce a misleading view of coverage. We could check this at
  concat time — at the cost of forcing every dataset to have a canonical
  UID field, which is not yet the case.

## Memory cost

Concatenation is **eager**: the result holds its own complete `(S, N)` grid
of response references. At deepSTRF scales (S, N in the low hundreds) this
is negligible — cross-block entries are single-element `(1, 1)` NaN
tensors weighing ~8 bytes each. The full `aa1 + aa2 + ns1` cross-block
overhead is on the order of kilobytes.

## See also

- [`examples/dataset_concatenation.ipynb`](../../examples/dataset_concatenation.ipynb)
  — runnable end-to-end demo on AA1 + AA2.
- [`data_paradigm.md`](data_paradigm.md) — the broader storage and
  missingness conventions that make concatenation natural.
