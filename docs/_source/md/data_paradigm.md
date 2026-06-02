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
| `self.stims`          | `list` of length `S`               | Each element is a stimulus tensor of modality-specific shape. Audio spectrogram: `(1, F, T_stim)`. Audio waveform: `(1, T_stim)` mono at `self.audio_fs` Hz. Video: `(1, H, W, T_stim)`. `T_stim` varies. |
| `self.responses`      | `list[list]` of length `S × N`     | `responses[s][n]` is a float tensor of shape `(R_{s,n}, T_resp_s)` (spike counts per repeat × time). `T_resp_s` is the **neural** time-bin count at `self.dt_ms`; in spectrogram mode it equals the stim's last axis, but for raw waveforms it is finer than the stim T (the stim runs at `audio_fs`, the response at neural rate). |
| `self.stim_meta`      | `list` of length `S`               | Per-stim metadata **dict**: e.g. `{"name": "...", "type": "...", ...}`. Fields vary per dataset.  |
| `self.nrn_meta`| `list` of length `N`               | Per-neuron metadata **dict**: e.g. `{"cell_id": "...", "animal_id": "...", "area": "..."}`.       |
| `self.N_neurons`      | `int`                              | Total neurons; equals `len(self.nrn_meta)`.                                                |
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
shapes become `(B, ..., T_stim_max)` for stims and `(B, N, R_max, T_resp_max)`
for responses. The two `T_*` axes are sized independently — equal in
spectrogram mode (one bin per neural sample), unequal in the raw-waveform
mode where the stim runs at `audio_fs` and responses stay at the dataset's
neural rate.

### 3.3 Repeat padding: `R_{s,n}` varies across neurons within a stim

Also introduced at batch time by the collate function. Responses are NaN-padded
along the repeat dimension up to `R_max`.

### 3.4 Raw-waveform inputs (audio only)

Audio datasets may expose an opt-in waveform-input mode (e.g.
`NS1Dataset(return_waveform=True, audio_fs=48000)`). The spectrogram transform
that would otherwise be baked into the dataset moves *into the model*, as the
first slot of the canonical pipeline (`wav2spec → prefiltering → core →
readout`), so it can be **learned** (SincNet, ICNet) rather than fixed.

In this mode:

- `self.stims[s]` is a `(1, T_audio)` mono float32 tensor at `self.audio_fs` Hz.
- `self.responses[s][n]` is unchanged — still `(R, T_neural)` at the dataset's
  `dt_ms`.
- The model owns the rate change: pair the waveform stim with a model whose
  `wav2spec` slot is a non-`Identity()` module (see `deepSTRF.models.wav2spec`).
  The slot maps `(B, 1, T_audio) → (B, 1, F, T_neural)` and the rest of the
  pipeline runs exactly as in spectrogram mode.

**Strict causality factors into three composable pieces.** deepSTRF's hard
requirement — model output at response bin `t` depends only on stimulus up to
wall-clock time `(t+1)·dt` — is preserved because:

1. **Grid lock (dataset).** `hop = audio_fs · dt_ms / 1000` is an integer and
   `T_audio = T_neural · hop`, pinning audio sample `j` to response bin
   `j // hop`.
2. **Per-frame causality (wav2spec).** Output frame `t` sees only audio samples
   `[0, (t+1)·hop)` — enforced by the Jacobian probe in
   `tests/test_wav2spec.py`.
3. **Downstream causality (rest of the pipeline).** Once wav2spec emits a
   frame-aligned causal spectrogram at the neural rate, the existing spec-domain
   contract (`tests/test_audio_models.py`) takes over unchanged.

Compose (1) and (2): frame `t` sees exactly the audio of bins `0…t` and nothing
later. The wav2spec is effectively a *causal resampler* from audio rate to neural
rate, and `hop` is the bridge between the two grids.

**Conventions for a dataset's waveform branch:**

- **C1 — grid lock.** `audio_fs · dt_ms / 1000` must be a positive integer, and
  each waveform must be cropped / padded to exactly `T_resp_s · hop` samples.
  Enforced for every audio dataset by `AudioNeuralDataset.validate()` (and
  surfaced as `self.hop`).
- **C2 — right-pad.** Variable-length waveforms are zero-padded on the *end*
  (append), mirroring the response NaN-padding, so bin-0 alignment is preserved.
  `neural_collate` does this automatically — one unified collate serves both
  modes (stims zero-padded, responses NaN-padded), so users never swap collate.
- **C3 — mono.** Waveforms are `(1, T_audio)`, downmixed to mono.
- **C4 — opt-in.** `return_waveform=True` is implemented only where genuine
  source audio exists. Synthetic-spectrogram-only stimuli (with no underlying
  waveform) stay spec-mode; a dataset with no source audio raises
  `NotImplementedError`.

**Matching a `wav2spec` to its dataset.** A front-end's `audio_fs` and `hop` must
agree with the dataset's, or frames won't align with response bins. The simple,
explicit path is to read them off the dataset:

```python
w = SincNet(audio_fs=ds.audio_fs, hop_ms=ds.dt, n_filters=ds.F)
# or:  w = make_wav2spec("sincnet", audio_fs=ds.audio_fs, dt_ms=ds.dt, ...)
```

A gross mismatch (non-`hop`-divisible `T_audio`) raises in the wav2spec's
`forward`; a subtler one (matching `audio_fs`, wrong `hop`) surfaces as a
prediction-vs-response shape error at the loss. There is no dataset↔model
auto-binding by design — the model holds no dataset reference, and the explicit
path above keeps things simple.

- **`hearing_range_hz` (informational).** Audio datasets may set an optional
  `(low, high)` Hz bound on the species' canonical hearing range (e.g.
  `(200.0, 40000.0)` for ferret). Purely advisory — nothing is enforced against
  it; tooling can display it and users may choose to clamp a wav2spec's
  frequency limits. `None` when unknown.
- Datasets without a waveform branch leave `self.audio_fs = None` (and `self.hop`
  is then `None`).

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

| Name          | Shape                              | Contents                                                                                  |
|---------------|------------------------------------|-------------------------------------------------------------------------------------------|
| `stims`       | `(B, 1, F, T_stim_max)` (audio spec) or `(B, 1, T_stim_max)` (audio waveform) | Float tensor. Zero-padded on the right along `T`. **Never contains NaN.**                 |
| `responses`   | `(B, N, R_max, T_resp_max)`        | Float tensor. NaN-padded on the right along `R` and `T_resp`; full-NaN slab where neuron n didn't hear stim s. |
| `valid_mask`  | `(B, N, R_max, T_resp_max)` `bool` | `~responses.isnan()`. Derived once per batch. Canonical "this position holds real data."  |
| `stim_metas`  | `list` length `B`                  | Per-stim metadata dicts, same as stored in `dataset.stim_meta`.                           |

If you need the coarser "did this neuron hear this stim" per batch-item
mask, recover it as `valid_mask.any(dim=(-1, -2))` — a `(B, N)` bool
tensor.

## 6. Recommended training-loop loss pattern

NaN handling at the loss / metric site is the responsibility of
``deepSTRF.metrics`` — every shipped function in that module is
NaN-aware by default. The training loop therefore stays simple:

```python
from deepSTRF.metrics import mse_loss, corrcoef, normalized_corrcoef

for stims, responses, valid_mask, stim_metas in loader:
    pred    = model(stims)                                # (B, N, 1, T_max)
    gt_psth = responses.nanmean(dim=2, keepdim=True)      # (B, N, 1, T_max)

    loss = mse_loss(pred, gt_psth)                        # scalar; handles NaN internally
    loss.backward()
    optimizer.step()

    if val_step:
        cc      = corrcoef(pred, gt_psth, reduction='none')
        cc_norm = normalized_corrcoef(pred, responses, method='schoppe',
                                      reduction='none')
```

Notes:

- ``responses.nanmean(dim=2, keepdim=True)`` is the canonical PSTH —
  carries NaN where all repeats are NaN, paired with ``pred`` shape.
- The dataloader's ``valid_mask`` is informational; redundant with
  ``~gt_psth.isnan()`` for prediction-vs-PSTH metrics.
- For non-default masking (e.g. excluding stimulus onsets), every
  metric accepts a ``mask=`` override.

**Detailed treatment** (per-neuron flattening, ``mask=`` override
semantics, length-weighting across stims, single-trial degenerate
handling, eval-only metrics): see
[`metrics_paradigm.md`](metrics_paradigm.md).

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
7. **`__len__` and `__getitem__` honour the current selection — bidirectionally.**
   See §8 below — the contract deserves its own section.
8. **Waveform mode obeys the grid lock.** When `self.audio_fs` is set, every
   stim is `(1, T_audio)` with `T_audio = T_resp_s · hop` and integer
   `hop = audio_fs · dt_ms / 1000` (§3.4, conventions C1–C4). Enforced by
   `AudioNeuralDataset.validate()`.

## 8. Iteration honours the current selection (bidirectional)

deepSTRF datasets carry two independent selection filters:

| Attribute     | Type           | "No restriction" sentinel | Default         |
|---------------|----------------|---------------------------|-----------------|
| `self.I`      | `list[int]`    | `[]`                      | `[]` (= all neurons) |
| `self.S_sel`  | `list[int]` or `None` | `None`             | `None` (= all stims) |

The two filters compose. `__len__` and `__getitem__` always agree on the
iterable subset of stimuli, defined as the **intersection** of:

- the stimuli with at least one valid response among the currently selected
  neurons (`self.I`-side filter), and
- the stimuli explicitly listed in `self.S_sel`, if it is set (`self.S_sel`-side filter).

The same intersection drives the *neuron* side: when `self.S_sel` is set,
neurons whose only valid responses lie outside the selected stim subset
are also hidden from the yielded responses. So selecting NAT4's val
subset (18 stims) automatically drops the 33 A1 cells that have no val
data — they would otherwise surface as full-NaN responses.

This is what we mean by "bidirectional": narrowing one side narrows the
other side too, when missingness in `nrn_masks` makes that the
information-preserving choice.

### Selection API

```python
# neuron-side — exact-match (one of the following)
ds.select_neuron(i)                          # single
ds.select_population([0, 1, 5])              # explicit list
ds.select_pop_by_nrn_attr("area", "MLd")     # by neuron metadata
ds.select_pop_by_stim_attr("subset", "val")  # neurons with >=1 valid resp on val stims

# neuron-side — predicate (threshold, range, compound)
ds.select_pop_by_nrn_predicate(lambda n: n.get("snr", 0) > 0.5)
ds.select_pop_by_stim_predicate(lambda s: s["duration_s"] >= 2.0)

# stim-side (one of the following)
ds.select_stim(i)                            # single
ds.select_stims([0, 5, 10])                  # explicit list
ds.select_stims_by_attr("subset", "val")     # by stim metadata
ds.select_stims_by_predicate(lambda s: s["type"] in {"song", "call"})
ds.reset_stim_selection()                    # clear S_sel (back to None)
```

The `*_predicate` variants take any `callable(dict) -> bool`, so threshold
(`snr > 0.5`), range (`200 <= depth_um <= 800`), and compound conditions
are expressible. Forgiving missing-key semantics match the `*_attr`
siblings (`KeyError` / `TypeError` silently skip), so a single predicate
works on a concatenated dataset whose sources carry heterogeneous metadata
schemas.

### Opt-in per-neuron quality metrics

Call `ds.compute_neuron_quality()` once after construction to write two
scalars into each `nrn_meta[i]`:

```python
ds.compute_neuron_quality()
ds.nrn_meta[0]   # → {..., "snr": 0.52, "ccmax": 0.91}
ds.select_pop_by_nrn_predicate(lambda n: n["snr"] > 0.5)
```

- `'snr'` — Sahani-Linden signal-to-noise ratio $\text{SP}_n / \text{NP}_n$,
  length-weighted across stims (weight = number of valid time bins per
  stim, matching the convention in `metrics_paradigm.md` §11). NaN
  when the neuron has no stim with $R_{b,n} \ge 2$ and $T_b \ge 2$.
- `'ccmax'` — Hsu/Spearman-Brown noise ceiling, length-weighted across
  stims with $R_{b,n} \ge 2$. Falls back to `1.0` when the neuron has
  zero such stims (R=1 everywhere — no normalization possible, so
  `cc_norm = cc_raw`).

Opt-in (not auto-called in `__init__`) because CCmax is
$O(S \cdot N \cdot R^2 \cdot \text{max\_iters})$ in the worst case — on
big datasets (AA4, NAT4, Espejo NAT) this runs in tens of seconds.

Why two selectors are sometimes both needed:

- `select_pop_by_stim_attr("subset", "val")` keeps the neurons with val
  data, but leaves *all* stims iterable (you might want est responses
  for those neurons too).
- `select_stims_by_attr("subset", "val")` keeps the val stims, and
  *also* drops the cells that lack val data via the bidirectional rule.

So the "I want to train on val only, with val-having cells only" recipe
is one call to `select_stims_by_attr("subset", "val")` — the bidirectional
rule does the other half.

### Why this matters

- **DataLoader compatibility.** PyTorch iterates `range(len(ds))` and
  calls `ds[i]` for each. The two have to agree, or the loader requests
  indices that map to fully-NaN items. The filter is what makes them
  agree.
- **Concatenated datasets are safe.** When you concatenate sources A and
  B and then select only A's neurons, B's stimuli are *automatically*
  hidden — their responses against A's neurons are all NaN, so they
  don't survive the filter. No special-case logic in the training loop.
- **Selecting on stim attributes is implicit too.** Selecting only the
  neurons that heard a particular stim type narrows iteration to those
  stims, by transitivity. This subsumes most one-off "iterate only the
  stims that match X" needs.

### Concrete consequences

```python
ds = concat_neural_datasets([aa1, aa2])     # 30 + 117 stims, 100 + 494 neurons
len(ds)                                      # 147 — full population by default

ds.select_population(list(range(100)))       # only AA1's neurons
len(ds)                                      # 30 — AA2's stims now hidden
ds[0]                                        # the FIRST iterable stim under selection
                                             #  i.e. AA1's stim 0; same as before selection
ds[29]                                       # AA1's stim 29 — last iterable
ds[30]                                       # IndexError, not a fully-NaN AA2 stim

ds.select_pop_by_nrn_attr("area", "MLd")     # MLd neurons across A and B
len(ds)                                      # however many stims any MLd neuron heard

# stim filter + bidirectional rule
nat4 = NAT4Dataset(area="A1")               # 593 stims, 849 cells
nat4.select_stims_by_attr("subset", "val")   # 18 val stims, 816 val-having cells
len(nat4)                                    # 18; the 33 val-less cells are gone
```

### `S_sel = None` vs `S_sel = []`

The two are intentionally distinct:

- `S_sel = None` means "no restriction" — iteration spans all stims.
  Returned by `reset_stim_selection()` and the default at construction.
- `S_sel = []` means "explicit zero-stim selection" — iteration yields
  zero items. This is what `select_stims_by_attr("foo", "bar")` produces
  when no stim matches. The asymmetry with `self.I` (which uses `[]` for
  "no restriction") is deliberate: selecting by an attribute that no stim
  carries should *not* silently disable the filter.

### Raw access vs iteration

The stored attributes (`self.stims`, `self.responses`, `self.stim_meta`,
`self.nrn_meta`, `self.nrn_masks`) are *not* filtered. They keep
the dataset's full structure regardless of `self.I` or `self.S_sel`. To
address a raw stim by its absolute index, read those attributes directly:

```python
ds.stims[42]                # raw stim 42, regardless of selection
ds.responses[42]            # all neurons' responses to stim 42
ds.nrn_masks[42]            # which neurons have data for stim 42
```

The selection filter applies only at the iteration interface (`__len__`,
`__getitem__`).

## 9. Gotchas

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

## 10. Dataset concatenation

The NaN-sentinel convention enables a bonus feature: concatenating neural
datasets along *both* the stim and neuron axes is essentially free — the
cross-block `(stim_A, neuron_B)` entries reuse the same `(1, 1)` NaN tensor
used elsewhere for structural missingness, and `nrn_masks` reports the
block-diagonal coverage automatically.

```python
from deepSTRF.utils.data import concat_neural_datasets
combined = concat_neural_datasets([aa1, aa2])   # or: aa1 + aa2
```

See [`dataset_concatenation.md`](dataset_concatenation.md) for the full
feature page (rationale, compatibility rules, chimeric-model use case)
and [`examples/dataset_concatenation.ipynb`](../../examples/dataset_concatenation.ipynb)
for a runnable demo.

## 11. When the paradigm might need to evolve

- Migration to `torch.nested` tensors once the ecosystem matures — would
  remove explicit padding, possibly with model-side support gaps.
- Length-bucketed batching for datasets with very heterogeneous `T_s` —
  optimization, not a redesign.
- Per-repeat weighting (some repeats noisier than others) — would require
  either extending the mask from bool to float, or adding a separate weight
  tensor.

These are explicitly *not* on the current roadmap; flagged here so the
discussion doesn't have to be rediscovered.
