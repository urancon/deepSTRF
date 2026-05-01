# The deepSTRF metrics paradigm

This note documents how deepSTRF computes losses and performance metrics on
predictions produced by encoding models, and what invariants any metric
function must respect. It is the third leg of the data/model/metrics
contract, alongside [`data_paradigm.md`](data_paradigm.md) and
[`model_paradigm.md`](model_paradigm.md), and is the single reference for
the public functional API in `deepSTRF.metrics`.

## 1. Scope: a functional, NaN-aware, single-axis API

`deepSTRF.metrics` ships **pure functions** that map predictions and ground
truth onto either a per-neuron tensor (`reduction='none'`) or a single
scalar (`reduction='mean'|'sum'`). Two flavours:

| Flavour       | Examples                                           | Gradient? | NaN handling                                         |
|---            |---                                                  |---        |---                                                   |
| **Losses**    | `mse_loss`, `poisson_loss`                          | yes       | Internal: NaN-positions are dropped before reducing  |
| **Metrics**   | `corrcoef`, `normalized_corrcoef`, `fve`, `signal_power`, `noise_power`, `snr`, `coherence` | no — `@torch.no_grad()` | Internal, same as losses                             |

**Out of scope for v1** (deferred to follow-ups, listed at the end):
torchmetrics-style stateful module API, `feature_decorrelation`,
`pop_corrcoef`. Anything population-aggregated. Cross-batch accumulators.

## 2. Input shape conventions

The metrics API matches the data and model contracts exactly. There is no
"reshape your tensors before calling a metric" step.

| Tensor              | Shape                  | Source                                                      |
|---                  |---                      |---                                                          |
| `pred`              | `(B, N, 1, T)`          | model `forward()` output (`R = 1` axis from `model_paradigm.md` §3) |
| `responses`         | `(B, N, R, T)`          | dataloader output, NaN-padded                               |
| `gt_psth`           | `(B, N, 1, T)`          | typically `responses.nanmean(dim=2, keepdim=True)`           |
| `valid_mask`        | `(B, N, R, T)` bool     | dataloader output, derived from `~responses.isnan()`         |

The R-axis is the only friction point. Most metrics need a PSTH (one
ground-truth time series per neuron); the repeat-aware metrics
(`signal_power`, `noise_power`, `snr`, `normalized_corrcoef`) need the raw
repeats. The two functional signatures handle this without a class
hierarchy:

```python
# prediction-vs-PSTH metrics: pred and gt have the SAME shape, R=1 on both sides.
corrcoef(pred, gt_psth, mask=None, reduction='mean')              # (B, N, 1, T) × (B, N, 1, T)
fve(pred, gt_psth, mask=None, reduction='mean')                   # (B, N, 1, T) × (B, N, 1, T)
mse_loss(pred, gt_psth, mask=None, reduction='mean')              # (B, N, 1, T) × (B, N, 1, T)
poisson_loss(pred, gt_psth, mask=None, reduction='mean')          # (B, N, 1, T) × (B, N, 1, T)

# repeat-aware metrics: ground truth keeps its R dimension.
signal_power(responses, mask=None, reduction='mean')              # (B, N, R, T)
noise_power(responses, mask=None, reduction='mean')               # (B, N, R, T)
snr(responses, mask=None, reduction='mean')                       # (B, N, R, T)
normalized_corrcoef(pred, responses, method='schoppe',            # (B, N, 1, T) × (B, N, R, T)
                    mask=None, reduction='mean')

# eval-only frequency-domain metric: needs a regular grid, NaN-free
coherence(pred, gt_psth, dt_ms, reduction='mean')                  # (B, N, 1, T) × (B, N, 1, T)
```

The `R = 1` singleton on `pred` mirrors what the model emits. We do *not*
silently squeeze it — that would break a future probabilistic model that
populates the axis with multiple samples.

## 3. The reduction over time happens inside

Every metric computes one scalar per neuron by reducing across the *entire
batch and time axis*, in a way that depends on the metric's mathematical
definition. The user does not pick this. Specifically:

- `corrcoef`, `fve`, `normalized_corrcoef`, `coherence`, `signal_power`,
  `noise_power`, `snr` all flatten `(B, T)` (and `R`, when present) into
  one long pseudo-time series per neuron, and compute their formula on
  that. Boolean indexing with `valid_mask` makes this safe even when stim
  durations vary or some neurons are uncorded for some stims.
- `mse_loss` and `poisson_loss` compute the per-element residual, then
  apply `reduction` over the *valid* positions only.

This matches Schoppe (2016)'s original derivation, which assumes one
concatenated long signal per neuron. Per-stim CC values that are then
averaged across stims (the "Pennington recipe", §6.4) are a *separate*
quantity; we do not ship it in v1, and document the difference if a future
follow-up adds it.

The takeaway: **the time axis is collapsed by the metric's formula, not by
`reduction`.** What `reduction` controls is the *neuron* axis only.

## 4. NaN handling

The data paradigm uses NaN as the universal sentinel for missingness
(`data_paradigm.md` §3–4). All metrics in `deepSTRF.metrics` are
NaN-aware by default and require **no** preprocessing:

```python
loss = mse_loss(pred, gt_psth)         # picks up NaNs from gt_psth automatically
cc   = corrcoef(pred, gt_psth)         # same
sp   = signal_power(responses)         # NaN-padded repeats handled internally
```

Internally, each metric derives `valid = ~gt.isnan()` (or the analogous
mask on `responses`) and uses boolean indexing per-neuron — i.e. for
neuron `n`:

```python
pred_n = pred[..., n, :, :][valid[..., n, :, :]]   # 1-D
gt_n   = gt  [..., n, :, :][valid[..., n, :, :]]   # 1-D
result[n] = formula(pred_n, gt_n)
```

This is the same recipe `data_paradigm.md` §6 recommends for the training
loop. Per-neuron flattening keeps every metric robust to:

- Uncorded `(stim, neuron)` pairs (full-NaN slabs in `responses`).
- Variable `T_s` across stims (right-NaN-padded by `neural_collate`).
- Variable `R_{s, n}` across neurons within a stim (right-NaN-padded).

### `mask=` override

For cases where the canonical "valid iff not NaN" rule is not what the
user wants — for example, masking out the first 50 ms of every stim to
exclude onset transients — every metric accepts an optional `mask` kwarg:

```python
mask = ~gt_psth.isnan() & (time_idx >= onset_offset)
cc   = corrcoef(pred, gt_psth, mask=mask)
```

`mask` must be a bool tensor broadcastable to the ground-truth tensor's
shape. If `mask=None` (the default), the metric falls back to
`~gt.isnan()`. If `mask` is provided, it **replaces** (not augments) the
NaN-derived mask: the user is now responsible for keeping the
NaN-positions out — typically by `&`ing them in as shown above. We
document this explicitly because the alternative ("mask is always
intersected with `~isnan`") is harder to reason about when the user
deliberately wants to mark NaN positions as valid (impossible — they would
contaminate the result; we do not protect against this).

## 5. Reduction semantics (PyTorch convention)

Every public metric has a `reduction` keyword with three values, matching
`torch.nn.functional.mse_loss` and friends:

| `reduction` | Returned shape | Meaning                                    |
|---          |---              |---                                         |
| `'none'`    | `(N,)`          | one value per neuron                       |
| `'mean'`    | `()` (scalar)   | mean across neurons                        |
| `'sum'`     | `()` (scalar)   | sum across neurons                         |

**Reduction is over the neuron axis only.** Reduction over time is decided
by the metric's mathematical formula and is never controllable via
`reduction` (§3 above).

Edge case: when one or more neurons have **zero valid positions** under
the chosen mask (e.g. no val data for a particular cell), the per-neuron
result is `NaN`. Under `'mean'` and `'sum'` we use `torch.nanmean` /
`torch.nansum` so the scalar reduction stays well-defined. Under
`'none'`, the NaN is preserved — the caller can see which cells were
dropped.

## 6. Per-metric definitions

Each metric is documented with its formula, shape contract, edge cases,
and reference. Where the existing implementation has a known bug or
band-aid (`abs(ttrc)`, capitalized `'None'`, etc.), the rewrite either
fixes or removes the band-aid; this section is the authoritative spec.

### 6.1 `mse_loss(pred, gt, mask=None, reduction='mean')`

Boolean-masked mean squared error.

For each neuron, take the per-element squared residual, drop positions
where `mask` is False, and reduce.

```text
mse_n = mean_{(b,t) ∈ valid_n}  (pred[b, n, 0, t] - gt[b, n, 0, t])²
```

- `reduction='none'`: returns `(N,)`.
- `reduction='mean'`: scalar, `nanmean(mse_n, dim=0)`.
- `reduction='sum'`: scalar, `nansum(mse_n, dim=0)`.

Differentiable. No `@torch.no_grad`.

### 6.2 `poisson_loss(pred, gt, mask=None, reduction='mean', eps=1e-8)`

Negative Poisson log-likelihood, dropping the `log(rate!)` term that does
not depend on `pred`. Following Wang et al. 2025 and Singer et al. 2023:

```text
poisson_n = mean_{(b,t) ∈ valid_n}  pred[b, n, 0, t]  -  gt[b, n, 0, t] · log(pred[b, n, 0, t] + ε)
```

`pred` must be non-negative (e.g. `Softplus` output). Negative `pred` is
not silently corrected — we raise `ValueError` if any masked-in element
is negative, mirroring how `data_paradigm.md` errs on the side of loud
failure.

`eps` keeps `log(0)` finite without changing the gradient meaningfully.
Reduction over `N` is the same as `mse_loss`. Differentiable.

The current implementation in `losses.py` (`NegativePoissonLogLikelihood`)
has a sign bug (`torch.log(-prediction + 1e-9)`). The rewrite removes
this; see commit message of the dedicated commit.

### 6.3 `corrcoef(pred, gt, mask=None, reduction='mean')`

Pearson correlation coefficient between prediction and PSTH, computed
per-neuron over all valid positions:

```text
ρ_n = corr(pred_n, gt_n)
```

where `pred_n`, `gt_n` are the 1-D vectors of valid `(b, t)` positions for
neuron `n` (§4). Uses `torch.corrcoef`, denominator regularized with a
small floor to avoid `0/0` on constant predictions.

- `mean` / `sum` reduce over `N`.
- A neuron with zero variance in either `pred_n` or `gt_n` returns NaN
  for that index — propagation rules from §5 apply.

### 6.4 `normalized_corrcoef(pred, responses, method='schoppe', mask=None, reduction='mean')`

Noise-corrected correlation coefficient. Two formulations are shipped;
they target the same quantity with different bias profiles.

The shape contract differs from `corrcoef`: ground truth is the raw
repeat-bearing tensor `(B, N, R, T)`, not the PSTH. The PSTH is computed
internally as `responses.nanmean(dim=2, keepdim=True)`, but the per-trial
information is needed for the noise correction.

#### `method='schoppe'`

Schoppe et al. 2016, "Measuring the Performance of Neural Models",
Frontiers in Neuroscience.

```text
CCnorm_n = cov(pred_n, psth_n) / sqrt( var(pred_n) · SP_n )
```

where `pred_n`, `psth_n` are the 1-D vectors of valid `(b, t)` positions
for neuron `n` (concatenated across stims via boolean indexing, §3),
and `SP_n` is the Sahani–Linden signal power for neuron `n` (§6.5,
per-stim-then-`nanmean` across stims). Replaces the noise-inflated
`var(psth)` in the denominator of the raw Pearson correlation with the
unbiased signal-only variance estimator. Cells with `SP_n ≤ 0` (the
noise-floor regime where the unbiased estimator fails) return NaN.

#### `method='hsu'`

Hsu, Borst & Theunissen 2004, with the `CCmax` formulation popularized
by Schoppe et al. 2016.

```text
CCnorm_n     = corr(pred_n, psth_n) / CCmax_n
CCmax_{b,n}  = sqrt( 2 · ρ_half_{b,n} / (1 + ρ_half_{b,n}) )           (Spearman–Brown, per stim)
ρ_half_{b,n} = E_{i,j}[ corr( psth_half_{i,b,n}, psth_half_{j,b,n} ) ] (over disjoint half-trial splits)
CCmax_n      = nanmean_b ( CCmax_{b,n} )
```

Computed per-stim by sampling up to `ccmax_iters=126` disjoint
half-trial pairs, then averaged across stims (`nanmean`). Same
robustness convention as `signal_power` (§6.5). Per-stim cells with
`ρ_half ≤ 0` (worse-than-chance ceiling — too noisy to estimate)
contribute NaN to the average.

#### Single-trial degenerate case

When `R = 1` (or every cell of `responses[..., n, :, :]` has only one
valid repeat after masking), the noise-correction factor is undefined:
both `SP_n` and `CCmax_n` would require ≥ 2 trials. In that case
`normalized_corrcoef` returns the raw `corrcoef(pred, psth)` for that
neuron. This is a *worst-case* assumption: with no information about
trial-to-trial variability, we cannot know how much of the prediction
gap is noise vs model error, so we report the un-corrected number. It is
the same convention used by the existing implementation and by
Schoppe (2016) §3.5.

#### Pennington 2023 — *not shipped in v1*

The Pennington & David (2023) noise-corrected prediction correlation
divides by `sqrt(TTRC)` where `TTRC` is the average pairwise trial-pair
correlation. The existing implementation includes this as
`pennington_prediction_correlation`, with an `abs(ttrc)` band-aid that
the audit flags as wrong. Re-derivation: TTRC can be slightly negative
in the low-SNR regime and Pennington's paper is silent on what to do; the
field's typical fix is to clip-to-ε with a warning, but this changes the
reported number for poorly-driven cells in a way that has been
benchmarked in deepSTRF. We **defer** the Pennington method to a
follow-up rather than ship a rushed fix.

### 6.5 `signal_power(responses, mask=None, reduction='mean')`

Sahani & Linden 2003, NIPS, "How linear are auditory cortical responses?".

For one `(b, n)` cell with `R` valid repeats, the variance of the
multi-trial response decomposes as

```text
TP_{b,n} = E_r [ var_t( y_{r, b, n}(t) ) ]            = SP_{b,n} + NP_{b,n}             (single-trial variance)
           var_t( E_r[ y_{r, b, n}(t) ] )             = SP_{b,n} + NP_{b,n} / R          (variance of the PSTH)
```

Solving for `SP_{b,n}`:

```text
SP_{b,n} = ( R · var_t(psth_{b,n}) - TP_{b,n} ) / (R - 1)
```

The per-neuron quantity is the **mean across stimuli** of the per-stim
estimates:

```text
SP_n = nanmean_b ( SP_{b,n} )
```

This per-stim-then-average convention is what makes the metric robust to
the variable-`R` setting of the data paradigm: each stim contributes its
own valid repeat count, and stims with fewer than 2 valid repeats or
fewer than 2 valid time bins are skipped entirely. Cells without any
qualifying stim return NaN under `reduction='none'`.

### 6.6 `noise_power(responses, mask=None, reduction='mean')`

```text
NP_{b,n} = TP_{b,n} - SP_{b,n}
NP_n     = nanmean_b ( NP_{b,n} )
```

Same per-stim-then-average convention as `signal_power`.

### 6.7 `snr(responses, mask=None, reduction='mean')`

```text
SNR_n = SP_n / NP_n
```

Returns `+inf` for noiseless cells (`NP_n = 0`) — caller decides whether
to filter.

### 6.8 `coherence(pred, gt, dt_ms, reduction='mean')`

Magnitude-squared coherence between prediction and PSTH per neuron, as a
scalar per neuron computed as the mean over frequency bins. Uses
`scipy.signal.coherence` internally, so this is **eval-only** (no
gradient).

Coherence requires a regular time grid and **does not tolerate NaN**.
The caller is responsible for picking a NaN-free subset (e.g. one fully
recorded validation stim, or pre-flatten across stims of equal length).
We document this explicitly and raise `ValueError` if NaN appears in the
input. Future follow-up: a chunked-coherence variant that handles
NaN-padded variable-length stims.

This metric is preserved largely as it was — the existing implementation
is correct for its supported shape — but adapted to the new
`(B, N, 1, T)` × `(B, N, 1, T)` contract.

## 7. Recommended training-loop pattern

Mirroring `data_paradigm.md` §6, the canonical loop now looks like:

```python
from deepSTRF.metrics import mse_loss, corrcoef, normalized_corrcoef, signal_power

for stims, responses, valid_mask, stim_metas in loader:
    pred     = model(stims)                                       # (B, N, 1, T)
    gt_psth  = responses.nanmean(dim=2, keepdim=True)             # (B, N, 1, T)

    loss = mse_loss(pred, gt_psth)                                # scalar
    loss.backward()
    optimizer.step()

    if val_step:
        cc       = corrcoef(pred, gt_psth, reduction='none')       # (N,)
        cc_norm  = normalized_corrcoef(pred, responses,            # (N,)
                                       method='schoppe',
                                       reduction='none')
```

Notes:

1. **No manual masking.** `gt_psth` carries the NaN sentinels through
   from `responses.nanmean(dim=2, keepdim=True)`; every metric picks them
   up via §4. The dataloader's `valid_mask` is informational; it is
   redundant with `~gt_psth.isnan()` for prediction-vs-PSTH metrics.
2. **`responses.nanmean(dim=2, keepdim=True)` is the canonical PSTH.**
   It survives full-NaN slabs (returns NaN at those positions, which the
   metric drops), survives R-padded NaNs, and keeps the `R=1` singleton
   so it pairs trivially with `pred`.
3. **`reduction='none'` is the default for val metrics.** Per-neuron
   numbers are usually what you want for diagnostics (best/worst cells,
   distribution plots). Aggregate yourself with `cc.nanmean()` if a
   single number is needed.
4. **Use `mask=` only when overriding the default NaN rule.** The 90 %
   case (mask = `~isnan`) is automatic.

## 8. Invariants for metric authors

Anyone adding a new metric to `deepSTRF.metrics` must respect:

1. **Shape signature matches §2.** Either `(B, N, 1, T)` × `(B, N, 1, T)`
   for prediction-vs-PSTH metrics, or `(B, N, R, T)` for repeat-aware
   metrics. No other input shape is supported in v1.
2. **NaN-aware by default.** Derive `valid` internally; expose `mask=`
   for override.
3. **`reduction='none'|'mean'|'sum'` over the neuron axis.** Use
   `nanmean` / `nansum` for `mean` / `sum` to handle dropped cells
   gracefully.
4. **`@torch.no_grad()` on metrics; no `no_grad` on losses.** Losses
   participate in backprop; metrics do not. Mixing them up is the
   single biggest bug pattern in the existing module.
5. **Reduction is over neurons, never over time.** Time is collapsed by
   the metric's formula. If your formula needs a different time-axis
   convention, document it explicitly and add a test.
6. **Loud failure on shape mismatch.** Raise `ValueError` with the
   expected vs received shape; do not auto-broadcast.
7. **Tests under `tests/test_metrics.py`** assert: (a) shape contracts,
   (b) NaN handling on a synthetic full-NaN slab, (c) reduction
   semantics, (d) at least one analytic correctness check
   (e.g. `corrcoef` of a tensor with itself = 1, `signal_power` on a
   pure noise signal ≈ 0).

## 9. Known bugs in the existing implementation (audit)

This section is the rewrite checklist. The existing `metrics/performance.py`
and `metrics/losses.py` will be rewritten — not patched in place — to
satisfy the contracts above. Issues found during the 2026-04-28 audit:

1. **`@torch.no_grad()` on losses.** `NegativePoissonLogLikelihood` is
   correctly a `nn.Module` (no `no_grad`), but every "performance" function
   in `performance.py` has `@torch.no_grad()`. Splitting losses vs metrics
   into two modules makes the contract obvious.
2. **No NaN handling anywhere.** Silently produces NaN if any input
   element is NaN. Critical given the data paradigm.
3. **`covariance` uses `(T - 1)` denominator; `var(dim=-1)` is
   Bessel-corrected by default.** Mixing biased and unbiased estimators in
   the same formula introduces a small but real bias. Pick one (Bessel)
   and apply consistently.
4. **`reduction='None'` (capitalized)** in `correlation_coefficient` and
   `normalized_correlation_coefficient` is a typo. Switch to PyTorch
   convention `'none'|'mean'|'sum'`.
5. **`pennington_prediction_correlation` uses `abs(ttrc)`** — a band-aid
   for sign issues that shouldn't appear in a correctly-derived formula.
   Re-derive (or defer; see §6.4).
6. **`compute_CCmax` uses `1/sqrt(cchalf²)` = `1/|cchalf|`** instead of
   `1/cchalf`. Same band-aid pattern as #5; equivalent to `|ρ_half|` in
   the Spearman–Brown formula. Replace with the textbook form
   `sqrt(2·ρ_half / (1+ρ_half))` and clip-or-NaN low-SNR cells.
7. **`fill_missing_repeats`** is a dataset-side helper that the modernized
   NaN-sentinel path obsoletes. Drop from metrics.
8. **`coherence` returns a `(B, N, F)` tensor that is then averaged over
   B, not over F** — different from the per-neuron-scalar contract here.
   Adapt to §6.8.
9. **`NegativePoissonLogLikelihood` computes `log(-prediction + 1e-9)`** —
   negation inside `log` is a bug. Should be `log(prediction + ε)`. Also
   uses `'None'` (capitalized) and computes mean over `(-2, -1)` then over
   batch, which is `'mean'` over all elements — fine for the loss but the
   reduction semantics are inconsistent with the new convention.

## 10. Deferred items

These were considered for v1 and explicitly cut. Restoration will
re-open this doc.

- **Module API (torchmetrics-style stateful `update()` / `compute()`).**
  Useful for streaming val metrics across an epoch without materializing
  every batch. The functional API is enough for v1; module API is easy
  to add later as a thin wrapper.
- **`feature_decorrelation`** / redundancy-reduction regularizer. Not
  part of the original deepSTRF scope and uses a different design pattern
  (forward hooks on `model.core`).
- **`pop_corrcoef`** (correlation of population-summed PSTH). Uncommon
  metric; no immediate use case.
- **Pennington-style noise-corrected prediction correlation** (§6.4).
  Re-derivation pending; current `abs(ttrc)` is wrong, simple ε-clip is
  not vetted.
- **Per-stim-then-averaged metrics.** Some of the literature reports
  `mean_s CC(stim_s)` rather than `CC(concat_s)`. Different quantity;
  not shipped in v1.
- **Chunked-coherence for NaN-padded variable-length stims** (§6.8).

## 11. References

- **Sahani, M. & Linden, J. F. (2003).** "How linear are auditory cortical
  responses?" *Advances in Neural Information Processing Systems (NIPS)*.
  — Original signal/noise power decomposition.
- **Hsu, A., Borst, A. & Theunissen, F. E. (2004).** "Quantifying
  variability in neural responses and its application for the validation
  of model predictions." *Network: Computation in Neural Systems*. —
  CCmax formulation via Spearman–Brown.
- **Schoppe, O., Harper, N. S., Willmore, B. D. B., King, A. J. &
  Schnupp, J. W. H. (2016).** "Measuring the performance of neural
  models." *Frontiers in Computational Neuroscience*. — CCnorm
  (signal-power formulation), reconciliation with Hsu et al.
- **Pennington, J. R. & David, S. V. (2023).** "A convolutional neural
  network provides a generalizable model of natural sound coding by
  neural populations in auditory cortex." *PLOS Computational Biology*. —
  TTRC-based noise-corrected prediction correlation.
- **Gill, P. et al. (2025).** "Sound representation methods for
  spectro-temporal receptive field estimation." *Journal of Computational
  Neuroscience*. — Coherence-information metric (deferred follow-up).
