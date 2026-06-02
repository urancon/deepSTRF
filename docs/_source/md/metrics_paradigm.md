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
# prediction-vs-PSTH metrics: gt may be the PSTH (B, N, 1, T) OR raw responses (B, N, R, T).
# When R > 1 the metric collapses to PSTH internally via nanmean(dim=2, keepdim=True).
corrcoef(pred, gt, mask=None, reduction='mean')                   # (B, N, 1, T) × (B, N, 1|R, T)
fve(pred, gt, mask=None, reduction='mean')                        # (B, N, 1, T) × (B, N, 1|R, T)
mse_loss(pred, gt, mask=None, reduction='mean')                   # (B, N, 1, T) × (B, N, 1|R, T)
poisson_loss(pred, gt, mask=None, reduction='mean')               # (B, N, 1, T) × (B, N, 1|R, T)

# repeat-aware metrics: ground truth keeps its R dimension.
signal_power(responses, mask=None, reduction='mean')              # (B, N, R, T)
noise_power(responses, mask=None, reduction='mean')               # (B, N, R, T)
snr(responses, mask=None, reduction='mean')                       # (B, N, R, T)
normalized_corrcoef(pred, responses, method='schoppe',            # (B, N, 1, T) × (B, N, R, T)
                    mask=None, reduction='mean')

# eval-only frequency-domain metric: needs a regular grid, NaN-free, PSTH-only
coherence(pred, gt_psth, dt_ms, reduction='mean')                  # (B, N, 1, T) × (B, N, 1, T)
```

The `R = 1` singleton on `pred` mirrors what the model emits. We do *not*
silently squeeze it — that would break a future probabilistic model that
populates the axis with multiple samples.

### Auto-PSTH collapse on the `gt` argument

The four prediction-vs-PSTH metrics (`mse_loss`, `poisson_loss`,
`corrcoef`, `fve`) accept `gt` in either canonical shape:

- `(B, N, 1, T)` — pre-computed PSTH or a deliberately chosen
  single-trial target — used as-is.
- `(B, N, R, T)` with `R > 1` — raw responses — collapsed to PSTH
  internally via `nanmean(dim=2, keepdim=True)` before the formula is
  applied.

This means the canonical training step is

```python
loss = mse_loss(pred, responses)            # one line, auto-PSTH
```

with no caller-side `responses.nanmean(...)` boilerplate. NaN-padded
missing-trial slabs flow through the `nanmean` cleanly: a position where
every repeat is NaN comes out NaN at the collapsed PSTH and is dropped
by the metric's NaN-derived mask (§4). When `gt` is already
`(B, N, 1, T)` the collapse is a no-op, so passing a custom target
(e.g. for a non-PSTH fitting objective) keeps working unchanged.

The repeat-aware metrics (`signal_power`, `noise_power`, `snr`,
`normalized_corrcoef`) and `coherence` do **not** auto-collapse: the
first three need the per-trial structure for their formulas, and
`coherence` is documented as PSTH-only and NaN-intolerant (§6.8).

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

### Loud failure on misuse

If the user passes a `mask` that marks a NaN position as valid, the
metric's per-element computation is performed at that position
(`(pred − NaN)**2 = NaN`, `gt · log(pred + ε)` with `gt = NaN` = NaN,
etc.). NaN propagates through the per-neuron sum into the per-neuron
output. The result is then a visibly NaN per-neuron value — *not* a
silently-dropped or zero-contaminated number. This matches the
loud-failure ethos of the data paradigm (`data_paradigm.md` §4): users
spot misuse via NaN in their reported metric, not via subtle drift.

Implementation note: this is achieved by `masked_fill(~valid, 0.0) →
sum / count` rather than `nanmean`, which would silently drop NaN.

### Boolean indexing vs multiplicative masking

Two natural ways to compute a masked reduction:

```python
# (a) Boolean indexing — flat 1-D vector of valid positions
ssr = ((pred[valid] - gt[valid]) ** 2).mean()

# (b) Multiplicative masking — preserves axis structure
per_element = (pred - gt.nan_to_num(0.0)) ** 2 * valid.float()
per_axis    = per_element.sum(dim=AXIS) / valid.float().sum(dim=AXIS)
```

The shipped metrics use **boolean indexing** internally for the per-neuron
formula, because:

- The flat 1-D form keeps `cov` / `var` / Pearson correlation textbook-clean
  (no per-row `.nansum()` denominators).
- It is numerically robust against `0 · NaN = NaN` traps that
  multiplicative masking can hit when NaN has leaked into `pred`.

Multiplicative masking is the right choice when you need to preserve the
`(B, N, T)` axis structure for further aggregation (e.g. per-neuron
losses with custom batch reduction outside the metric API). In that
case, work directly with the dataloader's `valid_mask` rather than
calling a deepSTRF metric — the metrics are designed to terminate at
per-neuron scalars.

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

### 6.2 `poisson_loss(pred, gt, mask=None, reduction='mean', log_input=False, validate_input=False, eps=1e-8)`

Negative Poisson log-likelihood, dropping the `log(gt!)` term. Two
parameterisations of the prediction are supported, matching the
canonical-link logic of generalised linear models:

#### `log_input=False` (default): pred is the rate `λ`

```text
poisson_n = mean_{(b,t) ∈ valid_n}  pred[b, n, 0, t]  -  gt[b, n, 0, t] · log(pred[b, n, 0, t] + ε)
```

Requires `pred ≥ 0` for the `log` to be meaningful. The implementation
*silently clamps* `pred` to `≥ ε` inside the `log` to prevent NaN from
propagating when the model temporarily emits a small negative
prediction. The linear `pred` term outside the `log` keeps its sign,
so the gradient still pushes negative predictions toward positive
values. With `validate_input=True`, the loss instead raises
`ValueError` on the first negative prediction at a masked-in position
— useful for debugging an output-activation bug, but adds a per-step
CPU sync, so it is opt-in.

#### `log_input=True`: pred is the log-rate `η = log(λ)`

```text
poisson_n = mean_{(b,t) ∈ valid_n}  exp(pred[b, n, 0, t])  -  gt[b, n, 0, t] · pred[b, n, 0, t]
```

Derivation: starting from the Poisson NLL `λ − gt · log(λ)` and
substituting `λ = exp(η)`, the loss becomes `exp(η) − gt · η`, well-defined
for any real-valued `η`. This is the **canonical link** for the Poisson
distribution in GLM language — same role that the logit link plays for
Bernoulli. Practically: pair `log_input=True` with **any** readout
(Linear, ParametricSigmoid, etc.) and read the model output as
log-rate. To recover rate for visualisation or downstream metrics
expecting rate (e.g. `corrcoef` against a count-valued PSTH),
`exp(pred)` first.

#### Stirling term

The full Poisson NLL has a `gt · log(gt) − gt + 0.5·log(2π·gt)` term
(Stirling's approximation to `log(gt!)`). It is **not** added by
default because:

- It is constant in `pred`, so it does not affect optimisation.
- It is meaningless when `gt` is non-integer (e.g. trial-averaged
  PSTH binned counts — the typical deepSTRF target). `log(gt!)` is
  defined only on non-negative integers.

Users who want the full likelihood for AIC/BIC model comparison on
integer count targets can add the term outside the loss.

#### Pairing with output activations

| Output activation                                | Recommended `log_input` | Why                                              |
|---                                               |---                       |---                                                |
| `nn.Softplus`                                     | `False`                  | Output already `> 0`; rate interpretation natural |
| `ParametricSoftplus` with `non_negative_output=True` (default) | `False`        | Same; default zoo activation (NS1, NAT4, ...)     |
| `ParametricSigmoid` with `non_negative_output=True` | `False`                | Same                                              |
| `ParametricDoubleExponential` with `non_negative_output=True` | `False`        | Same                                              |
| `nn.Identity` / Linear                            | `True`                   | Output unbounded; treat as log-rate               |
| `nn.Sigmoid` (output in [0, 1])                   | `False`                  | Output already `> 0`                              |
| Any with `non_negative_output=False`              | `True`                   | Output may go negative                            |

The default `log_input=False` matches PyTorch's
`torch.nn.PoissonNLLLoss(log_input=False)` and the historical deepSTRF
behaviour.

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
length-weighted across stims). The aggregation convention for the
denominator is chosen to match the natural sample-count weighting that
the numerator's concatenated `cov` and `var` perform automatically — see
§11. Replaces the noise-inflated `var(psth)` in the denominator of the
raw Pearson correlation with the unbiased signal-only variance estimator.
Cells with `SP_n ≤ 0` (the noise-floor regime where the unbiased
estimator fails) return NaN.

#### `method='hsu'`

Hsu, Borst & Theunissen 2004, with the `CCmax` formulation popularized
by Schoppe et al. 2016.

```text
CCnorm_n     = corr(pred_n, psth_n) / CCmax_n
CCmax_{b,n}  = sqrt( 2 · ρ_half_{b,n} / (1 + ρ_half_{b,n}) )           (Spearman–Brown, per stim)
ρ_half_{b,n} = E_{i,j}[ corr( psth_half_{i,b,n}, psth_half_{j,b,n} ) ] (over disjoint half-trial splits)
CCmax_n      = sum_b ( T_b · CCmax_{b,n} ) / sum_b T_b                 (length-weighted, §11)
```

Computed per-stim by sampling up to `ccmax_iters=126` disjoint
half-trial pairs, then length-weighted across stims with `T_b` as the
weight. Same robustness convention as `signal_power` (§6.5). Per-stim
cells with `ρ_half ≤ 0` (worse-than-chance ceiling — too noisy to
estimate) are dropped from the weighted average rather than
contaminating it.

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

The per-neuron quantity is the **length-weighted average** of the
per-stim estimates:

```text
SP_n = sum_b ( T_b · SP_{b,n} ) / sum_b T_b
```

where `T_b` is the number of valid time bins for stim `b`. This is the
BLUE estimator under the assumption that every per-stim `SP_{b,n}` is
unbiased: weighting by sample count gives the minimum-variance combiner.
It also matches the natural sample weighting that `cov` and `var` apply
in the numerator of `corrcoef` / `normalized_corrcoef` (which run over
the full concatenated `(b, t)` series), so a 500 ms clip drives the
denominator the same way it drives the numerator. See §11 for the
derivation.

A given `(b, n)` cell is included iff it has ≥ 2 valid repeats and ≥ 2
valid time bins. Cells without any qualifying stim return NaN under
`reduction='none'`.

### 6.6 `noise_power(responses, mask=None, reduction='mean')`

```text
NP_{b,n} = TP_{b,n} - SP_{b,n}
NP_n     = sum_b ( T_b · NP_{b,n} ) / sum_b T_b
```

Same length-weighted aggregation as `signal_power`.

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
from deepSTRF.metrics import mse_loss, corrcoef, normalized_corrcoef

for batch in loader:                                             # batch is a dict
    stims, responses = batch['stims'], batch['responses']
    pred = model(stims)                                          # (B, N, 1, T)

    loss = mse_loss(pred, responses)                             # auto-PSTH inside
    loss.backward()
    optimizer.step()

    if val_step:
        cc       = corrcoef(pred, responses, reduction='none')    # (N,)  auto-PSTH
        cc_norm  = normalized_corrcoef(pred, responses,           # (N,)  needs raw repeats
                                       method='schoppe',
                                       reduction='none')
```

Notes:

1. **No manual masking and no manual PSTH.** Pass `responses` directly to
   any prediction-vs-PSTH metric; the auto-PSTH collapse (§2) and the
   NaN-derived mask (§4) are both internal. The dataloader's `valid_mask`
   is informational at this point.
2. **Pass `responses.nanmean(dim=2, keepdim=True)` explicitly only when
   you want a non-default target** — a custom objective, a single-trial
   fit, etc. The `(B, N, 1, T)` shape is treated as authoritative and
   passed through unchanged.
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

## 11. Why length-weighting? (BLUE derivation)

Several v1 metrics combine per-stim estimates into a per-neuron number:
`signal_power`, `noise_power`, `snr`, and `normalized_corrcoef(method='hsu')`.
The *aggregation* across stims is length-weighted:

```text
X_n = sum_b ( T_b · X_{b,n} ) / sum_b T_b
```

with `T_b` the number of valid time bins for stim `b`. This subsection
records *why*, since "average across stims" is also a valid choice and
the literature is split.

### Setup

Let `X_{b,n}` be a per-stim estimate of some neuron-level quantity
`X_n` (e.g. signal power). Treat `X_{b,n}` as an unbiased estimator
with variance `Var(X_{b,n}) ∝ 1 / T_b` — i.e. estimator variance shrinks
linearly with the number of time samples used. This is the standard
behaviour of variance-of-variance estimators on stationary signals:
doubling `T` halves the noise on the variance estimate.

The general weighted estimator

```text
X̂_n = sum_b ( w_b · X_{b,n} ) / sum_b w_b
```

is unbiased for any positive `w_b` summing to one. Its variance is
minimised — i.e. it is the **Best Linear Unbiased Estimator (BLUE)** —
when `w_b ∝ 1 / Var(X_{b,n}) = T_b`. So weighting by `T_b` extracts
the most information possible from the per-stim estimates under that
variance assumption.

### Coherence with the numerator

`corrcoef(pred_n, psth_n)` and `var(pred_n)` in the Schoppe formula
operate over the **concatenated** valid `(b, t)` time series for neuron
`n`. A 500 ms clip contributes 10× the samples of a 50 ms clip and
correspondingly drives the numerator's `cov` / `var` 10× harder. If we
were to combine per-stim SPs with equal weights instead of length
weights, the numerator and denominator would weight stims differently,
which makes the resulting CCnorm value awkward to interpret —
especially when comparing models on heterogeneous-length validation
sets. Length-weighting on the denominator brings everything into
agreement.

### The equal-weight alternative

`mean_b X_{b,n}` is also defensible:

- Pros: each stim contributes one *answer*, not one *fact*. If you
  trust short-stim estimates as much as long-stim estimates (e.g. your
  short stims are repeated many more times so the per-stim variance
  gain compensates), equal weighting reflects that.
- Cons: ignores the variance structure described above; short noisy
  stims can pull the average around.

We pick length-weighting as the default because it composes correctly
with the natural sample-count weighting elsewhere in the pipeline. A
future kwarg `weighting='length' | 'equal'` could expose the choice if a
clear use case emerges.

### Practical consequence

A neuron with two clips — a 500 ms validation stim with `SP=10` and a
50 ms validation stim with `SP=1` — gets `SP_n ≈ (500·10 + 50·1)/550 ≈
9.18` under length-weighting, vs `5.5` under equal weighting. The
length-weighted answer matches what `var(psth_n)` over the concatenated
time series would have given (modulo the `1/(R−1)` Bessel correction);
the equal-weighted answer would produce an internally inconsistent
CCnorm.

## 12. References

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
