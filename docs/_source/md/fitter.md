# The deepSTRF Fitter

This note documents `deepSTRF.training.Fitter`, an opt-in convenience
wrapper around the canonical training loop spelled out in
[`metrics_paradigm.md`](metrics_paradigm.md) §7. It is the fourth leg of
the data/model/metrics/training contract, alongside
[`data_paradigm.md`](data_paradigm.md),
[`model_paradigm.md`](model_paradigm.md), and
[`metrics_paradigm.md`](metrics_paradigm.md).

## 1. Scope: a thin, opt-in PyTorch loop

`Fitter` is a ~150-line class that wires a `NeuralModel`, a
`DataLoader`, an `Optimizer`, and the `deepSTRF.metrics` API into a
standard fit-evaluate-early-stop training loop. It is **pure PyTorch**:
no Lightning, no Accelerate, no `huggingface/transformers`-style
abstractions. Single-GPU first; distributed training is a v2 follow-up.

The Fitter is **opt-in**, not the canonical training path. The
canonical path is the three-line loop documented in
`metrics_paradigm.md` §7:

```python
for stims, responses, valid_mask, stim_metas in loader:
    pred = model(stims)                                    # (B, N, 1, T)
    loss = mse_loss(pred, responses)                       # auto-PSTH inside
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

If you want a custom learning-rate schedule, gradient accumulation,
mixed-precision, distributed training, a curriculum, or any other
non-standard loop shape — **write the loop**. The deepSTRF metrics
already do the heavy lifting (auto-PSTH collapse, NaN-aware reductions,
length-weighted aggregation), so the loop body stays small. The Fitter
exists to spare casual users the boilerplate of early stopping,
checkpoint selection, and per-epoch metric accumulation. It is not the
only sanctioned way to train a deepSTRF model.

## 2. When NOT to use the Fitter

| Need                                                | Use the Fitter? | Alternative                                  |
|---                                                   |---              |---                                            |
| Fit one model, log val metrics, save best checkpoint | yes             | —                                             |
| Quick sanity check on a notebook                     | yes             | —                                             |
| Multi-seed sweep with averaged metrics               | yes — use `fit_multi_seed` (§4.2)                       | —                            |
| Mixed-precision (`autocast` + `GradScaler`)          | no              | Custom loop, or a v2 `MixedPrecisionFitter`   |
| Multi-GPU / multi-node                                | no              | `accelerate` + custom loop (v2)              |
| Curriculum / dynamic dataloader                      | no              | Custom loop                                   |
| Hyperparameter search with many models per process   | no              | Custom loop; instantiate models cheaply       |
| Adversarial / GAN-style multi-step                    | no              | Custom loop                                   |
| Reinforcement learning–style outer loop              | no              | Custom loop                                   |

The rule of thumb: if your training story doesn't fit "one model, one
optimizer, one train loader, one val loader, one loss, one early-stop
patience, save best-on-val," write the loop.

## 3. Public API

```python
from deepSTRF.training import Fitter

fitter = Fitter(
    model,                                # NeuralModel
    train_loader,                          # DataLoader
    val_loader,                            # DataLoader, batch_size=1 recommended
    *,
    loss_fn       = mse_loss,             # callable(pred, gt_psth) -> scalar
    val_metrics   = None,                  # dict[name, callable] | None (None = canonical pair)
    optimizer     = None,                  # Optimizer | None (None = AdamW(lr=1e-3, wd=1e-4))
    device        = 'cpu',                 # str | torch.device — 'cpu' | 'cuda' | 'mps'
    max_epochs    = 1000,                  # int
    patience      = 10,                    # int — early-stop patience
    monitor       = 'val_cc_norm',         # str — key in epoch dict to early-stop on
    mode          = 'max',                  # 'max' | 'min' — direction for `monitor`
    ckpt_path     = None,                  # str | Path | None — save best on `monitor`
    log_fn        = print,                 # callable(epoch_dict) -> None
)
history = fitter.fit()                     # list[dict] — one entry per epoch
test_metrics = fitter.evaluate(test_loader)  # dict[str, Tensor]
```

Every constructor argument has a sensible default; the minimum
invocation is `Fitter(model, train_loader, val_loader)`.

```{note}
**Device support.** deepSTRF is device-agnostic: pass `device='cpu'`,
`'cuda'`, or `'mps'` (Apple silicon). Models follow the device of their
parameters, and the canonical training loop moves each batch with
`.to(device)`, so the standard audio models (Linear, ConvNet2D,
Transformer, StateNet-GRU, DNet, NRF) run on all three. The state-space
models (S4, LMU, Mamba) rely on complex-valued tensors and custom kernels
that the MPS backend does not yet support — run those on CPU or CUDA.
```

## 4. The default training loop

The Fitter implements:

```text
best_score = -inf if mode == 'max' else +inf
better    = (lambda new, best: new > best) if mode == 'max' else (lambda new, best: new < best)

for epoch in range(max_epochs):
    train_metrics = self._train_one_epoch()      # backprop + per-batch loss
    val_metrics   = self._evaluate(val_loader)   # cross-batch metric concat (§6)
    epoch_dict    = {**train_metrics, **val_metrics}
    self.on_epoch_end(epoch, epoch_dict)

    score = _to_scalar(epoch_dict[self.monitor])  # nanmean if per-neuron tensor
    if better(score, best_score):
        best_score = score
        if self.ckpt_path: torch.save(self.model.state_dict(), self.ckpt_path)
        epochs_no_improvement = 0
    else:
        epochs_no_improvement += 1

    if epochs_no_improvement >= self.patience:
        break

if self.ckpt_path:
    self.model.load_state_dict(torch.load(self.ckpt_path))   # restore best
```

The training step inside `_train_one_epoch` is the four-line canonical
loop. The validation step inside `_evaluate` is the cross-batch
concat-then-compute pattern documented in §6.

### 4.1 Per-cell snapshot restoration (`track_per_cell_best=True`)

Optional opt-in: track each cell's best-on-`monitor` epoch independently
and overlay its individual-best parameter slice on top of the global
checkpoint restore at end-of-fit. The training trajectory is identical
to a vanilla Fitter run — no gradient masking, no per-cell stopping,
same termination epoch. Only difference: which checkpoint is loaded at
the end of `fit()`.

```python
fitter = Fitter(
    model, train_loader, val_loader,
    monitor='val_cc_norm', patience=50,
    track_per_cell_best=True,
)
fitter.fit()
```

**How it works.** At every val evaluation, for each cell $n$ whose
monitor score improved against its own running best, the Fitter snapshots
`model.readout`'s per-$N$ parameter and buffer slices for that cell.
Snapshots are continuously overwritten as long as the cell keeps
improving, then preserved after the last improvement. At end-of-fit,
after the existing `ckpt_path` restore (which resets every parameter to
its global-best state), each cell's snapshot is overlaid on top.

**Strict guarantee on no-shared-params models.** When every learnable
scalar under `model.readout` has $N$ as leading axis (the post-2026-05-19
audio convention — Linear / LinearNonlinear with the per-neuron BN
inside `STRFReadout`), the post-hoc per-cell restored state is at least
as good as the vanilla restored state **per cell on the validation set,
by construction**. Each cell ends up at its individual val peak, which
is ≥ its score at the population peak.

**Empirical (Espejo NAT, animal=AMT, 168 cells, Linear, patience=50):**
mean test cc_norm goes from +0.378 (vanilla) to +0.402 (post-hoc per-cell
restore), +0.024 absolute. Val gain transfers cleanly when the val set
is large enough (~85 stims here). On small val sets (≤5 stims) the
per-cell val peaks are noise-dominated and the val→test transfer
breaks; mean test movement was within ±0.005 on NS1 (3 val stims).

**Caveat on shared-core models.** The bidirectional `ckpt_path` ↔
per-cell restore order means non-readout parameters (e.g. a shared core
or input normalization) end up at their global-best state while each
cell's readout slice ends up at its individual-best epoch. If those
epochs differ, the cell's readout was trained against a different core
state than what it sees at eval — an inconsistency that empirically
shows up as a small (~2/168) negative-Δ tail on val. The strict
guarantee only holds when there is nothing to be inconsistent with.
Models built before the 2026-05-19 BN refactor that still have a
non-trivial `core` should treat `track_per_cell_best` as best-effort,
not strict.

**Requirements.** The `val_metrics[monitor.removeprefix('val_')]`
callable must return a `(N,)` per-cell tensor; the default
`_default_val_metrics` does this. With a scalar monitor, the flag
raises `ValueError` at the first eval.

### 4.2 Multi-seed init-variance sweeps (`fit_multi_seed`)

A second opt-in convenience: `deepSTRF.training.fit_multi_seed` runs the
same Fitter configuration K times under different seeds and aggregates
per-neuron val and test metrics across seeds.

```python
from deepSTRF.training import fit_multi_seed

results = fit_multi_seed(
    model_factory  = lambda seed: Linear(n_frequency_bands=34,
                                          temporal_window_size=9,
                                          out_neurons=N),
    loader_factory = lambda seed: (train_loader_fn(seed),
                                    val_loader_fn(seed),
                                    test_loader_fn(seed)),
    n_seeds        = 5,
    fitter_kwargs  = {"patience": 50, "monitor": "val_cc_norm", "mode": "max",
                       "track_per_cell_best": True},
    logger_factory = None,                    # see §4.3 for the optional logger hook
    output_dir     = "runs/ns1-linear",       # auto-save JSON + best.pt per seed (§4.4)
)
```

**Scope.** This is multi-seed *initialization* variance: same data
split, same hyperparameters, different initial weights and shuffle
order. It is **not** k-fold or leave-one-stim-out cross-validation —
those need a split-factory rather than a seed sweep and are separate
roadmap items.

**Contract.**

- `model_factory(seed) -> nn.Module` is called *after*
  `set_random_seed(seed)`, so the model's weight init draws from the
  seeded RNG.
- `loader_factory(seed) -> (train, val, test)` is a required 3-tuple.
  Fresh loaders per seed let each run get a deterministic shuffle
  generator if the user wires one in the `DataLoader(..., generator=...)`
  slot.
- `fitter_kwargs` is forwarded verbatim to `Fitter(...)`. The three
  managed kwargs (`model`, `train_loader`, `val_loader`) raise if
  present. `ckpt_path` is auto-suffixed with `_seed{seed}` so seeds
  don't overwrite each other's checkpoints.

**Returns.** `results` is a flat dict (regardless of `output_dir`):

| Key                            | Shape / type                  | Notes                                            |
|---                             |---                             |---                                              |
| `seeds`                         | `list[int]`                   | `[0, 1, ..., n_seeds-1]` by default              |
| `per_seed_val_<m>`              | `(n_seeds, N)`                | per-neuron val metric, one row per seed          |
| `mean_val_<m>`                  | `(N,)`                        | `nanmean` over seeds                              |
| `std_val_<m>`                   | `(N,)`                        | population nanstd over seeds                      |
| `per_seed_test_<m>`             | `(n_seeds, N)`                | same triple for the test loader                  |
| `mean_test_<m>`                 | `(N,)`                        |                                                  |
| `std_test_<m>`                  | `(N,)`                        |                                                  |
| `per_seed_val_loss`             | `(n_seeds, 1)`                | scalar metrics become length-1 along axis 1      |
| `per_seed_histories`            | `list[list[dict]]`            | raw per-seed Fitter histories                    |
| `best_seed`                     | `int`                         | argmax (or argmin if `mode='min'`) of mean val monitor |
| `best_state_dict`               | `OrderedDict[str, Tensor]`    | deep copy of the best seed's post-fit state      |

### 4.3 Logger hook (`logger_factory`)

`fit_multi_seed` does not import any dashboard. The optional
`logger_factory` arg takes a `Callable[[int], SeedLogger]` — anything
duck-typed to the following protocol:

```python
class SeedLogger:
    def __call__(self, epoch_dict: Mapping[str, Any]) -> None: ...
    def finalize(self, final_metrics: Mapping[str, Any]) -> None: ...   # optional
    def close(self) -> None: ...                                          # optional
```

`__call__` is invoked once per epoch with the epoch dict (same shape
the Fitter's `log_fn` receives). `finalize` is invoked at end of seed
with `{'val': val_post, 'test': test_post}` — the post-fit re-evaluated
metrics on both loaders. `close` is invoked last. Both extras are
optional (`hasattr` is checked); a logger can be just a callable.

WandB users get a reference implementation in
[`deepSTRF.training.wandb_log`](../../deepSTRF/training/wandb_log.py):

```python
from deepSTRF.training import fit_multi_seed
from deepSTRF.training.wandb_log import make_wandb_logger_factory

fit_multi_seed(
    model_factory=..., loader_factory=..., n_seeds=3,
    logger_factory=make_wandb_logger_factory(
        project="deepstrf", entity="urancon",
        group="ns1-linear", mode="offline",   # offline by default; "disabled" no-ops
    ),
    fitter_kwargs={...},
)
```

The `WandbSeedLogger` per-epoch hook reduces every per-neuron tensor
(e.g. `val_cc_norm: (N,)`) to a `{mean, p10, p50, p90}` scalar quadruple
plus a `wandb.Histogram` of the cell distribution — so the dashboard
shows population-level percentile lines instead of a single average,
and the per-epoch histogram panel surfaces tails. `finalize` pushes
test metrics (`test_cc`, `test_cc_norm`, `test_loss`) plus their
percentile scalars into `run.summary`, where they become sortable
columns in the run table.

Run names auto-derive: `<name>-seed{i}` if `name=` is given,
`<group>-seed{i}` if `group=` is given, else `seed{i}`. The seed value
is added to `wandb.config` so dashboards can colour-code by seed.

For MLflow / TensorBoard / a local CSV writer, implement the
`SeedLogger` protocol in your own module — `fit_multi_seed` does not
care which dashboard, only that the protocol is met.

### 4.4 Auto-save (`output_dir`)

Independent of any dashboard, set `output_dir=` to materialise every
run's metrics and best state on disk. The library treats this as part
of its job: a 30-minute training shouldn't lose all its metrics just
because the user forgot to pickle the returned dict.

```
output_dir/
├── seed0/
│   ├── history.json            # per-epoch dict; per-neuron tensors are
│   │                            # summarised to {mean, p10/p50/p90, n_valid}
│   ├── final.json              # post-fit val + test population summaries
│   ├── final_neurons.pt        # per-neuron (N,) tensors for downstream stats
│   └── best.pt                  # state_dict (deep copy of post-fit model)
├── seed1/...
├── seed2/...
├── summary.json                 # across-seed mean/std + best_seed + monitor
└── summary_neurons.pt           # per-neuron mean / std / per-seed tensors
```

`history.json` and `final.json` are human-readable / machine-greppable.
`.pt` files hold the full per-neuron tensors as a `torch.save` dict;
load with `torch.load(path, weights_only=False)` and read keys like
`saved['val']['cc_norm']` (shape `(N,)`). The
`output_dir/summary_neurons.pt` mirrors the in-memory `results` dict
for the across-seed aggregates.

## 5. Hooks for customization

Two equivalent ways to override defaults: pass a callable as a kwarg, or
subclass and override the corresponding method. Three hook points:

### 5.1 `loss_fn(pred, responses) -> Tensor`

Default: `mse_loss`. Pass a different callable to use Poisson, weighted
MSE, etc. The Fitter wraps it as:

```python
def compute_loss(self, pred, responses):
    return self.loss_fn(pred, responses)
```

The default `mse_loss` (and the other deepSTRF losses) auto-collapse
`responses` to PSTH via `metrics_paradigm.md` §2. Subclass and override
`compute_loss` if your loss needs the `valid_mask`, the `stim_metas`,
or any other per-batch quantity.

**No `log_input` autodetection.** The user picks the loss; pairing with
the model's output activation is their responsibility (see
`metrics_paradigm.md` §6.2 for the table). For Poisson-with-log-rate:

```python
from functools import partial
from deepSTRF.metrics import poisson_loss

fitter = Fitter(model, ..., loss_fn=partial(poisson_loss, log_input=True))
```

### 5.2 `val_metrics: dict[name, callable]`

Default:

```python
{
    'cc':      lambda pred, responses: corrcoef(pred, responses, reduction='none'),
    'cc_norm': lambda pred, responses: normalized_corrcoef(pred, responses,
                                                            method='schoppe',
                                                            reduction='none'),
}
```

Each callable receives `(pred, responses)`. Prediction-vs-PSTH metrics
(`corrcoef`, `fve`, `mse_loss`, ...) auto-collapse internally
(`metrics_paradigm.md` §2). Repeat-aware metrics (`normalized_corrcoef`,
`signal_power`, ...) take `responses` directly. The Fitter calls each
callable once per epoch on the cross-batch concatenated tensors (§6).

The metric is stored under the key `f'val_{name}'` in the epoch dict
(e.g. `'cc'` → `'val_cc'`). For early-stopping and `log_fn`, per-neuron
tensors are reduced to a scalar via `nanmean`; users who only want the
mean can short-circuit by writing `reduction='mean'` in their lambda
(returns a scalar directly, the reduction step becomes a no-op).

User overrides:

```python
from deepSTRF.metrics import fve

fitter = Fitter(model, ..., val_metrics={
    'cc':  lambda p, r: corrcoef(p, r, reduction='none'),
    'fve': lambda p, r: fve(p, r, reduction='none'),
})
```

`val_loss` is always computed and added to the dict automatically; do
not include it in `val_metrics`.

### 5.3 `on_epoch_end(epoch, epoch_dict) -> None`

Default: `self.log_fn(epoch_dict)` (i.e. `print` by default). Override
to log to WandB, write to a file, push to a dashboard, etc.

```python
import wandb

class WandbFitter(Fitter):
    def on_epoch_end(self, epoch, epoch_dict):
        wandb.log({k: v.nanmean().item() if torch.is_tensor(v) else v
                   for k, v in epoch_dict.items()}, step=epoch)
```

WandB is **not** a dependency. The Fitter ships zero logger
integrations; users wire whatever they want via this hook. A 5-line
WandB example will be included in the example notebooks.

## 6. Cross-batch metric accumulation

This is the one place where the Fitter does something non-trivial.

`metrics_paradigm.md` §3 says every metric collapses `(B, T)` into one
long pseudo-time series per neuron. Per-batch metrics computed
independently and then averaged are **biased** — most pertinently for
`normalized_corrcoef`, whose `signal_power` denominator needs many time
samples to stabilize. The legacy `utils/training.py` solved this by
concatenating predictions, responses, and PSTH across all val batches,
then calling the metric once at end-of-epoch. We preserve that pattern.

The Fitter accumulates per-batch tensors in a list, then at end-of-epoch:

```python
preds_cat     = _pad_and_cat(preds,     dim_time=-1, dim_batch=0)   # (B_total, N, 1, T_max)
responses_cat = _pad_and_cat(responses, dim_time=-1, dim_batch=0)   # (B_total, N, R_max, T_max)
for name, fn in self.val_metrics.items():
    out[f'val_{name}'] = fn(preds_cat, responses_cat)               # (N,)
```

`_pad_and_cat` right-pads each batch's time axis (and the responses
tensor's R axis) to the global max with NaN, then concatenates along
the batch axis. The padding is dropped by every metric's NaN-derived
mask (`metrics_paradigm.md` §4). Variable batch sizes are fine.

**Memory caveat.** This holds the entire val set in GPU memory before
calling the metric. For typical encoding-model val sets (hundreds of
seconds × few hundred neurons × few repeats) this is well under a
gigabyte. For very large val sets, override `_evaluate` and stream.

**Why not torchmetrics-style streaming state?** Out of scope for v1.
A streaming `signal_power` requires running sums of `var(psth)` and
`E[var(y_r)]` per neuron per stim, which doubles the API surface and
introduces a new failure mode (state forgotten between calls). The
concat-then-compute path matches the metrics paradigm's flat
pseudo-time-series semantics exactly and is what the legacy code
already did correctly. Streaming is a clean v2 add when a user hits
the memory ceiling.

**Train-set metrics.** The Fitter computes train-set CC and CCnorm with
the same concat-then-compute pattern. Per-batch train *loss* is
accumulated as a running mean (the standard "loss curve" quantity),
since loss is gradient-bearing and per-batch is the natural unit.

## 7. Reproducibility

`deepSTRF.training.set_random_seed(seed, *, strict=False)` seeds Python
`random`, NumPy, PyTorch CPU, and CUDA. With `strict=True` it also sets
`torch.use_deterministic_algorithms(True)` and
`torch.backends.cudnn.deterministic = True` — useful for debugging,
but disables non-deterministic CUDA kernels (notably some convolutions),
trading speed for bit-exactness across runs.

This is migrated from `deepSTRF.utils.training.set_random_seed` (which
becomes a deprecated re-export until the next major release).

The Fitter does **not** call `set_random_seed` itself. The user seeds
once before instantiating the Fitter; multi-seed sweeps loop over seeds
and rebuild the model + Fitter in each iteration:

```python
for seed in [0, 1, 2, 3, 4]:
    set_random_seed(seed)
    model = build_model()
    Fitter(model, train_loader, val_loader,
           ckpt_path=f'best_seed{seed}.pt').fit()
```

For the K-seed-then-aggregate case, use `fit_multi_seed` (§4.2) — it
calls `set_random_seed(seed)` internally before each
`model_factory(seed)` invocation, auto-suffixes `ckpt_path`, and
aggregates per-neuron val + test metrics into `(n_seeds, N)` tensors.

## 8. What is NOT in v1

These were considered and explicitly cut. Restoration will re-open this
doc.

- **Multi-GPU / DDP.** Out of scope. v2 candidate built on `accelerate`.
- **Mixed-precision.** Same.
- **LR schedulers.** Pass an optimizer with the scheduler attached;
  the Fitter does not call `scheduler.step()` itself in v1. v1.1
  candidate.
- **Gradient accumulation.** Custom loop.
- **Logger integrations** (WandB, MLflow, TensorBoard). One-line override
  via `on_epoch_end`.
- **Module-API metrics** (torchmetrics-style stateful `update()` /
  `compute()`). Same reasoning as in `metrics_paradigm.md` §10:
  functional API is enough for v1.
- **Streaming val accumulation** (§6). Cleaner once a real memory
  bottleneck appears.
- **Callbacks framework** (Keras-style). The three hook points (§5)
  cover the 95% case without a `Callback` base class.
- **Profiling integration** (`torch.profiler`). User decorates `fit()`
  themselves.

## 9. Invariants for Fitter authors

Anyone changing `deepSTRF.training.Fitter` must respect:

1. **The default training step is verbatim the three-line canonical loop**
   from `metrics_paradigm.md` §7. Any deviation is a bug.
2. **Default `val_metrics` are `corrcoef` and `normalized_corrcoef`
   (Schoppe), and the default early-stop monitor is `val_cc_norm`
   (`mode='max'`).** These are the canonical val metrics per
   `metrics_paradigm.md` §6.3 / §6.4.
3. **Cross-batch accumulation uses concat-then-compute, not per-batch
   averaging.** Per-batch CCnorm is biased.
4. **No autodetection of `log_input`, no autodetection of activation
   pairing, no autodetection of anything.** The user picks the loss;
   the Fitter does not outsmart them.
5. **Zero hard dependencies beyond what `deepSTRF` already requires.**
   No WandB, no Lightning, no `accelerate`, no `tqdm` even (use a
   `log_fn` if you want a progress bar).
6. **The `model.detach()` no-op base method is called after every
   training and validation step**, to support stateful models
   (StateNet, future RNN/SSM cores) without special-casing them.
7. **Tests under `tests/test_fitter.py`** assert: (a) one-epoch fit on
   a tiny synthetic dataset reduces loss, (b) early stopping fires
   when the monitor metric plateaus, (c) checkpoint round-trip
   preserves parameters, (d) hook overrides (loss, val_metrics,
   on_epoch_end) fire and propagate, (e) `set_random_seed` produces
   bit-identical training runs, (f) `monitor='val_loss', mode='min'`
   and `monitor='val_cc_norm', mode='max'` both work end-to-end.

## 10. Migration from `deepSTRF.utils.training`

The legacy module exposes `optimize_multiple_seeds`,
`optimize_one_seed`, `train_one_epoch`, `evaluate`, and
`set_random_seed`. All five are obsolete:

| Legacy                       | Replacement                                                         |
|---                           |---                                                                  |
| `optimize_multiple_seeds`    | User loop over seeds + one `Fitter` per seed (§7)                   |
| `optimize_one_seed`          | `Fitter.fit()`                                                      |
| `train_one_epoch`            | `Fitter._train_one_epoch` (private; subclass to override)           |
| `evaluate`                   | `Fitter.evaluate(loader)`                                           |
| `set_random_seed`            | `deepSTRF.training.set_random_seed` (kept as deprecated re-export)  |

Behavioral differences worth flagging:
- The legacy code unpacks `(spectrogram, responses, ccmax, ttrc)`. The
  new code unpacks `(stims, responses, valid_mask, stim_metas)` from
  `neural_collate`. `ccmax` and `ttrc` are no longer dataloader-side
  pre-computed tensors — they are computed on demand by
  `normalized_corrcoef` from raw `responses`.
- The legacy code calls `prediction.squeeze(-2)` to drop the R-axis
  before metrics. The new metrics expect `(B, N, 1, T)` per the model
  paradigm — no squeeze.
- The legacy code uses `responses.mean(dim=-2)` for the PSTH (NaN-
  contaminating). The new code uses `responses.nanmean(dim=2,
  keepdim=True)`.
- The legacy code uses `correlation_coefficient` /
  `normalized_correlation_coefficient`. These remain as deprecated
  aliases in `deepSTRF.metrics` until this branch lands; once the
  Fitter replaces every legacy caller, the aliases are removed (per
  `metrics_paradigm.md` "backward-compat aliases" note).

The legacy `utils/training.py`, `utils/training_pop.py`, and the two
untracked `utils/training_parallel*.py` files are deleted in this
branch. The `scripts/example_fit_*.py` scripts that imported them are
either rewritten as example notebooks (`fit_audio_*` notebooks) or
retired.

## 11. References

- `metrics_paradigm.md` §7 — the canonical three-line training step.
- `metrics_paradigm.md` §2 (auto-PSTH collapse) — why `loss_fn(pred,
  responses)` works without caller-side `nanmean`.
- `metrics_paradigm.md` §6.2, §6.3, §6.4 — loss/metric pairing rules.
- `data_paradigm.md` §6 — NaN-aware boolean indexing in the training
  loop.
- `model_paradigm.md` §3 — the `(B, N, R=1, T)` model output contract
  and `model.detach()` hook.
