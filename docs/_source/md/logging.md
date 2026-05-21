# Logging your training runs

This page is a hands-on guide to wiring **WandB** or **TensorBoard** into
a deepSTRF training run, including the multi-seed wrapper. It covers the
common cases first and the configuration / aggregation niceties second.

Reading this in order will take ~10 minutes; copy-pasting the snippets
should give you a working dashboard at the end.

If you only want one sentence: pass a `log_fn` to `Fitter`, or a
`logger_factory` to `fit_multi_seed`, with the helpers in
`deepSTRF.training.wandb_log` / `deepSTRF.training.tb_log`. The rest of
this page is the long version.

## 1. The logging hook

`Fitter` calls one user-supplied callable once per epoch:

```python
fitter = Fitter(model, train_loader, val_loader, log_fn=my_log_fn)
```

`my_log_fn(epoch_dict)` receives a dict like
`{"epoch": 0, "train_loss": 0.5, "val_loss": 0.4, "val_cc_norm": tensor([...]), ...}`.
The default is a one-line print. Replace it with anything — a CSV writer,
a Slack webhook, a wandb call.

That's the whole extension surface. Loggers in deepSTRF are just
prebuilt `log_fn` factories.

## 2. WandB in 5 lines

WandB is the recommended cloud dashboard for cross-run comparison. It's
already in deepSTRF's hard dependencies — no extra `pip install`.

```python
from deepSTRF.training import Fitter
from deepSTRF.training.wandb_log import WandbSeedLogger

logger = WandbSeedLogger(
    seed=0, project="deepstrf", entity="your-username",
    group="linear-ns1-T9-F34", mode="offline",       # see §2.1 for "offline"
)

fitter = Fitter(model, train_loader, val_loader, log_fn=logger)
fitter.fit()
logger.finalize({"val": fitter.evaluate(val_loader),
                  "test": fitter.evaluate(test_loader)})
logger.close()
```

You'll get:
- A run named `linear-ns1-T9-F34-seed0` under your wandb project.
- Per-epoch `train_loss`, `val_loss`, `val_cc`, `val_cc_norm` as time series.
- Per-neuron metrics summarised to `{name}` (population mean) plus
  `{name}/p10`, `/p50`, `/p90` percentile scalars and a
  `{name}/hist` per-epoch histogram of the cell-by-cell distribution.
- `test_cc`, `test_cc_norm`, `test_loss` (and their percentiles) written
  to `run.summary` — sortable columns in the project run table.

### 2.1 Offline by default

`WandbSeedLogger` sets `WANDB_MODE=offline` if the env var is unset, so
the first run works with no account and no network. WandB writes
everything under `./wandb/offline-run-<timestamp>-<id>/` and prints a
sync command at the end:

```
wandb sync --entity your-username wandb/offline-run-*
```

That uploads the local files to your wandb cloud account whenever you
want. To go live from the start, pass `mode="online"` to the logger.

### 2.2 Browsing without uploading

If you don't want a cloud account at all, look at the local files:

```bash
ls wandb/offline-run-*-<id>/files/   # config.yaml, requirements.txt, output.log
```

The training history lives in the binary `run-<id>.wandb` protobuf; the
`output_dir` JSON tree (§6) is the friendlier path for "just give me the
numbers."

## 3. TensorBoard

If you prefer a local-only viewer with no account, TensorBoard ships in
deepSTRF's dependencies too. The setup is symmetric:

```python
from deepSTRF.training.tb_log import TensorBoardSeedLogger

logger = TensorBoardSeedLogger(
    seed=0, log_dir="tb_logs", group="linear-ns1-T9-F34",
)
fitter = Fitter(model, train_loader, val_loader, log_fn=logger)
fitter.fit()
logger.finalize({"val": fitter.evaluate(val_loader),
                  "test": fitter.evaluate(test_loader)})
logger.close()
```

Then browse:

```bash
tensorboard --logdir tb_logs
# opens at http://localhost:6006
```

You'll see one event tree per seed under
`tb_logs/linear-ns1-T9-F34/linear-ns1-T9-F34-seed0/`. The **Scalars**
tab has the per-epoch traces, the **Histograms** tab has the per-neuron
distributions over epochs, and the **Hparams** tab has the config in a
sortable table.

`TensorBoardSeedLogger` writes the post-fit val/test metrics under a
`final/` prefix at step 0, since TB has no separate run-summary concept.
Look for `final/test_cc_norm` in the Scalars tab.

## 4. Multi-seed sweeps

For "train K seeds, aggregate the metrics across them," use
`fit_multi_seed` with a `logger_factory`:

```python
from deepSTRF.training import fit_multi_seed
from deepSTRF.training.wandb_log import make_wandb_logger_factory

results = fit_multi_seed(
    model_factory  = lambda seed: Linear(n_frequency_bands=34,
                                          temporal_window_size=9,
                                          out_neurons=N),
    loader_factory = lambda seed: (train_loader, val_loader, test_loader),
    n_seeds        = 3,
    fitter_kwargs  = {"patience": 10, "monitor": "val_cc_norm",
                       "track_per_cell_best": True, "track_train_metrics": True},
    logger_factory = make_wandb_logger_factory(
        project="deepstrf", entity="your-username",
        group="linear-ns1-T9-F34", mode="offline",
    ),
)
```

`make_wandb_logger_factory(...)` returns a `Callable[[int], WandbSeedLogger]`.
`fit_multi_seed` invokes it once per seed, with each seed getting a
fresh wandb run (`-seed0`, `-seed1`, `-seed2`). `finalize` and `close`
fire automatically at the end of each seed.

For TensorBoard:

```python
from deepSTRF.training.tb_log import make_tensorboard_logger_factory

logger_factory = make_tensorboard_logger_factory(
    log_dir="tb_logs", group="linear-ns1-T9-F34",
)
```

## 5. Config & group naming

The convention that makes a wandb / TB URL self-documenting is to
encode the experiment's key hparams in the group name. `auto_config`
extracts a JSON-friendly dict from a model + fitter_kwargs;
`slug_from_config` derives a short group name from it:

```python
from deepSTRF.training import auto_config, slug_from_config

config = auto_config(
    model           = your_model_instance,
    fitter_kwargs   = {"patience": 10, "max_epochs": 30, "monitor": "val_cc_norm"},
    dataset_name    = "NS1",
)
# -> {"model": "Linear", "dataset": "NS1", "n_frequency_bands": 34,
#     "temporal_window_size": 9, "out_neurons": 119, "prefiltering": "Identity",
#     "core": "Identity", "readout": "STRFReadout", ...}

group = slug_from_config(config)              # -> "linear-ns1-T9-F34"

logger_factory = make_wandb_logger_factory(
    project="deepstrf", entity="your-username",
    group=group, config=config,                # << config lands in wandb.config
    mode="offline",
)
```

`config=` is forwarded to `wandb.init`, so every field becomes a
sortable/filterable column in the wandb run table. A second sweep with
`temporal_window_size=15` lands under group `linear-ns1-T15-F34`,
distinct from the T=9 sweep — side-by-side comparison in the wandb
project page just works.

The TB logger accepts the same `config=` kwarg and surfaces each entry
as an `hparams/<key>` scalar plus a markdown `config` text panel.

## 6. Logger-agnostic auto-save (`output_dir`)

Whether you wire wandb, TB, or nothing, `fit_multi_seed` can save the
full metrics tree to disk:

```python
results = fit_multi_seed(
    ...,
    output_dir="runs/linear-ns1-T9",
)
```

Layout:

```
runs/linear-ns1-T9/
├── seed0/
│   ├── history.json           # per-epoch dict; per-neuron metrics are
│   │                           # summarised to {mean, p10, p50, p90}
│   ├── final.json             # post-fit val + test summaries
│   ├── final_neurons.pt       # per-neuron (N,) tensors for downstream stats
│   └── best.pt                 # state_dict (deep copy of post-fit model)
├── seed1/...
├── seed2/...
├── summary.json                # across-seed mean/std + best_seed + monitor
└── summary_neurons.pt          # per-neuron mean/std/per-seed tensors
```

This is the ground truth — every value any dashboard could display is
also here, in plain JSON. Particularly useful for paper figures:

```python
import json, torch

summary = json.loads((Path("runs/linear-ns1-T9") / "summary.json").read_text())
print("mean test cc_norm :", summary["mean_test_cc_norm"]["mean"])
print("std  test cc_norm :", summary["std_test_cc_norm"]["mean"])

per_neuron = torch.load("runs/linear-ns1-T9/summary_neurons.pt",
                         weights_only=False)
per_neuron["mean_test_cc_norm"]      # (N,) — exact value per cell
per_neuron["std_test_cc_norm"]       # (N,) — across-seed std per cell
```

## 7. Where to find mean / std across seeds

For per-neuron tensors and exact numbers, prefer `summary.json` /
`summary_neurons.pt` from §6. They're authoritative and dashboard-free.

### In the WandB UI

1. Open your project page (e.g. `https://wandb.ai/<entity>/<project>`).
2. In the run table, click the **Group** button in the toolbar.
3. Set "Group runs by" → `Group` (the field literally named "Group" —
   holds the value you passed as `group="..."`).
4. The 3 per-seed rows collapse to one. Summary metric columns
   (`test_cc_norm`, `test_loss`, ...) now show mean ± a min/max swatch.
5. Click the column header → toggle "stddev" if you want it numeric.

For chart panels (line plots over epochs), edit the chart → Grouping
pane → "Group by" = `Group`, aggregate = `mean`, range = `stddev`. The
3 traces merge into a single mean-with-band line. This works for
time-series metrics (val_cc_norm over epochs) but **not** for
`test_cc_norm`, which is a single scalar in `run.summary` — no time
axis to plot.

### In the TensorBoard UI

TensorBoard does not aggregate across runs natively. The Scalars tab
overlays multiple selected runs but won't compute a mean trace; the
Hparams tab is a sortable table but not aggregated.

So for std across seeds in TB, read `summary.json` (or use the in-memory
`results["std_test_cc_norm"]` if you still have the Python session).
TB's strength is per-seed inspection — trajectories, histograms, hparam
filters; aggregation lives on disk.

## 8. Using two loggers at once

`fit_multi_seed` accepts one `logger_factory`. To send to wandb **and**
TensorBoard from a single fit, write a 12-line fan-out adapter:

```python
class FanOutSeedLogger:
    """Broadcast __call__ / finalize / close to multiple inner loggers."""
    def __init__(self, *loggers):
        self._loggers = loggers
    def __call__(self, epoch_dict):
        for lg in self._loggers: lg(epoch_dict)
    def finalize(self, final_metrics):
        for lg in self._loggers:
            if hasattr(lg, "finalize"): lg.finalize(final_metrics)
    def close(self):
        for lg in self._loggers:
            if hasattr(lg, "close"): lg.close()

wandb_f = make_wandb_logger_factory(project="deepstrf", entity="me",
                                      group=group, config=config, mode="offline")
tb_f    = make_tensorboard_logger_factory(log_dir="tb_logs",
                                            group=group, config=config)

results = fit_multi_seed(
    ...,
    logger_factory=lambda seed: FanOutSeedLogger(wandb_f(seed), tb_f(seed)),
)
```

This is the standard "fan-out" pattern — one input call, N outputs. The
adapter isn't in the library because most users will pick one
dashboard, but it's small enough to copy when you want both.

## 9. Troubleshooting

### `wandb login` crashes with `TypeError: the JSON object must be str...`

Known bug in wandb 0.26.0: the client tries to print "Logged in as
<username>" after authenticating, hits a server response with
`flags=null`, and `json.loads(None)` raises. The credentials themselves
are fine. Workarounds:

- Skip `wandb login` entirely. The offline-mode workflow doesn't need
  it; `wandb sync` reads credentials from `~/.netrc` automatically.
- Or `pip install -U wandb` to a release where the bug is fixed.

### `entity not specified, and viewer has no default entity` on sync

WandB couldn't infer your namespace. Either pass `entity=` to your
`WandbSeedLogger` (cleanest) or pass `--entity <username>` to
`wandb sync`. If you've already synced once into the wrong place, remove
the `.synced` marker file and resync:

```bash
rm wandb/offline-run-*/run-*.wandb.synced
wandb sync --entity <username> wandb/offline-run-*
```

### Test metrics don't show on the time-series chart

They're scalars logged once at end-of-fit via `run.summary.update(...)`,
not via `run.log(...)`. Find them in the run's **Overview** tab and as
columns in the project run table. They are *not* on the Scalars chart
by design — there's no time axis for a single scalar.

### TB doesn't show the across-seed std

It doesn't — see §7. Read `summary.json` for the numbers.

## 10. Reference

- :doc:`fitter` — the `Fitter` and `fit_multi_seed` API contract.
- :doc:`metrics_paradigm` — what the val / test metrics mean.
- `deepSTRF.training.wandb_log` — `WandbSeedLogger`,
  `make_wandb_logger_factory`.
- `deepSTRF.training.tb_log` — `TensorBoardSeedLogger`,
  `make_tensorboard_logger_factory`.
- `deepSTRF.training.config` — `auto_config`, `slug_from_config`.
