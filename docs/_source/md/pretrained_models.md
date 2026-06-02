# Pretrained models

This page is the canonical registry of pretrained checkpoints we publish on
the [Hugging Face Hub](https://huggingface.co/). For each entry you'll find
the architecture, the training recipe, the dataset and split, the test
metrics we reported at upload time, and the deepSTRF commit SHA used to
train it — enough to reproduce the model from scratch if needed.

## Loading a pretrained model

Every `NeuralModel` subclass exposes the same one-liner, regardless of
where the checkpoint lives:

```python
from deepSTRF.models.audio import StateNet

model = StateNet.from_pretrained("urancon/deepSTRF-statenet-gru-ns1")
model.eval()
```

The first call resolves the repo, downloads the checkpoint (~few MB),
caches it under `~/.cache/huggingface/hub`, and reconstructs the model
from the saved `config.json`. Subsequent calls hit the cache.

Common kwargs:

```python
model = StateNet.from_pretrained(
    "urancon/deepSTRF-statenet-gru-ns1",
    map_location="cuda",            # or a torch.device
    revision="v1.0",                # pin a Hub branch / tag / commit SHA
    return_metadata=True,           # also get the metadata.json blob
)
```

If a model was trained with a non-JSON `__init__` argument (e.g. a custom
`prefiltering` `nn.Module`), re-supply it through `extra_kwargs`:

```python
from deepSTRF.models.prefiltering import make_prefiltering
model = SomeModel.from_pretrained(
    "owner/repo",
    extra_kwargs={"prefiltering": make_prefiltering("adaptrans", 34, dt=5.0)},
)
```

## Publishing your own checkpoint

Same shape as loading, in reverse. After training:

```python
model.push_to_hub(
    "your-username/your-model-name",
    metadata={"dataset": "...", "test_cc_norm": ...},
)
```

You need to be authenticated with write access — run `hf auth login` once.
The repo is created automatically on first push.

The library is happy to host community checkpoints; we just don't list
them on this page (which is the curated set we trained ourselves and
maintain). If you'd like yours added, open an issue with the `repo_id`
and a one-liner about the training recipe.

---

## Registry

### `urancon/deepSTRF-statenet-gru-ns1`

> StateNet GRU population model fit jointly on the 119 ferret A1 cells of
> NS1 (Harper et al. 2016, Rahman et al. 2020). Reference checkpoint for
> the canonical fit demo.

- **Hub repo:** <https://huggingface.co/urancon/deepSTRF-statenet-gru-ns1>
- **Model class:** [`StateNet`](https://github.com/urancon/deepSTRF/blob/develop/deepSTRF/models/audio/audio_zoo.py)
- **Architecture kwargs:**

  ```python
  StateNet(
      n_frequency_bands=34,
      hidden_channels=7,
      kernel_size=7, stride=3,
      connectivity="LC",
      rnn_type="GRU",
      out_neurons=119,    # population over all NS1 cells
  )
  ```
- **Trainable params:** ~39k.
- **Dataset:** NS1 (`deepSTRF.datasets.audio.ns1.NS1Dataset`),
  `dt_ms=5.0`, `smooth=True`, auto-downloaded from OSF + the DNet
  companion repo on first instantiation.
- **Split (by stim index, 0-indexed, 20 stims total):**
    - train: `[0..13]`  (14 stims)
    - val:   `[14, 15, 16]`
    - test:  `[17, 18, 19]`
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=0.0)`.
- **Loss:** `mse_loss(pred, gt_psth)` where `gt_psth = responses.nanmean(dim=2, keepdim=True)`.
- **Fitter config:** `max_epochs=80, patience=15, monitor="val_cc_norm", mode="max"`.
- **Random seed:** `0` (via `set_random_seed`).

**Test metrics** (held-out 3-stim split, recorded at upload time — see
`metadata.json` on the Hub for the full blob):

| Metric                          | Value  |
|---------------------------------|--------|
| `cc` (Pearson r, mean)          | +0.659 |
| `cc` (Pearson r, median)        | +0.681 |
| `cc_norm` (Schoppe, mean)       | +0.770 |
| `cc_norm` (Schoppe, median)     | +0.796 |
| MSE loss                        |  0.016 |

For orientation: published mean `cc_norm` for StateNet on NS1 is in the
0.7–0.8 range with leave-one-out training.

**Training cost:** ~5 min on a single mid-range GPU, 80 epochs.

**Trained at deepSTRF commit:** `c07e561f` (= the commit that introduced
this registry entry — pinning it reproduces the recipe exactly).

**Reproducing**

The canonical fit demo is `examples/fit_ns1_statenet.ipynb` (also rendered
on RTD), which runs the same recipe interactively. For the upload itself
we used a throwaway driver `_train_push_statenet_ns1.py` (gitignored — the
recipe above is the full record) which calls `model.push_to_hub(...)`
after training.

The deepSTRF commit SHA used at upload time is recorded in
`metadata.json` next to the checkpoint on the Hub
(`metadata["library_commit"]`); pinning that commit + the recipe above
should reproduce the published numbers within sampling noise.

---

*Last updated: 2026-05-05*
